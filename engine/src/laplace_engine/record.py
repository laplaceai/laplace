"""The live record's format: forecast leaves and the Merkle tree whose root is committed on-chain every hour.

A leaf is one forecast: which issue, which forecaster, which question, and the ten basis-point probabilities. Its hash
is ``keccak256(bytes.concat(keccak256(abi.encode(issue, forecaster, asset, horizon, bps))))`` with the Solidity types
``(uint64, bytes32, uint8, uint16, uint16[10])``, exactly as ``LaplaceLedger.leafHash`` computes it; hashing twice
keeps a leaf from ever being mistaken for an inner node. Inner nodes hash the sorted pair of their children
(OpenZeppelin's ``MerkleProof`` convention), so a proof is the list of sibling hashes and any contract can check one
against the committed root with ``LaplaceLedger.verifyForecast``.

- ``issue``: the hour the forecasts are made, in hours since the Unix epoch; candles up to that hour have closed.
- ``forecaster``: keccak256 of the forecaster's definition (a program's source, or a pool's specification), so every
  leaf commits to exactly what produced it.
- ``asset``: index into the append-only registry ``ASSETS``; ``horizon``: hours ahead.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from eth_abi.abi import encode
from eth_hash.auto import keccak

from laplace_engine.quantize import TOTAL
from laplace_research.evolve.feed import N_BINS

LEAF_TYPES = ("uint64", "bytes32", "uint8", "uint16", "uint16[10]")


@dataclass(frozen=True, slots=True)
class Leaf:
    issue: int
    forecaster: bytes
    asset: int
    horizon: int
    bps: tuple[int, ...]

    def __post_init__(self) -> None:
        if not 0 < self.issue < 2**64 or len(self.forecaster) != 32 or not 0 <= self.asset < 256:
            raise ValueError("leaf field out of range")
        if not 0 < self.horizon < 2**16:
            raise ValueError("horizon out of range")
        if len(self.bps) != N_BINS or sum(self.bps) != TOTAL or min(self.bps) < 1:
            raise ValueError("bps must be ten positive integers summing to 10,000")

    def hash(self) -> bytes:
        fields = [self.issue, self.forecaster, self.asset, self.horizon, list(self.bps)]
        return bytes(keccak(keccak(encode(LEAF_TYPES, fields))))


def hash_pair(a: bytes, b: bytes) -> bytes:
    return bytes(keccak(a + b if a < b else b + a))


@dataclass(frozen=True)
class Tree:
    """Levels from the sorted leaf hashes up to the root; an odd node at the end of a level is carried up as is."""

    levels: tuple[tuple[bytes, ...], ...]

    @classmethod
    def build(cls, leaf_hashes: Iterable[bytes]) -> Tree:
        level = tuple(sorted(leaf_hashes))
        if not level:
            raise ValueError("a tree needs at least one leaf")
        if len(set(level)) != len(level):
            raise ValueError("duplicate leaves")
        levels = [level]
        while len(level) > 1:
            pairs = range(0, len(level) - 1, 2)
            level = tuple(hash_pair(level[i], level[i + 1]) for i in pairs) + level[len(level) - len(level) % 2 :]
            levels.append(level)
        return cls(tuple(levels))

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    def proof(self, leaf_hash: bytes) -> list[bytes]:
        index = self.levels[0].index(leaf_hash)
        siblings = []
        for level in self.levels[:-1]:
            if index ^ 1 < len(level):
                siblings.append(level[index ^ 1])
            index //= 2
        return siblings


def verify(proof: Sequence[bytes], root: bytes, leaf_hash: bytes) -> bool:
    node = leaf_hash
    for sibling in proof:
        node = hash_pair(node, sibling)
    return node == root
