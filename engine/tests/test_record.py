import json

import pytest
from eth_abi.abi import encode
from eth_hash.auto import keccak

from laplace_engine import vectors
from laplace_engine.record import Leaf, Tree, verify

BPS = (100, 400, 900, 1_600, 2_000, 2_000, 1_600, 900, 400, 100)


def leaf(n: int) -> Leaf:
    """The n-th of 21 distinct questions (7 assets x 3 horizons), for forecaster n // 21."""
    forecaster = bytes(keccak(b"forecaster %d" % (n // 21)))
    return Leaf(issue=497_983, forecaster=forecaster, asset=n % 7, horizon=(1, 4, 24)[n // 7 % 3], bps=BPS)


def test_leaf_hash_is_the_double_keccak_of_the_abi_encoding() -> None:
    item = leaf(0)
    encoded = encode(["uint64", "bytes32", "uint8", "uint16", "uint16[10]"], [497_983, item.forecaster, 0, 1, BPS])
    assert len(encoded) == 14 * 32
    assert item.hash() == keccak(keccak(encoded))


@pytest.mark.parametrize("n", [1, 2, 3, 7, 8, 21, 50, 273])
def test_every_proof_verifies_and_nothing_else_does(n: int) -> None:
    items = [Leaf(i + 1, bytes(keccak(b"f")), i % 7, 1 + i % 3, BPS) for i in range(n)]
    tree = Tree.build(x.hash() for x in items)
    outsider = Leaf(10_000, bytes(keccak(b"f")), 0, 1, BPS).hash()
    for item in items:
        proof = tree.proof(item.hash())
        assert verify(proof, tree.root, item.hash())
        assert not verify(proof, tree.root, outsider)
    assert len(tree.proof(items[0].hash())) <= max(1, (n - 1).bit_length())


def test_root_does_not_depend_on_leaf_order() -> None:
    hashes = [leaf(n).hash() for n in range(30)]
    assert Tree.build(hashes).root == Tree.build(reversed(hashes)).root


def test_rejects_duplicates_and_malformed_leaves() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        Tree.build([leaf(0).hash(), leaf(0).hash()])
    with pytest.raises(ValueError, match="at least one"):
        Tree.build([])
    with pytest.raises(ValueError, match="bps"):
        Leaf(1, bytes(32), 0, 1, (1_000,) * 9 + (999,))
    with pytest.raises(ValueError, match="bps"):
        Leaf(1, bytes(32), 0, 1, (0, *BPS[1:-1], 200))
    with pytest.raises(ValueError, match="range"):
        Leaf(1, bytes(31), 0, 1, BPS)


def test_contract_vectors_are_current() -> None:
    # The Foundry tests check these vectors against LaplaceLedger; regenerate with `python -m laplace_engine.vectors`.
    assert json.loads(vectors.PATH.read_text()) == vectors.build()
