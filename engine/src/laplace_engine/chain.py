"""Committing sealed issues to LaplaceLedger on Robinhood Chain, and reading commitments back.

The commit key signs one kind of transaction only: ``commit`` on the ledger. Before signing, the ledger is asked for
the issue's root, so a rerun after a crash recovers the existing commitment instead of sending a second transaction.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from itertools import count
from pathlib import Path
from typing import Any

import httpx
from eth_abi.abi import decode, encode
from eth_account import Account
from eth_account.signers.local import LocalAccount
from eth_hash.auto import keccak

from laplace_engine.release import canonical, hex32

CHAIN_ID = 4663
RPC_URL = "https://rpc.mainnet.chain.robinhood.com"
LEDGER = "0x15AaE63A0EF391559153bCF9c56E5E2BE916b2BE"
LEDGER_BLOCK = 96_984_063  # the ledger's deployment block; no commitment can be older
COMMIT_WINDOW_S = 15 * 60
LOW_BALANCE_WEI = 5 * 10**14  # 0.0005 ETH: about three weeks of hourly commits at today's fees

log = logging.getLogger(__name__)


def selector(signature: str) -> bytes:
    return bytes(keccak(signature.encode()))[:4]


COMMIT = selector("commit(uint64,bytes32,bytes32,uint32)")
ROOT_OF = selector("rootOf(uint64)")
COMMITTER = selector("committer()")
COMMITTED = bytes(keccak(b"Committed(uint64,bytes32,bytes32,uint32)"))
ERRORS = {
    selector(signature): signature.split("(")[0]
    for signature in (
        "NotCommitter()",
        "IssueNotAfterLast(uint64,uint64)",
        "OutsideCommitWindow(uint64,uint256)",
        "EmptyCommitment()",
    )
}


class RPCError(RuntimeError):
    def __init__(self, message: str, data: str | None = None) -> None:
        super().__init__(message)
        self.data = data


class LedgerRevert(RPCError):
    """The ledger refused the commitment; ``reason`` is the contract's custom error."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"the ledger refused the commitment: {reason}")
        self.reason = reason


class RPC:
    """A JSON-RPC client that retries transport errors, rate limits and server errors with backoff."""

    RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})

    def __init__(self, url: str = RPC_URL, *, client: httpx.Client | None = None, attempts: int = 5) -> None:
        self._url = url
        self._client = client or httpx.Client(timeout=httpx.Timeout(20.0, connect=10.0))
        self._attempts = attempts
        self._ids = count(1)

    def call(self, method: str, *params: Any) -> Any:
        payload = {"jsonrpc": "2.0", "id": next(self._ids), "method": method, "params": list(params)}
        for attempt in range(1, self._attempts + 1):
            try:
                response = self._client.post(self._url, json=payload)
            except httpx.TransportError as exc:
                if attempt == self._attempts:
                    raise RPCError(f"{method}: {exc!r}") from exc
            else:
                if response.status_code == httpx.codes.OK:
                    body = response.json()
                    error = body.get("error")
                    if error is None:
                        return body["result"]
                    if "rate" not in str(error.get("message", "")).lower() or attempt == self._attempts:
                        raise RPCError(f"{method}: {error.get('message')}", error.get("data"))
                elif response.status_code not in self.RETRY_STATUS or attempt == self._attempts:
                    raise RPCError(f"{method}: HTTP {response.status_code}")
            time.sleep(min(8.0, 0.25 * 2**attempt) * (0.5 + random.random()))
        raise AssertionError("unreachable")


@dataclass(frozen=True)
class Commitment:
    issue: int
    root: str
    manifest: str
    leaves: int
    tx: str
    block: int
    timestamp: int
    chain_id: int
    ledger: str

    def to_json(self) -> bytes:
        return canonical(asdict(self))

    @classmethod
    def from_json(cls, data: bytes) -> Commitment:
        return cls(**json.loads(data))

    @property
    def in_window(self) -> bool:
        opens = self.issue * 3600
        return opens <= self.timestamp < opens + COMMIT_WINDOW_S


def load_account(path: Path) -> LocalAccount:
    """The commit key from its JSON file (``privateKey``); the key itself is never logged."""
    account: LocalAccount = Account.from_key(json.loads(path.read_text())["privateKey"])
    return account


class Ledger:
    def __init__(self, rpc: RPC, address: str = LEDGER, chain_id: int = CHAIN_ID) -> None:
        self._rpc = rpc
        self.address = address
        self.chain_id = chain_id

    def _call(self, data: bytes) -> bytes:
        result: str = self._rpc.call("eth_call", {"to": self.address, "data": "0x" + data.hex()}, "latest")
        return bytes.fromhex(result[2:])

    def root_of(self, issue: int) -> bytes:
        return self._call(ROOT_OF + encode(["uint64"], [issue]))[:32]

    def committer(self) -> str:
        (address,) = decode(["address"], self._call(COMMITTER))
        return str(address)

    def balance(self, address: str) -> int:
        return int(self._rpc.call("eth_getBalance", address, "latest"), 16)

    def block_timestamp(self, number: int) -> int:
        return int(self._rpc.call("eth_getBlockByNumber", hex(number), False)["timestamp"], 16)

    def block_at(self, timestamp: int) -> int:
        """The last block at or before ``timestamp`` (binary search from the ledger's deployment)."""
        lo, hi = LEDGER_BLOCK, int(self._rpc.call("eth_blockNumber"), 16)
        if self.block_timestamp(hi) <= timestamp:
            return hi
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.block_timestamp(mid) <= timestamp:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def find(self, issue: int) -> Commitment | None:
        """The issue's ``Committed`` event, searched in the blocks of its commit window."""
        opens = issue * 3600
        first, last = self.block_at(opens), self.block_at(opens + COMMIT_WINDOW_S) + 1
        topics = ["0x" + COMMITTED.hex(), "0x" + issue.to_bytes(32, "big").hex()]
        logs = self._rpc.call(
            "eth_getLogs", {"address": self.address, "fromBlock": hex(first), "toBlock": hex(last), "topics": topics}
        )
        if not logs:
            return None
        event = logs[0]
        root, manifest, leaves = decode(["bytes32", "bytes32", "uint32"], bytes.fromhex(event["data"][2:]))
        block = int(event["blockNumber"], 16)
        return Commitment(
            issue,
            hex32(root),
            hex32(manifest),
            int(leaves),
            event["transactionHash"],
            block,
            self.block_timestamp(block),
            self.chain_id,
            self.address,
        )

    def commit(
        self,
        account: LocalAccount,
        issue: int,
        root: bytes,
        manifest_hash: bytes,
        leaves: int,
        *,
        timeout_s: float = 120.0,
    ) -> Commitment:
        """Commit an issue, or return its commitment if this exact root is committed already."""
        existing = self.root_of(issue)
        if existing == root:
            found = self.find(issue)
            if found is not None:
                log.info("issue %d was already committed in %s", issue, found.tx)
                return found
        elif existing != bytes(32):
            raise RPCError(f"issue {issue} is committed with another root, 0x{existing.hex()}")
        if self.committer().lower() != account.address.lower():
            raise ValueError("this key is not the ledger's committer")

        data = COMMIT + encode(["uint64", "bytes32", "bytes32", "uint32"], [issue, root, manifest_hash, leaves])
        call = {"from": account.address, "to": self.address, "data": "0x" + data.hex()}
        try:
            gas = int(self._rpc.call("eth_estimateGas", call), 16)
        except RPCError as exc:
            reason = ERRORS.get(bytes.fromhex((exc.data or "0x")[2:10]))
            raise (LedgerRevert(reason) if reason else exc) from exc
        base_fee = int(self._rpc.call("eth_getBlockByNumber", "latest", False)["baseFeePerGas"], 16)
        tx: dict[str, Any] = {
            "type": 2,
            "chainId": self.chain_id,
            "nonce": int(self._rpc.call("eth_getTransactionCount", account.address, "pending"), 16),
            "to": self.address,
            "value": 0,
            "data": data,
            "gas": gas * 13 // 10,
            "maxFeePerGas": 2 * base_fee,
            "maxPriorityFeePerGas": 0,
        }
        signed = account.sign_transaction(tx)
        tx_hash: str = self._rpc.call("eth_sendRawTransaction", "0x" + signed.raw_transaction.hex())
        receipt = self._receipt(tx_hash, timeout_s)
        if int(receipt["status"], 16) != 1:
            raise RPCError(f"commit transaction {tx_hash} reverted")
        block = int(receipt["blockNumber"], 16)
        commitment = Commitment(
            issue,
            hex32(root),
            hex32(manifest_hash),
            leaves,
            tx_hash,
            block,
            self.block_timestamp(block),
            self.chain_id,
            self.address,
        )
        if not commitment.in_window:
            raise RPCError(
                f"commit {tx_hash} was mined outside the issue's window"
            )  # the contract makes this unreachable
        return commitment

    def _receipt(self, tx_hash: str, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while True:
            receipt: dict[str, Any] | None = self._rpc.call("eth_getTransactionReceipt", tx_hash)
            if receipt is not None:
                return receipt
            if time.monotonic() > deadline:
                raise RPCError(f"no receipt for {tx_hash} after {timeout_s:.0f}s")
            time.sleep(0.5)
