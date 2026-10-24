"""Test doubles: a JSON-RPC node that hosts a LaplaceLedger, and an in-memory store."""

from __future__ import annotations

import json
from typing import Any

import httpx
from eth_abi.abi import decode, encode
from eth_account import Account
from eth_account.typed_transactions import TypedTransaction
from eth_hash.auto import keccak
from hexbytes import HexBytes

from laplace_engine.chain import COMMIT, COMMITTED, COMMITTER, LEDGER, LEDGER_BLOCK, ROOT_OF


class JsonRpcError(Exception):
    def __init__(self, message: str, data: str | None = None) -> None:
        super().__init__(message)
        self.data = data


class FakeNode:
    """Enough of a Robinhood Chain node for the engine: a ledger contract, blocks, transactions and logs."""

    def __init__(self, committer: str, *, head_timestamp: int, blocks: int = 4_000) -> None:
        """One block per second since the ledger's deployment; the head (where transactions land) is at
        ``head_timestamp``."""
        self.committer = committer
        self.head = LEDGER_BLOCK + blocks - 1
        self.timestamps = {self.head - i: head_timestamp - i for i in range(blocks)}
        self.roots: dict[int, bytes] = {}
        self.logs: list[dict[str, Any]] = []
        self.receipts: dict[str, dict[str, Any]] = {}
        self.sent: list[dict[str, Any]] = []
        self.nonce = 0
        self.revert: str | None = None
        self.rate_limited = 0  # answer this many requests with HTTP 429 first
        self.requests = 0

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        if self.rate_limited:
            self.rate_limited -= 1
            return httpx.Response(429)
        body = json.loads(request.content)
        reply: dict[str, Any] = {"jsonrpc": "2.0", "id": body["id"]}
        try:
            reply["result"] = getattr(self, body["method"])(*body["params"])
        except JsonRpcError as exc:
            reply["error"] = {"code": 3, "message": str(exc), "data": exc.data}
        return httpx.Response(200, json=reply)

    # -- eth_* -------------------------------------------------------------------------------------------------
    def eth_call(self, call: dict[str, str], block: str) -> str:
        data = bytes.fromhex(call["data"][2:])
        if data[:4] == ROOT_OF:
            (issue,) = decode(["uint64"], data[4:])
            return "0x" + self.roots.get(issue, bytes(32)).hex()
        if data[:4] == COMMITTER:
            return "0x" + encode(["address"], [self.committer]).hex()
        raise JsonRpcError("unknown function")

    def eth_estimateGas(self, call: dict[str, str]) -> str:
        if self.revert is not None:
            raise JsonRpcError("execution reverted", self.revert)
        return hex(51_000)

    def eth_blockNumber(self) -> str:
        return hex(self.head)

    def eth_getBlockByNumber(self, number: str, full: bool) -> dict[str, str]:
        n = self.head if number == "latest" else int(number, 16)
        return {"number": hex(n), "timestamp": hex(self.timestamps[n]), "baseFeePerGas": hex(20_000_000)}

    def eth_getTransactionCount(self, address: str, tag: str) -> str:
        return hex(self.nonce)

    def eth_getBalance(self, address: str, tag: str) -> str:
        return hex(10**17)

    def eth_sendRawTransaction(self, raw: str) -> str:
        tx = TypedTransaction.from_bytes(HexBytes(raw)).as_dict()
        tx["from"] = Account.recover_transaction(raw)
        data = bytes(tx["data"])
        assert data[:4] == COMMIT
        issue, root, manifest, leaves = decode(["uint64", "bytes32", "bytes32", "uint32"], data[4:])
        tx_hash = "0x" + bytes(keccak(bytes.fromhex(raw[2:]))).hex()
        self.roots[issue] = root
        self.logs.append(
            {
                "address": LEDGER,
                "blockNumber": hex(self.head),
                "transactionHash": tx_hash,
                "topics": ["0x" + COMMITTED.hex(), "0x" + issue.to_bytes(32, "big").hex()],
                "data": "0x" + encode(["bytes32", "bytes32", "uint32"], [root, manifest, leaves]).hex(),
            }
        )
        self.receipts[tx_hash] = {"status": "0x1", "blockNumber": hex(self.head)}
        self.sent.append(tx)
        self.nonce += 1
        return tx_hash

    def eth_getTransactionReceipt(self, tx_hash: str) -> dict[str, Any] | None:
        return self.receipts.get(tx_hash)

    def eth_getLogs(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        lo, hi = int(query["fromBlock"], 16), int(query["toBlock"], 16)
        return [e for e in self.logs if e["topics"] == query["topics"] and lo <= int(e["blockNumber"], 16) <= hi]


class FakeStore:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str, str]] = {}

    def put(self, key: str, body: bytes, *, content_type: str, cache_control: str) -> None:
        self.objects[key] = (body, content_type, cache_control)
