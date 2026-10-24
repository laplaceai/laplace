import httpx
import pytest
from eth_abi.abi import encode
from eth_account import Account
from eth_account.signers.local import LocalAccount

from laplace_engine.chain import COMMIT, ERRORS, LEDGER, LEDGER_BLOCK, RPC, Ledger, LedgerRevert, RPCError, selector

from .conftest import ISSUE
from .fakes import FakeNode

ROOT, MANIFEST = b"\x11" * 32, b"\x22" * 32


def setup() -> tuple[FakeNode, Ledger, LocalAccount]:
    """A ledger on a fake node whose latest block is two minutes into the issue's commit window."""
    account: LocalAccount = Account.create()
    node = FakeNode(account.address, head_timestamp=ISSUE * 3600 + 120)
    rpc = RPC("https://rpc.test", client=httpx.Client(transport=node.transport()))
    return node, Ledger(rpc), account


def test_commit_signs_one_commit_transaction_to_the_ledger() -> None:
    node, ledger, account = setup()
    commitment = ledger.commit(account, ISSUE, ROOT, MANIFEST, 273)
    assert len(node.sent) == 1
    tx = node.sent[0]
    assert tx["from"] == account.address
    assert bytes(tx["to"]) == bytes.fromhex(LEDGER[2:])
    assert tx["chainId"] == 4663
    assert tx["value"] == 0
    assert tx["maxPriorityFeePerGas"] == 0
    assert bytes(tx["data"]) == COMMIT + encode(
        ["uint64", "bytes32", "bytes32", "uint32"], [ISSUE, ROOT, MANIFEST, 273]
    )
    assert tx["gas"] == 51_000 * 13 // 10
    assert commitment.root == "0x" + ROOT.hex()
    assert commitment.manifest == "0x" + MANIFEST.hex()
    assert commitment.leaves == 273
    assert commitment.in_window


def test_a_rerun_recovers_the_commitment_instead_of_sending_again() -> None:
    node, ledger, account = setup()
    first = ledger.commit(account, ISSUE, ROOT, MANIFEST, 273)
    again = ledger.commit(account, ISSUE, ROOT, MANIFEST, 273)
    assert again == first
    assert len(node.sent) == 1
    assert ledger.find(ISSUE) == first


def test_refuses_to_commit_over_another_root_or_with_another_key() -> None:
    node, ledger, account = setup()
    node.roots[ISSUE] = b"\x33" * 32
    with pytest.raises(RPCError, match="another root"):
        ledger.commit(account, ISSUE, ROOT, MANIFEST, 273)
    node.roots.clear()
    node.committer = Account.create().address
    with pytest.raises(ValueError, match="committer"):
        ledger.commit(account, ISSUE, ROOT, MANIFEST, 273)
    assert not node.sent


def test_contract_errors_are_named() -> None:
    node, ledger, account = setup()
    window = selector("OutsideCommitWindow(uint64,uint256)")
    assert ERRORS[window] == "OutsideCommitWindow"
    node.revert = "0x" + (window + encode(["uint64", "uint256"], [ISSUE, ISSUE * 3600 + 900])).hex()
    with pytest.raises(LedgerRevert) as raised:
        ledger.commit(account, ISSUE, ROOT, MANIFEST, 273)
    assert raised.value.reason == "OutsideCommitWindow"
    assert not node.sent


def test_rpc_retries_rate_limits_then_gives_up() -> None:
    node, ledger, _ = setup()
    node.rate_limited = 2
    assert ledger.root_of(ISSUE) == bytes(32)
    impatient = Ledger(RPC("https://rpc.test", client=httpx.Client(transport=node.transport()), attempts=2))
    node.rate_limited = 5
    with pytest.raises(RPCError, match="429"):
        impatient.root_of(ISSUE)


def test_block_at_is_the_last_block_at_or_before_a_time() -> None:
    node, ledger, _ = setup()
    t = node.timestamps[LEDGER_BLOCK + 1_234]
    assert ledger.block_at(t) == LEDGER_BLOCK + 1_234
    assert ledger.block_at(10**12) == node.head
    assert ledger.find(ISSUE + 1) is None
