import json
import time
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from eth_account import Account

from laplace_engine import candles, chain, cli, publish, release
from laplace_engine.baselines import NAIVE
from laplace_engine.chain import RPC
from laplace_engine.release import Pool, Program, Release
from laplace_research.evolve.panel import Panel

from .conftest import ISSUE
from .fakes import FakeNode, FakeStore


def test_issue_then_verify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_panel: Callable[..., Panel],
    capsys: pytest.CaptureFixture[str],
) -> None:
    rel = Release(
        "t1",
        "test",
        (Program("naive", NAIVE, True, "baseline"), Program("double", NAIVE + "\n", False, "test#2")),
        (Pool("demon", (("naive", 500_000), ("double", 500_000))),),
    )
    release.write(rel, tmp_path / "release", tmp_path / "private")
    monkeypatch.setattr(candles, "wait_for", lambda issue, cache_dir, deadline: make_panel(end_issue=issue))
    common = ["--release", str(tmp_path / "release"), "--private", str(tmp_path / "private")]
    data = ["--data", str(tmp_path / "data"), "--issue", str(ISSUE)]

    assert cli.main(["issue", *common, *data, "--workers", "2"]) == 0
    assert "63 forecasts" in capsys.readouterr().out
    assert (tmp_path / "data" / "issues" / str(ISSUE) / "manifest.json").exists()
    assert cli.main(["issue", *common, *data]) == 0  # sealed already: nothing is recomputed
    assert cli.main(["verify", *data, "--offline"]) == 0
    assert "63 forecasts and 1 pool(s) check out" in capsys.readouterr().out


def test_commit_and_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_panel: Callable[..., Panel],
    capsys: pytest.CaptureFixture[str],
) -> None:
    account = Account.create()
    node = FakeNode(account.address, head_timestamp=int(time.time()))
    store = FakeStore()
    rel = Release("r9", "test", (Program("naive", NAIVE, True, "baseline"),), ())
    release.write(rel, tmp_path / "r9", tmp_path / "private")
    number = candles.current_issue()
    monkeypatch.setattr(candles, "wait_for", lambda issue, cache_dir, deadline: make_panel(end_issue=issue))
    monkeypatch.setattr(chain, "COMMIT_WINDOW_S", 3600)  # the test may run at any minute of the hour
    monkeypatch.setattr(chain, "load_account", lambda path: account)
    monkeypatch.setattr(chain, "RPC", lambda url: RPC(url, client=httpx.Client(transport=node.transport())))
    monkeypatch.setattr(publish.R2, "from_env", classmethod(lambda cls: store))
    monkeypatch.setenv("LAPLACE_COMMIT_KEY_FILE", str(tmp_path / "unused.json"))
    argv = ["issue", "--release", str(tmp_path / "r9"), "--private", str(tmp_path / "private")]
    argv += ["--data", str(tmp_path / "data"), "--workers", "1", "--commit", "--publish"]

    assert cli.main(argv) == 0
    assert len(node.sent) == 1
    assert "committed in 0x" in capsys.readouterr().out
    assert json.loads(store.objects["v1/latest.json"][0])["issue"] == number
    assert f"v1/issues/{number}/commitment.json" in store.objects
    assert "v1/releases/r9/programs/naive.py" in store.objects
    assert cli.main(argv) == 0  # a rerun neither seals, commits nor publishes again
    assert len(node.sent) == 1

    assert cli.main(["status", "--data", str(tmp_path / "data"), "--hours", "2"]) == 0
    assert capsys.readouterr().out.splitlines()[-1].endswith("sealed committed published")


def test_a_closed_window_leaves_the_issue_uncommitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_panel: Callable[..., Panel]
) -> None:
    rel = Release("r9", "test", (Program("naive", NAIVE, True, "baseline"),), ())
    release.write(rel, tmp_path / "r9", tmp_path / "private")
    monkeypatch.setattr(candles, "wait_for", lambda issue, cache_dir, deadline: make_panel(end_issue=issue))
    argv = ["issue", "--release", str(tmp_path / "r9"), "--private", str(tmp_path / "private")]
    argv += ["--data", str(tmp_path / "data"), "--issue", str(ISSUE), "--workers", "1", "--commit"]
    assert cli.main(argv) == 1
    assert not (tmp_path / "data" / "issues" / str(ISSUE) / "commitment.json").exists()
