from collections.abc import Callable
from pathlib import Path

import pytest

from laplace_engine import candles, cli, release
from laplace_engine.baselines import NAIVE
from laplace_engine.release import Pool, Program, Release
from laplace_research.evolve.panel import Panel

from .conftest import ISSUE


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
    assert cli.main(["verify", *data]) == 0
    assert "63 forecasts and 1 pool(s) check out" in capsys.readouterr().out
