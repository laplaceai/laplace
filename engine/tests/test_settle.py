import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from laplace_engine.settle import BLOCK_ISSUES, Settlement, load_settlements, scoreboard, settle, settle_due

ASSETS = ["BTC", "ETH"]
BPS = [100, 400, 900, 1_600, 2_000, 2_000, 1_600, 900, 400, 100]


def manifest(issue: int, closes: dict[str, float]) -> dict[str, Any]:
    candles = {a: [c, c, c, c, 1.0] for a, c in closes.items()}
    return {"issue": issue, "feed": {"assets": ASSETS, "horizons": [1, 4, 24]}, "inputs": {"last_candle": candles}}


def forecasts(issue: int) -> dict[str, Any]:
    rows = [
        {"forecaster": n, "asset": a, "horizon": h, "bps": BPS}
        for n in ("demon", "naive")
        for a in ASSETS
        for h in (1, 4, 24)
    ]
    return {"issue": issue, "forecasts": rows}


def test_scores_the_bin_the_return_fell_in() -> None:
    s = settle(manifest(100, {"BTC": 100.0, "ETH": 100.0}), forecasts(100), 1, {"BTC": 100.5, "ETH": 90.0}, "issue 101")
    assert s.bins == {"BTC": 6, "ETH": 0}  # +0.50% is in [0.3%, 0.7%); -10.5% is below -3%
    assert s.scores["demon"]["BTC"] == pytest.approx(math.log(0.16))
    assert s.scores["naive"]["ETH"] == pytest.approx(math.log(0.01))
    assert Settlement.from_json(s.to_json()) == s
    assert json.loads(s.to_json())["returns"]["BTC"] == pytest.approx(math.log(1.005))


def write_issue(issues: Path, issue: int, close: float, *, committed: bool = True) -> None:
    d = issues / str(issue)
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(manifest(issue, {"BTC": close, "ETH": close})))
    (d / "forecasts.json").write_text(json.dumps(forecasts(issue)))
    if committed:
        (d / "commitment.json").write_text("{}")


def test_settles_on_committed_prices_and_falls_back_to_the_exchange_for_gaps(tmp_path: Path) -> None:
    for issue, close in ((100, 100.0), (101, 101.0), (110, 100.0)):
        write_issue(tmp_path, issue, close)
    write_issue(tmp_path, 104, 99.0, committed=False)  # sealed but never committed: its prices are not used
    fetched = []

    def fetch_close(asset: str, open_time: int) -> float:
        fetched.append((asset, open_time))
        return 98.0

    written = settle_due(tmp_path, 110, fetch_close)
    assert sorted(p.relative_to(tmp_path).as_posix() for p in written) == [
        "100/settlement-1h.json",
        "100/settlement-4h.json",
        "101/settlement-1h.json",
        "101/settlement-4h.json",
    ]
    one = Settlement.from_json((tmp_path / "100" / "settlement-1h.json").read_bytes())
    assert one.source == "issue 101"
    assert one.end == {"BTC": 101.0, "ETH": 101.0}
    four = Settlement.from_json((tmp_path / "100" / "settlement-4h.json").read_bytes())
    assert four.source == "coinbase"
    assert four.end["BTC"] == 98.0
    assert ("BTC", 103 * 3600) in fetched
    assert settle_due(tmp_path, 110, fetch_close) == []  # 24h is not due yet; nothing is settled twice


def test_waits_for_the_current_issue_to_commit(tmp_path: Path) -> None:
    write_issue(tmp_path, 109, 100.0)
    write_issue(tmp_path, 110, 100.0, committed=False)
    assert settle_due(tmp_path, 110, lambda asset, t: pytest.fail("must not fetch")) == []


def test_scoreboard_pairs_forecasters_with_baselines_by_question() -> None:
    rng = np.random.default_rng(0)
    settlements = []
    for issue in range(1_000, 1_000 + 2 * BLOCK_ISSUES):
        naive = {a: float(rng.normal(-2.0, 0.5)) for a in ASSETS}
        demon = {a: naive[a] + 0.05 for a in ASSETS}
        settlements.append(Settlement(issue, 1, "test", {}, {}, {}, {"demon": demon, "naive": naive}))
    board = scoreboard(settlements)
    demon = board["forecasters"]["demon"]
    assert demon["questions"] == 2 * 2 * BLOCK_ISSUES
    assert demon["vs"]["naive"]["mean"] == pytest.approx(0.05)
    lo, hi = demon["vs"]["naive"]["interval95"]
    assert lo == pytest.approx(0.05)
    assert hi == pytest.approx(0.05)
    assert "garch" not in demon["vs"]
    early = scoreboard(settlements[:BLOCK_ISSUES])
    assert early["forecasters"]["demon"]["vs"]["naive"]["interval95"] is None  # too few issues for an interval
    assert early["settled_issues"] == BLOCK_ISSUES


def test_load_settlements_reads_every_file(tmp_path: Path) -> None:
    for issue in (100, 101, 102):
        write_issue(tmp_path, issue, 100.0)
    settle_due(tmp_path, 102, lambda asset, t: 100.0)
    assert {(s.issue, s.horizon) for s in load_settlements(tmp_path)} == {(100, 1), (101, 1)}
