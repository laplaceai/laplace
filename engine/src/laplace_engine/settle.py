"""Settling issues and scoring their forecasts from the committed record.

A question of issue N at horizon h settles on the close of the hourly candle that ends at the start of issue N + h.
That candle is issue N + h's last input candle, which its manifest carries and the chain commits, so every score of
issue N at horizon h normally follows from committed files alone. The starting price is issue N's own last input
candle. If issue N + h was never committed, the question settles on the exchange's public candle instead, so a missing
hour can never hide an outcome. Each forecast is scored with the binned log score of its rounded probabilities.

The scoreboard aggregates every settlement: each forecaster's mean log score, and its paired gain over each baseline
with a circular block-bootstrap interval over issues.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from laplace_engine.quantize import TOTAL
from laplace_engine.release import canonical
from laplace_research.evolve.feed import HORIZONS, bin_index
from laplace_research.scoring import circular_block_indices, interval

CLOSE = 3  # position of the close in a manifest's last_candle entries (open, high, low, close, volume)
SCHEMA = "laplace/settlement/1"
BOARD_SCHEMA = "laplace/scoreboard/1"
BLOCK_ISSUES = 72  # bootstrap blocks of 72 hourly issues, as in research
BASELINES = ("naive", "garch", "har")

FetchClose = Callable[[str, int], float]  # (asset, candle open time) -> close


def settlement_name(horizon: int) -> str:
    return f"settlement-{horizon}h.json"


def closes(manifest: dict[str, Any]) -> dict[str, float]:
    return {a: float(c[CLOSE]) for a, c in manifest["inputs"]["last_candle"].items()}


@dataclass(frozen=True)
class Settlement:
    issue: int
    horizon: int
    source: str  # "issue <n>" when the settling price is committed, else "coinbase"
    start: dict[str, float]
    end: dict[str, float]
    bins: dict[str, int]
    scores: dict[str, dict[str, float]]  # forecaster -> asset -> log score in nats

    def to_json(self) -> bytes:
        returns = {a: math.log(self.end[a] / self.start[a]) for a in self.start}
        return canonical(
            {
                "schema": SCHEMA,
                "issue": self.issue,
                "horizon": self.horizon,
                "source": self.source,
                "start": self.start,
                "end": self.end,
                "returns": returns,
                "bins": self.bins,
                "scores": self.scores,
            }
        )

    @classmethod
    def from_json(cls, data: bytes) -> Settlement:
        b = json.loads(data)
        return cls(b["issue"], b["horizon"], b["source"], b["start"], b["end"], b["bins"], b["scores"])


def settle(
    manifest: dict[str, Any], forecasts: dict[str, Any], horizon: int, end: dict[str, float], source: str
) -> Settlement:
    """Score an issue's forecasts at ``horizon`` given the settling closes ``end``."""
    assets = manifest["feed"]["assets"]
    if horizon not in manifest["feed"]["horizons"] or set(end) != set(assets):
        raise ValueError("the settling prices do not match the issue's questions")
    start = closes(manifest)
    bins = {a: int(bin_index(np.array([math.log(end[a] / start[a])]), horizon)[0]) for a in assets}
    scores: dict[str, dict[str, float]] = {}
    for f in forecasts["forecasts"]:
        if f["horizon"] == horizon:
            scores.setdefault(f["forecaster"], {})[f["asset"]] = math.log(f["bps"][bins[f["asset"]]] / TOTAL)
    return Settlement(manifest["issue"], horizon, source, start, end, bins, scores)


def _committed(issue_dir: Path) -> bool:
    return (issue_dir / "manifest.json").exists() and (issue_dir / "commitment.json").exists()


def settle_due(issues_dir: Path, current: int, fetch_close: FetchClose, *, lookback: int = 72) -> list[Path]:
    """Write every settlement that has become due for committed issues of the last ``lookback`` hours.

    Issue N at horizon h is due once issue N + h has started. It settles on issue N + h's committed prices if N + h is
    committed, on the exchange's candle if N + h is in the past and uncommitted, and waits if N + h is the current issue
    and not committed yet (its commit may still be in flight).
    """
    written = []
    for issue in range(current - lookback, current):
        issue_dir = issues_dir / str(issue)
        if not _committed(issue_dir):
            continue
        manifest = json.loads((issue_dir / "manifest.json").read_bytes())
        forecasts: dict[str, Any] | None = None
        for h in manifest["feed"]["horizons"]:
            target, later = issue_dir / settlement_name(h), issues_dir / str(issue + h)
            if target.exists() or issue + h > current:
                continue
            if _committed(later):
                end, source = closes(json.loads((later / "manifest.json").read_bytes())), f"issue {issue + h}"
            elif issue + h < current:
                end = {a: fetch_close(a, (issue + h - 1) * 3600) for a in manifest["feed"]["assets"]}
                source = "coinbase"
            else:
                continue
            forecasts = forecasts or json.loads((issue_dir / "forecasts.json").read_bytes())
            target.write_bytes(settle(manifest, forecasts, h, end, source).to_json())
            written.append(target)
    return written


def load_settlements(issues_dir: Path) -> list[Settlement]:
    paths = sorted(issues_dir.glob("*/settlement-*h.json"))
    return [Settlement.from_json(p.read_bytes()) for p in paths]


def scoreboard(settlements: Sequence[Settlement], *, baselines: Iterable[str] = BASELINES) -> dict[str, Any]:
    """Mean log scores per forecaster (overall and by horizon) and paired gains over each baseline, with 95% circular
    block-bootstrap intervals over issues. Only questions both sides forecast are compared."""
    scores: dict[str, dict[tuple[int, int, str], float]] = defaultdict(dict)  # forecaster -> (issue, h, asset) -> score
    for s in settlements:
        for name, by_asset in s.scores.items():
            for asset, value in by_asset.items():
                scores[name][(s.issue, s.horizon, asset)] = value
    board: dict[str, Any] = {
        "schema": BOARD_SCHEMA,
        "issues": sorted({s.issue for s in settlements}),
        "block_issues": BLOCK_ISSUES,
        "forecasters": {},
    }
    board["first_issue"] = board["issues"][0] if board["issues"] else None
    board["last_issue"] = board["issues"][-1] if board["issues"] else None
    board["settled_issues"] = len(board.pop("issues"))
    for name, by_question in sorted(scores.items()):
        entry: dict[str, Any] = {
            "questions": len(by_question),
            "mean_log_score": fmean(by_question.values()),
            "by_horizon": {
                str(h): fmean(v for (_, hh, _), v in by_question.items() if hh == h)
                for h in HORIZONS
                if any(hh == h for (_, hh, _) in by_question)
            },
            "vs": {},
        }
        for base in baselines:
            if base == name or base not in scores:
                continue
            per_issue: dict[int, list[float]] = defaultdict(list)
            for key, value in by_question.items():
                if key in scores[base]:
                    per_issue[key[0]].append(value - scores[base][key])
            series = [fmean(per_issue[i]) for i in sorted(per_issue)]
            if not series:
                continue
            # Fewer than two blocks of issues cannot support an interval; report none rather than a false certainty.
            ci = None
            if len(series) >= 2 * BLOCK_ISSUES:
                ci = list(interval(series, circular_block_indices(len(series), BLOCK_ISSUES)))
            entry["vs"][base] = {"mean": fmean(series), "interval95": ci, "issues": len(series)}
        board["forecasters"][name] = entry
    return board
