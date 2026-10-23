"""Blinded forecast cards for the Majors league, and how they settle.

A card describes one asset at one hour using only relative numbers (returns, volatility, range position, volume
ratio). It carries no ticker, price level or date, so a model can't answer from memory of what happened next.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import pstdev
from typing import Any

from laplace_research.data.candles import Candle

WEEK = 168

# 24-hour move bins: (id, lower bound, upper bound) on the simple return, bounds in fractions.
MOVE_BINS: tuple[tuple[str, float, float], ...] = (
    ("lt_m3", -math.inf, -0.03),
    ("m3_m1", -0.03, -0.01),
    ("m1_p1", -0.01, 0.01),
    ("p1_p3", 0.01, 0.03),
    ("gt_p3", 0.03, math.inf),
)
MOVE_LABELS: dict[str, str] = {
    "lt_m3": "down more than 3%",
    "m3_m1": "down between 1% and 3%",
    "m1_p1": "within 1% either way",
    "p1_p3": "up between 1% and 3%",
    "gt_p3": "up more than 3%",
}
BIG_MOVE_4H = 0.02


@dataclass(frozen=True, slots=True)
class Card:
    card_id: str
    asset: str
    ts: int
    features: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Outcome:
    move24: str
    big4: bool


def move_bin(ret: float) -> str:
    for bin_id, lo, hi in MOVE_BINS:
        if lo <= ret < hi:
            return bin_id
    raise ValueError(f"return {ret} fits no bin")


def hourly_log_returns(closes: Sequence[float]) -> list[float]:
    return [math.log(b / a) for a, b in itertools.pairwise(closes)]


def build_card(asset: str, alias: str, candles: Sequence[Candle], i: int) -> Card:
    """Card for the close of ``candles[i]``, using only candles up to and including ``i``."""
    if i < WEEK:
        raise ValueError("a card needs a week of history")
    window = candles[i - WEEK : i + 1]
    closes = [c.close for c in window]
    now = closes[-1]
    lr = hourly_log_returns(closes)
    hi, lo = max(c.high for c in window), min(c.low for c in window)
    vol_24h = sum(c.volume for c in window[-24:])
    vol_week_daily = sum(c.volume for c in window[1:]) / 7

    def ret(hours: int) -> float:
        return round(100 * (now / closes[-1 - hours] - 1), 3)

    features: dict[str, Any] = {
        "asset": f"Asset {alias}, a large-cap crypto asset traded around the clock",
        "returns_pct": {"1h": ret(1), "4h": ret(4), "24h": ret(24), "7d": ret(WEEK)},
        "hourly_returns_pct_last_24h": [round(100 * x, 3) for x in lr[-24:]],
        "realized_vol_pct": {
            "hourly_7d": round(100 * pstdev(lr), 3),
            "per_24h_from_last_24h": round(100 * pstdev(lr[-24:]) * math.sqrt(24), 3),
            "per_24h_from_last_7d": round(100 * pstdev(lr) * math.sqrt(24), 3),
        },
        "position_in_7d_range": round((now - lo) / (hi - lo), 3) if hi > lo else 0.5,
        "volume_24h_vs_7d_daily_avg": round(vol_24h / vol_week_daily, 3) if vol_week_daily else None,
    }
    return Card(card_id=f"{asset}:{candles[i].ts}", asset=asset, ts=candles[i].ts, features=features)


def settle(candles: Sequence[Candle], i: int) -> Outcome:
    """Outcome of the card at ``candles[i]``: closes 4 and 24 hours later."""
    now = candles[i].close
    return Outcome(
        move24=move_bin(candles[i + 24].close / now - 1),
        big4=abs(candles[i + 4].close / now - 1) > BIG_MOVE_4H,
    )
