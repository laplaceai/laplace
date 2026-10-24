"""What a forecasting program sees: the past, and nothing else."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from laplace_research.evolve.feed import FloatArray, edges
from laplace_research.evolve.lib import LIB, Lib
from laplace_research.evolve.panel import Panel

LOOKBACK = 720
HOUR = 3600


def _readonly(a: FloatArray) -> FloatArray:
    a.flags.writeable = False
    return a


EDGES = {h: _readonly(edges(h)) for h in (1, 4, 24)}


@dataclass(frozen=True, slots=True)
class Candles:
    open: FloatArray
    high: FloatArray
    low: FloatArray
    close: FloatArray
    volume: FloatArray


@dataclass(frozen=True, slots=True)
class Context:
    """Read-only inputs for one forecast.

    Arrays hold the last ``LOOKBACK`` hourly candles, oldest first; the last element is the candle that has just
    closed. ``peers`` holds the same window for every feed asset (including this one), aligned hour by hour.
    """

    asset: str
    horizon: int
    edges: FloatArray
    hour: int
    weekday: int
    open: FloatArray
    high: FloatArray
    low: FloatArray
    close: FloatArray
    volume: FloatArray
    peers: dict[str, Candles]
    lib: Lib


def build_context(panel: Panel, asset_index: int, t: int, horizon: int) -> Context:
    if t + 1 < LOOKBACK:
        raise ValueError("not enough history for a context")
    window = slice(t + 1 - LOOKBACK, t + 1)
    closed_at = datetime.fromtimestamp(int(panel.ts[t]) + HOUR, UTC)
    peers = {
        name: Candles(
            panel.open[i, window],
            panel.high[i, window],
            panel.low[i, window],
            panel.close[i, window],
            panel.volume[i, window],
        )
        for i, name in enumerate(panel.assets)
    }
    own = peers[panel.assets[asset_index]]
    return Context(
        asset=panel.assets[asset_index],
        horizon=horizon,
        edges=EDGES[horizon],
        hour=closed_at.hour,
        weekday=closed_at.weekday(),
        open=own.open,
        high=own.high,
        low=own.low,
        close=own.close,
        volume=own.volume,
        peers=peers,
        lib=LIB,
    )
