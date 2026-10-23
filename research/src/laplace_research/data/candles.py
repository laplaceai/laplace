"""Hourly candles from the Coinbase Exchange public API, cached on disk.

Only public, unauthenticated endpoints are used. Candles are cached per product and day so a rerun of an experiment
is reproducible and does not hit the API again.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

API = "https://api.exchange.coinbase.com"
HOUR = 3600
MAX_CANDLES_PER_REQUEST = 300


@dataclass(frozen=True, slots=True)
class Candle:
    """One hourly candle. ``ts`` is the candle's open time (unix seconds, UTC)."""

    ts: int
    low: float
    high: float
    open: float
    close: float
    volume: float


def _parse(rows: list[list[float]]) -> list[Candle]:
    # Coinbase rows are [time, low, high, open, close, volume], newest first.
    return sorted((Candle(int(r[0]), r[1], r[2], r[3], r[4], r[5]) for r in rows), key=lambda c: c.ts)


def fetch_hourly(
    product: str,
    start: datetime,
    end: datetime,
    *,
    client: httpx.Client,
    cache_dir: Path,
) -> list[Candle]:
    """Return hourly candles for ``product`` in ``[start, end)``, gap-free or raising ``ValueError``."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    candles: dict[int, Candle] = {}
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while day < end:
        path = cache_dir / f"{product}-{day:%Y%m%d}.json"
        complete_day = day + timedelta(days=1) <= datetime.now(UTC) - timedelta(hours=1)
        if path.exists():
            rows = json.loads(path.read_text())
        else:
            rows = _get(client, product, day, day + timedelta(days=1))
            if complete_day:
                path.write_text(json.dumps(rows))
        for c in _parse(rows):
            candles[c.ts] = c
        day += timedelta(days=1)

    lo, hi = int(start.timestamp()), int(end.timestamp())
    series = [candles[ts] for ts in sorted(candles) if lo <= ts < hi]
    gaps = [b.ts for a, b in itertools.pairwise(series) if b.ts - a.ts != HOUR]
    if gaps:
        raise ValueError(f"{product}: {len(gaps)} gaps in hourly candles, first at {gaps[0]}")
    return series


def _get(client: httpx.Client, product: str, start: datetime, end: datetime) -> list[list[float]]:
    assert (end - start) <= timedelta(hours=MAX_CANDLES_PER_REQUEST)
    response = client.get(
        f"{API}/products/{product}/candles",
        params={"granularity": HOUR, "start": start.isoformat(), "end": end.isoformat()},
    )
    response.raise_for_status()
    rows: list[list[float]] = response.json()
    return rows
