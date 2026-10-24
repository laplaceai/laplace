"""Hourly candles from the Coinbase Exchange public API, cached on disk.

Only public, unauthenticated endpoints are used. Candles are cached per product and day so a rerun of an experiment
is reproducible and does not hit the API again. Missing days are fetched in chunks of up to 12 days (288 candles, under
the API's 300-candle limit).
"""

from __future__ import annotations

import itertools
import json
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

API = "https://api.exchange.coinbase.com"
HOUR = 3600
DAY = 86400
MAX_CANDLES_PER_REQUEST = 300
CHUNK_DAYS = 12
REQUEST_INTERVAL_S = 0.15


@dataclass(frozen=True, slots=True)
class Candle:
    """One hourly candle. ``ts`` is the candle's open time (unix seconds, UTC)."""

    ts: int
    low: float
    high: float
    open: float
    close: float
    volume: float


def _parse(rows: Sequence[Sequence[float]]) -> list[Candle]:
    # Coinbase rows are [time, low, high, open, close, volume], newest first.
    return sorted((Candle(int(r[0]), r[1], r[2], r[3], r[4], r[5]) for r in rows), key=lambda c: c.ts)


def _cache_path(cache_dir: Path, product: str, day: datetime) -> Path:
    return cache_dir / f"{product}-{day:%Y%m%d}.json"


def _runs_of_consecutive_days(days: Sequence[datetime]) -> Iterator[list[datetime]]:
    """Split sorted days into runs of consecutive days, each at most ``CHUNK_DAYS`` long."""
    run: list[datetime] = []
    for day in days:
        if run and (day - run[-1] != timedelta(days=1) or len(run) == CHUNK_DAYS):
            yield run
            run = []
        run.append(day)
    if run:
        yield run


def fetch_hourly(
    product: str,
    start: datetime,
    end: datetime,
    *,
    client: httpx.Client,
    cache_dir: Path,
    max_filled_hours: int = 0,
) -> list[Candle]:
    """Return hourly candles for ``product`` in ``[start, end)``.

    Hours without a candle (no trades, or an exchange outage) are filled with a flat candle at the previous close and
    zero volume, up to ``max_filled_hours`` in total; beyond that a ``ValueError`` is raised.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    days = []
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while day < end:
        days.append(day)
        day += timedelta(days=1)
    complete_before = datetime.now(UTC) - timedelta(hours=1)

    rows_by_day: dict[datetime, list[list[float]]] = {}
    missing = []
    for day in days:
        path = _cache_path(cache_dir, product, day)
        if path.exists():
            rows_by_day[day] = json.loads(path.read_text())
        else:
            missing.append(day)

    for run in _runs_of_consecutive_days(missing):
        rows = _get(client, product, run[0], run[-1] + timedelta(days=1))
        for day in run:
            lo = int(day.timestamp())
            day_rows = [r for r in rows if lo <= int(r[0]) < lo + DAY]
            rows_by_day[day] = day_rows
            if day + timedelta(days=1) <= complete_before:
                _cache_path(cache_dir, product, day).write_text(json.dumps(day_rows))

    candles = {c.ts: c for day in days for c in _parse(rows_by_day[day])}
    lo, hi = int(start.timestamp()), int(end.timestamp())
    series = [candles[ts] for ts in sorted(candles) if lo <= ts < hi]
    return _fill_gaps(product, series, max_filled_hours)


def _fill_gaps(product: str, series: list[Candle], max_filled_hours: int) -> list[Candle]:
    filled: list[Candle] = []
    n_filled = 0
    for prev, cur in itertools.pairwise(series):
        filled.append(prev)
        for ts in range(prev.ts + HOUR, cur.ts, HOUR):
            filled.append(Candle(ts, prev.close, prev.close, prev.close, prev.close, 0.0))
            n_filled += 1
    if series:
        filled.append(series[-1])
    if n_filled > max_filled_hours:
        raise ValueError(f"{product}: {n_filled} missing hourly candles (allowed {max_filled_hours})")
    return filled


def _get(client: httpx.Client, product: str, start: datetime, end: datetime) -> list[list[float]]:
    if (end - start) > timedelta(hours=MAX_CANDLES_PER_REQUEST):
        raise ValueError("request spans more candles than the API returns")
    for attempt in range(6):
        time.sleep(REQUEST_INTERVAL_S)
        response = client.get(
            f"{API}/products/{product}/candles",
            params={"granularity": HOUR, "start": start.isoformat(), "end": end.isoformat()},
        )
        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            time.sleep(2**attempt)
            continue
        response.raise_for_status()
        rows: list[list[float]] = response.json()
        return rows
    raise RuntimeError(f"{product}: rate-limited on every attempt")
