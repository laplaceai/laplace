"""The candles an issue runs on: for every feed asset, the hours that closed before the issue hour."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from laplace_research.data.candles import HOUR
from laplace_research.evolve.context import LOOKBACK
from laplace_research.evolve.panel import Panel, build

WINDOW_HOURS = LOOKBACK + 24  # a day of spare history beyond what programs see

log = logging.getLogger(__name__)


class CandlesNotReady(RuntimeError):
    """The exchange has not yet published, for every asset, the last candle before the issue hour."""


def issue_time(issue: int) -> datetime:
    """When an issue's forecasts are made: the start of its hour, once the previous candle has closed."""
    return datetime.fromtimestamp(issue * HOUR, UTC)


def current_issue(now: datetime | None = None) -> int:
    return int((now or datetime.now(UTC)).timestamp()) // HOUR


def fetch(issue: int, cache_dir: Path) -> Panel:
    """The ``WINDOW_HOURS`` hours before the issue, ending with the candle that closed at the issue time."""
    end = issue_time(issue)
    panel = build(end - timedelta(hours=WINDOW_HOURS), end, cache_dir)
    if len(panel.ts) < LOOKBACK:
        raise CandlesNotReady(f"only {len(panel.ts)} shared hours before issue {issue}")
    if int(panel.ts[-1]) != (issue - 1) * HOUR:
        last = datetime.fromtimestamp(int(panel.ts[-1]), UTC)
        raise CandlesNotReady(f"the last candle every asset has opened at {last:%Y-%m-%d %H:%M}")
    return panel


def wait_for(issue: int, cache_dir: Path, *, deadline: datetime, poll_seconds: float = 10.0) -> Panel:
    """Fetch until the issue's candles are complete; past ``deadline`` the last error is raised."""
    while True:
        try:
            return fetch(issue, cache_dir)
        except (CandlesNotReady, httpx.HTTPError, ValueError) as exc:
            if datetime.now(UTC) + timedelta(seconds=poll_seconds) > deadline:
                raise
            log.info("issue %d: candles not ready (%s); retrying in %.0fs", issue, exc, poll_seconds)
            time.sleep(poll_seconds)
