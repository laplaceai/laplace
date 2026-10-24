from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from laplace_engine import candles
from laplace_research.evolve.panel import Panel

from .conftest import ISSUE


def test_issue_numbers_are_hours_since_the_epoch() -> None:
    assert candles.issue_time(ISSUE) == datetime(2026, 10, 23, 7, tzinfo=UTC)
    assert candles.current_issue(datetime(2026, 10, 23, 7, 59, 59, tzinfo=UTC)) == ISSUE


def test_fetch_requires_the_candle_that_closed_at_the_issue_time(
    monkeypatch: pytest.MonkeyPatch, make_panel: Callable[..., Panel], tmp_path: Path
) -> None:
    requested = []

    def build(start: datetime, end: datetime, cache_dir: Path) -> Panel:
        requested.append((start, end))
        return make_panel(end_issue=ISSUE - 1)  # the exchange has not published the last hour yet

    monkeypatch.setattr(candles, "build", build)
    with pytest.raises(candles.CandlesNotReady, match="05:00"):
        candles.fetch(ISSUE, tmp_path)
    assert requested == [(candles.issue_time(ISSUE) - timedelta(hours=candles.WINDOW_HOURS), candles.issue_time(ISSUE))]

    monkeypatch.setattr(candles, "build", lambda start, end, cache_dir: make_panel(end_issue=ISSUE))
    assert int(candles.fetch(ISSUE, tmp_path).ts[-1]) == (ISSUE - 1) * 3600


def test_wait_for_retries_until_the_deadline(
    monkeypatch: pytest.MonkeyPatch, make_panel: Callable[..., Panel], tmp_path: Path
) -> None:
    calls = []

    def build(start: datetime, end: datetime, cache_dir: Path) -> Panel:
        calls.append(start)
        return make_panel(end_issue=ISSUE if len(calls) >= 3 else ISSUE - 1)

    monkeypatch.setattr(candles, "build", build)
    deadline = datetime.now(UTC) + timedelta(seconds=5)
    assert candles.wait_for(ISSUE, tmp_path, deadline=deadline, poll_seconds=0.01).ts[-1] == (ISSUE - 1) * 3600
    assert len(calls) == 3
    calls.clear()
    monkeypatch.setattr(candles, "build", lambda start, end, cache_dir: make_panel(end_issue=ISSUE - 1))
    with pytest.raises(candles.CandlesNotReady):
        candles.wait_for(ISSUE, tmp_path, deadline=datetime.now(UTC), poll_seconds=0.01)
