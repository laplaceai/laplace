import json
from pathlib import Path

import pytest

from laplace_research.experiments.report import linear_pool, load, spearman


def test_spearman_of_monotone_series_is_one() -> None:
    assert spearman([1, 2, 3, 4], [10, 20, 30, 45]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_linear_pool_averages_members() -> None:
    pooled = linear_pool([{"move24": {"a": 1.0, "b": 0.0}, "big4": 0.2}, {"move24": {"a": 0.0, "b": 1.0}, "big4": 0.4}])
    assert pooled == {"move24": {"a": 0.5, "b": 0.5}, "big4": pytest.approx(0.3)}


def test_load_ignores_failures_answered_by_a_resumed_run(tmp_path: Path) -> None:
    (tmp_path / "cards.jsonl").write_text("")
    rows = [
        {"agent": "m/d", "card_id": "c1", "error": "HTTP 401"},
        {"agent": "m/d", "card_id": "c1", "move24": {}, "big4": 0.1},
        {"agent": "m/d", "card_id": "c2", "error": "HTTP 401"},
    ]
    (tmp_path / "forecasts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    _, forecasts, errors = load(tmp_path)
    assert set(forecasts["m/d"]) == {"c1"}
    assert errors == 1
