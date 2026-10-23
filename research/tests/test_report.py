import pytest

from laplace_research.experiments.report import linear_pool, spearman


def test_spearman_of_monotone_series_is_one() -> None:
    assert spearman([1, 2, 3, 4], [10, 20, 30, 45]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_linear_pool_averages_members() -> None:
    pooled = linear_pool([{"move24": {"a": 1.0, "b": 0.0}, "big4": 0.2}, {"move24": {"a": 0.0, "b": 1.0}, "big4": 0.4}])
    assert pooled == {"move24": {"a": 0.5, "b": 0.5}, "big4": pytest.approx(0.3)}
