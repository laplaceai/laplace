import pytest

from laplace_research import baselines
from laplace_research.cards import WEEK, Outcome
from laplace_research.data.candles import Candle


def test_uniform_sums_to_one() -> None:
    move, big4 = baselines.uniform()
    assert sum(move.values()) == pytest.approx(1)
    assert big4 == 0.5


def test_base_rate_applies_rule_of_succession() -> None:
    move, big4 = baselines.base_rate([Outcome("m1_p1", False)] * 3)
    assert move["m1_p1"] == pytest.approx(4 / 8)
    assert move["gt_p3"] == pytest.approx(1 / 8)
    assert big4 == pytest.approx(1 / 5)


def test_laplace_cdf_is_a_distribution() -> None:
    assert baselines.laplace_cdf(0.0, 0.01) == pytest.approx(0.5)
    assert baselines.laplace_cdf(-1.0, 0.01) == pytest.approx(0.0, abs=1e-12)
    assert baselines.laplace_cdf(1.0, 0.01) == pytest.approx(1.0)


def test_laplace_vol_sums_to_one_and_is_symmetric_in_log_returns(candles: list[Candle]) -> None:
    move, big4 = baselines.laplace_vol(candles, WEEK + 5)
    assert sum(move.values()) == pytest.approx(1)
    # -3% is further from zero than +3% in log space, so the lower tail is slightly thinner.
    assert move["lt_m3"] < move["gt_p3"]
    assert move["lt_m3"] == pytest.approx(move["gt_p3"], rel=0.1)
    assert 0 < big4 < 1
