import pytest

from laplace_research.learning import fit_shrink, hedge, mix
from laplace_research.scoring import log_loss


def test_mix_interpolates() -> None:
    assert mix({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}, 0.25) == {"a": 0.25, "b": 0.75}


def test_fit_shrink_drops_a_useless_agent_and_keeps_a_perfect_one() -> None:
    outcomes = ["a", "b"] * 10
    reference = [{"a": 0.5, "b": 0.5}] * 20
    wrong = [{"a": 0.0, "b": 1.0} if o == "a" else {"a": 1.0, "b": 0.0} for o in outcomes]
    right = [{"a": 1.0, "b": 0.0} if o == "a" else {"a": 0.0, "b": 1.0} for o in outcomes]
    assert fit_shrink(wrong, reference, outcomes, log_loss) == 0.0
    assert fit_shrink(right, reference, outcomes, log_loss) == 1.0


def test_hedge_waits_for_settlement_and_then_follows_the_better_member() -> None:
    outcomes = ["a"] * 6
    good = [{"a": 0.9, "b": 0.1}] * 6
    bad = [{"a": 0.1, "b": 0.9}] * 6
    times = [0, 1, 2, 3, 4, 5]
    pooled = hedge({"good": good, "bad": bad}, outcomes, times, settle_after=2, loss=log_loss, eta=5.0)
    # Nothing has settled for the first two cards, so they get equal weights.
    assert pooled[0]["a"] == pytest.approx(0.5)
    assert pooled[1]["a"] == pytest.approx(0.5)
    assert pooled[5]["a"] > 0.85
