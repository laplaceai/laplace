import numpy as np
import pytest

from laplace_research.evolve.feed import N_BINS, finalize
from laplace_research.evolve.pool import PoolDidNotConverge, fit_pool, mix, realized_probs, renormalized_contributions


def test_fit_puts_the_weight_on_the_member_that_explains_the_data() -> None:
    p = np.array([[0.9] * 50, [0.1] * 50])
    assert fit_pool(p).weights[0] > 0.99


def test_fit_mixes_complementary_members_and_certifies_the_gap() -> None:
    p = np.array([[0.9, 0.1] * 50, [0.1, 0.9] * 50])
    fit = fit_pool(p, tolerance=1e-10)
    assert fit.weights == pytest.approx([0.5, 0.5], abs=1e-6)
    assert fit.gap <= 1e-10


def test_training_contributions_are_nonnegative_at_the_certified_optimum() -> None:
    p = np.random.default_rng(3).uniform(0.01, 1.0, (6, 400))
    fit = fit_pool(p, tolerance=1e-9)
    contributions = renormalized_contributions(fit.weights, p)
    # At the optimum every renormalized deletion is feasible, so none can beat the full pool beyond the gap.
    assert contributions.min() >= -fit.gap


def test_unconverged_fit_raises() -> None:
    p = np.random.default_rng(4).uniform(0.01, 1.0, (6, 400))
    with pytest.raises(PoolDidNotConverge):
        fit_pool(p, tolerance=1e-15, max_iterations=3)


def test_pooling_identical_final_forecasts_preserves_their_score() -> None:
    final = finalize(np.random.default_rng(5).uniform(0, 1, (2, 3, 7, N_BINS)))
    realized = np.random.default_rng(6).integers(0, N_BINS, (2, 3, 7))
    pooled = mix(np.array([0.3, 0.7]), [final, final])
    assert np.allclose(realized_probs(pooled, realized), realized_probs(final, realized))
