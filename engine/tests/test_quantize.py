import numpy as np
import pytest

from laplace_engine.quantize import TOTAL, probabilities, to_basis_points
from laplace_research.evolve.feed import FLOOR, N_BINS, finalize


def test_rounds_to_the_nearest_basis_point_vector() -> None:
    rng = np.random.default_rng(0)
    for _ in range(2_000):
        final = finalize(rng.dirichlet(np.full(N_BINS, 0.3)))
        bps = to_basis_points(final)
        assert sum(bps) == TOTAL
        assert min(bps) >= 1
        assert np.max(np.abs(np.asarray(bps) - final * TOTAL)) < 1


def test_keeps_one_basis_point_in_floored_bins_despite_float_dust() -> None:
    final = np.full(N_BINS, FLOOR / N_BINS * (1 - 1e-12))
    final[4] = 1 - final[:4].sum() - final[5:].sum()
    bps = to_basis_points(final)
    assert bps[4] == TOTAL - (N_BINS - 1)
    assert all(b == 1 for i, b in enumerate(bps) if i != 4)


def test_is_deterministic_on_ties() -> None:
    assert to_basis_points(np.full(N_BINS, 0.1)) == (1_000,) * N_BINS
    # Seven equal bins: the four missing units go to the first four, then the three empty bins take one each from the
    # largest bins, again in bin order.
    sevenths = np.array([1.0] * 7 + [0.0] * 3)
    assert to_basis_points(sevenths) == (1_428, 1_428, 1_428, 1_429, 1_428, 1_428, 1_428, 1, 1, 1)


def test_round_trip_is_what_gets_scored() -> None:
    bps = to_basis_points(finalize(np.arange(1.0, N_BINS + 1)))
    assert probabilities(bps).sum() == pytest.approx(1)


@pytest.mark.parametrize(
    "bad", [np.ones(N_BINS - 1), np.full(N_BINS, np.nan), -np.ones(N_BINS), np.zeros(N_BINS)], ids=str
)
def test_rejects_non_forecasts(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="not a forecast"):
        to_basis_points(bad)
