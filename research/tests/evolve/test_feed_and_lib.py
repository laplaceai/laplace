import numpy as np
import pytest

from laplace_research.evolve import lib
from laplace_research.evolve.feed import FLOOR, HORIZONS, N_BINS, bin_index, edges, finalize


@pytest.mark.parametrize("h", HORIZONS)
def test_edges_are_symmetric_and_ascending(h: int) -> None:
    e = edges(h)
    assert len(e) == N_BINS - 1
    assert np.all(np.diff(e) > 0)
    assert np.allclose(e, -e[::-1])


def test_bin_index_puts_edges_in_the_upper_bin_and_tails_at_the_ends() -> None:
    e = edges(24)
    assert list(bin_index(np.array([-1.0, e[0], 0.0, -1e-9, 1.0]), 24)) == [0, 1, 5, 4, 9]


def test_finalize_normalizes_and_bounds_every_bin() -> None:
    p = finalize(np.array([[5.0] + [0.0] * (N_BINS - 1)]))
    assert p.sum() == pytest.approx(1)
    assert p.min() == pytest.approx(FLOOR / N_BINS)


@pytest.mark.parametrize(
    "probs",
    [
        lib.normal_probs(0.02, edges(24)),
        lib.laplace_probs(0.02, edges(24)),
        lib.student_t_probs(0.02, 4.0, edges(24)),
    ],
)
def test_helpers_return_distributions(probs: np.ndarray) -> None:
    assert probs.shape == (N_BINS,)
    assert probs.sum() == pytest.approx(1)
    assert np.allclose(probs, probs[::-1])  # zero-mean shapes are symmetric


def test_student_t_approaches_normal() -> None:
    assert np.allclose(lib.student_t_probs(0.02, 1e6, edges(24)), lib.normal_probs(0.02, edges(24)), atol=1e-4)


def test_mixture_averages() -> None:
    a, b = lib.normal_probs(0.01, edges(4)), lib.normal_probs(0.03, edges(4))
    assert np.allclose(lib.mixture([a, b], [1, 3]), 0.25 * a + 0.75 * b)
