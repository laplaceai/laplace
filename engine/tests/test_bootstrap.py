import numpy as np
import pytest

from laplace_engine import bootstrap
from laplace_research.scoring import circular_block_indices
from laplace_research.scoring import interval as research_interval


def test_resampled_means_follow_the_research_scheme_exactly() -> None:
    values = np.random.default_rng(3).normal(size=500)
    for block in (1, 7, 72, 500):
        starts = bootstrap.block_starts(len(values), block, n_boot=50)
        expected = [
            np.mean([values[(s + k) % len(values)] for s in row for k in range(block)][: len(values)]) for row in starts
        ]
        assert bootstrap.resampled_means(values, block, starts) == pytest.approx(expected)


def test_interval_agrees_with_the_research_implementation() -> None:
    values = list(np.random.default_rng(4).normal(0.01, 0.1, size=400))
    ours = bootstrap.interval(values, 24)
    theirs = research_interval(values, circular_block_indices(len(values), 24))
    assert ours == pytest.approx(theirs, abs=0.004)  # same scheme, different random draws
    assert bootstrap.interval([0.5] * 200, 72) == pytest.approx((0.5, 0.5))
