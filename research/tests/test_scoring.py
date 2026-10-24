import math

import pytest

from laplace_research.scoring import (
    EPS,
    binary_log_loss,
    block_bootstrap,
    brier,
    log_loss,
    normalize,
    skill,
)


def test_log_loss_of_certain_hit_is_zero_and_miss_is_clipped() -> None:
    assert log_loss({"a": 1.0, "b": 0.0}, "a") == pytest.approx(0.0)
    assert log_loss({"a": 1.0, "b": 0.0}, "b") == pytest.approx(-math.log(EPS))


def test_binary_log_loss_is_symmetric() -> None:
    assert binary_log_loss(0.8, True) == pytest.approx(binary_log_loss(0.2, False))


def test_brier_of_uniform_two_way_forecast() -> None:
    assert brier({"a": 0.5, "b": 0.5}, "a") == pytest.approx(0.5)


def test_skill_is_zero_at_reference_and_one_when_perfect() -> None:
    assert skill(0.7, 0.7) == 0
    assert skill(0.0, 0.7) == 1


def test_normalize_fills_missing_keys_and_rejects_zero_mass() -> None:
    assert normalize({"a": 2.0}, ["a", "b"]) == {"a": 1.0, "b": 0.0}
    with pytest.raises(ValueError, match="sum to zero"):
        normalize({"a": 0.0}, ["a"])


def test_block_bootstrap_brackets_the_mean() -> None:
    values = [float(i % 5) for i in range(100)]
    groups = [str(i // 10) for i in range(100)]
    lo, hi = block_bootstrap(groups, lambda idx: sum(values[i] for i in idx) / len(idx), n=500)
    assert lo <= 2.0 <= hi


def test_circular_blocks_cover_overlapping_outcomes_where_day_blocks_fall_short() -> None:
    """A 24-hour outcome issued every 3 hours overlaps eight others: d_t = mean of eight shared shocks."""
    import numpy as np

    from laplace_research.scoring import circular_block_indices, interval

    rng = np.random.default_rng(0)
    n, reps = 480, 120
    covered = {8: 0, 24: 0}  # one day vs three days of 3-hourly issues
    indices = {b: circular_block_indices(n, b, n_boot=400, seed=b) for b in covered}
    for _ in range(reps):
        shocks = rng.normal(size=n + 7)
        series = [float(np.mean(shocks[t : t + 8])) for t in range(n)]
        for b in covered:
            lo, hi = interval(series, indices[b])
            covered[b] += lo <= 0.0 <= hi
    assert covered[24] / reps >= 0.85
    assert covered[24] >= covered[8]


def test_circular_block_indices_wrap_and_keep_length() -> None:
    from laplace_research.scoring import circular_block_indices

    idx = circular_block_indices(10, 4, n_boot=50, seed=1)
    assert all(len(i) == 10 for i in idx)
    assert all(0 <= j < 10 for i in idx for j in i)
