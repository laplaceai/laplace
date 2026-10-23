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
