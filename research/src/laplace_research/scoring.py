"""Proper scoring rules and a day-block bootstrap for confidence intervals.

Cards that share a day overlap in time (a 24-hour question every 6 hours), so they are not independent. The
bootstrap resamples whole days, not single cards.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence

EPS = 1e-4


def log_loss(probs: Mapping[str, float], outcome: str) -> float:
    return -math.log(max(probs.get(outcome, 0.0), EPS))


def binary_log_loss(p_yes: float, outcome: bool) -> float:
    p = min(max(p_yes, EPS), 1 - EPS)
    return -math.log(p if outcome else 1 - p)


def brier(probs: Mapping[str, float], outcome: str) -> float:
    return sum((p - (k == outcome)) ** 2 for k, p in probs.items())


def skill(loss: float, reference_loss: float) -> float:
    """Skill score: 0 = as good as the reference, 1 = perfect, negative = worse."""
    return 1 - loss / reference_loss


def normalize(probs: Mapping[str, float], keys: Sequence[str]) -> dict[str, float]:
    values = {k: max(float(probs.get(k, 0.0)), 0.0) for k in keys}
    total = sum(values.values())
    if total <= 0:
        raise ValueError("probabilities sum to zero")
    return {k: v / total for k, v in values.items()}


def block_bootstrap(
    groups: Sequence[str],
    statistic: Callable[[Sequence[int]], float],
    *,
    n: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """95% interval of ``statistic(indices)`` when resampling whole groups with replacement."""
    by_group: dict[str, list[int]] = {}
    for idx, g in enumerate(groups):
        by_group.setdefault(g, []).append(idx)
    keys = sorted(by_group)
    rng = random.Random(seed)
    stats = sorted(statistic([i for k in rng.choices(keys, k=len(keys)) for i in by_group[k]]) for _ in range(n))
    return stats[int(0.025 * n)], stats[int(0.975 * n) - 1]


def circular_block_indices(n_obs: int, block: int, *, n_boot: int = 2000, seed: int = 0) -> list[list[int]]:
    """Resampled index sets for a circular block bootstrap of a time series.

    Each resample concatenates blocks of ``block`` consecutive observations, starting at uniform random positions and
    wrapping around the end, truncated to ``n_obs``. Reusing the same index sets for every comparison keeps their
    intervals paired. Unlike calendar-day blocks, blocks may straddle midnight, which matters when outcomes overlap
    (a 24-hour return issued every 3 hours overlaps eight others).
    """
    if not 1 <= block <= n_obs:
        raise ValueError("block length must be between 1 and the number of observations")
    rng = random.Random(seed)
    n_blocks = -(-n_obs // block)
    return [
        [(start + k) % n_obs for start in (rng.randrange(n_obs) for _ in range(n_blocks)) for k in range(block)][:n_obs]
        for _ in range(n_boot)
    ]


def interval(values: Sequence[float], indices: Sequence[Sequence[int]], level: float = 0.95) -> tuple[float, float]:
    """Percentile interval of the mean of ``values`` over bootstrap ``indices``."""
    means = sorted(sum(values[i] for i in idx) / len(idx) for idx in indices)
    tail = (1 - level) / 2
    return means[int(tail * len(means))], means[int((1 - tail) * len(means)) - 1]
