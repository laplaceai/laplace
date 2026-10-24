"""Rounding forecasts to whole basis points, and pooling rounded forecasts exactly.

The published, committed and scored forecast is the rounded one. Rounding is largest-remainder (the nearest integer
vector that sums to 10,000, ties broken by bin index), then every bin is given at least one basis point, taken from
the largest bins. Final forecasts already hold at least ``FLOOR / N_BINS`` = 1 bp in every bin, so the second step only
corrects floating-point dust, and the worst possible loss stays at log(10,000), as in research.

A pool mixes its members' rounded forecasts in integer arithmetic, so anyone can recompute a pool's forecast exactly
from its members' published ones and its published weights.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from laplace_research.evolve.feed import N_BINS, FloatArray

TOTAL = 10_000
WEIGHT_SCALE = 1_000_000  # pool weights are integers in parts per million


def to_basis_points(final: FloatArray) -> tuple[int, ...]:
    """Round one forecast over the feed's bins to integers that sum to 10,000, each at least 1."""
    p = np.asarray(final, dtype=np.float64)
    if p.shape != (N_BINS,) or not np.all(np.isfinite(p)) or np.any(p < 0) or p.sum() <= 0:
        raise ValueError("not a forecast over the feed's bins")
    scaled = p / p.sum() * TOTAL
    floors = np.floor(scaled)
    return _round([int(f) for f in floors], [float(r) for r in scaled - floors])


def mix_basis_points(forecasts: Sequence[Sequence[int]], weights: Sequence[int]) -> tuple[int, ...]:
    """The linear pool of rounded forecasts with integer weights summing to ``WEIGHT_SCALE``, rounded the same way."""
    if len(forecasts) != len(weights) or sum(weights) != WEIGHT_SCALE or min(weights) < 0:
        raise ValueError("weights must be non-negative integers summing to WEIGHT_SCALE, one per forecast")
    if any(len(f) != N_BINS or sum(f) != TOTAL for f in forecasts):
        raise ValueError("members must be basis-point forecasts")
    numerators = [sum(w * f[k] for w, f in zip(weights, forecasts, strict=True)) for k in range(N_BINS)]
    return _round([n // WEIGHT_SCALE for n in numerators], [n % WEIGHT_SCALE for n in numerators])


def probabilities(bps: Sequence[int]) -> FloatArray:
    """The forecast that a basis-point vector stands for; this is what gets scored."""
    out: FloatArray = np.asarray(bps, dtype=np.float64) / TOTAL
    return out


def _round(floors: list[int], remainders: Sequence[float] | Sequence[int]) -> tuple[int, ...]:
    order = sorted(range(N_BINS), key=lambda k: (-remainders[k], k))
    for k in order[: TOTAL - sum(floors)]:
        floors[k] += 1
    while min(floors) < 1:
        floors[floors.index(min(floors))] += 1
        floors[floors.index(max(floors))] -= 1
    return tuple(floors)
