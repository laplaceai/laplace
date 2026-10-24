"""Rounding final forecasts to whole basis points.

The published, committed and scored forecast is the rounded one. Rounding is largest-remainder (the nearest integer
vector that sums to 10,000, ties broken by bin index), then every bin is given at least one basis point, taken from
the largest bins. Final forecasts already hold at least ``FLOOR / N_BINS`` = 1 bp in every bin, so the second step only
corrects floating-point dust, and the worst possible loss stays at log(10,000), as in research.
"""

from __future__ import annotations

import numpy as np

from laplace_research.evolve.feed import N_BINS, FloatArray

TOTAL = 10_000


def to_basis_points(final: FloatArray) -> tuple[int, ...]:
    """Round one forecast over the feed's bins to integers that sum to 10,000, each at least 1."""
    p = np.asarray(final, dtype=np.float64)
    if p.shape != (N_BINS,) or not np.all(np.isfinite(p)) or np.any(p < 0) or p.sum() <= 0:
        raise ValueError("not a forecast over the feed's bins")
    scaled = p / p.sum() * TOTAL
    bps = np.floor(scaled).astype(np.int64)
    order = np.argsort(-(scaled - bps), kind="stable")
    bps[order[: TOTAL - int(bps.sum())]] += 1
    while bps.min() < 1:
        bps[int(np.argmin(bps))] += 1
        bps[int(np.argmax(bps))] -= 1
    return tuple(int(b) for b in bps)


def probabilities(bps: tuple[int, ...]) -> FloatArray:
    """The forecast that a basis-point vector stands for; this is what gets scored."""
    out: FloatArray = np.asarray(bps, dtype=np.float64) / TOTAL
    return out
