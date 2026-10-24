"""The Phase 1 feed: distributions of hourly-close log returns for major crypto assets.

Each question asks for the probability of each of ten return bins over 1, 4 or 24 hours. The bins are fixed in
percent so a consumer can read thresholds directly ("the chance BTC falls more than 3.5% in 24 hours").
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

ASSETS: tuple[str, ...] = ("BTC", "ETH", "SOL", "XRP", "DOGE", "AVAX", "LINK")
HORIZONS: tuple[int, ...] = (1, 4, 24)
# Positive bin edges in percent of log return; the bins are symmetric around zero.
EDGES_PCT: dict[int, tuple[float, ...]] = {
    1: (0.3, 0.7, 1.5, 3.0),
    4: (0.6, 1.4, 3.0, 6.0),
    24: (1.5, 3.5, 7.0, 14.0),
}
N_BINS = 2 * 4 + 2
# A raw forecast becomes final by mixing it with the uniform distribution at this weight, which bounds the loss of any
# bin at log(N_BINS / FLOOR) and treats every forecaster, baselines included, identically.
FLOOR = 1e-3

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def product(asset: str) -> str:
    return f"{asset}-USD"


def edges(horizon: int) -> FloatArray:
    """The nine inner bin edges for ``horizon``, in log-return units, ascending."""
    pos = np.asarray(EDGES_PCT[horizon], dtype=np.float64) / 100
    return np.concatenate([-pos[::-1], [0.0], pos])


def bin_index(log_returns: FloatArray, horizon: int) -> IntArray:
    """Bin of each log return: ``k`` such that ``edges[k-1] <= r < edges[k]``; 0 and 9 are the open tails."""
    return np.searchsorted(edges(horizon), log_returns, side="right").astype(np.int64)


def finalize(raw: FloatArray) -> FloatArray:
    """Turn raw forecasts into final forecasts: normalize along the last axis, then mix with the uniform distribution
    at weight ``FLOOR``.

    This is applied exactly once, where a raw forecast enters the system. The final forecast is what gets scored,
    pooled and published. A mixture of final forecasts is already final and must not be finalized again: that would
    smooth it twice and flatter its score.
    """
    p = raw / raw.sum(axis=-1, keepdims=True)
    out: FloatArray = (1 - FLOOR) * p + FLOOR / N_BINS
    return out
