"""Distribution helpers that forecasting programs may call as ``ctx.lib``.

Every helper takes the standard deviation of the horizon's log return (not a distribution's scale parameter), so
programs can swap shapes without re-deriving scale conversions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import special

from laplace_research.evolve.feed import FloatArray


def _bins_from_cdf(cdf_at_edges: FloatArray) -> FloatArray:
    out: FloatArray = np.diff(np.concatenate([[0.0], cdf_at_edges, [1.0]]))
    return np.clip(out, 0.0, 1.0)


def normal_probs(sd: float, edges: FloatArray, loc: float = 0.0) -> FloatArray:
    """Bin probabilities of a normal distribution with standard deviation ``sd``."""
    sd = max(float(sd), 1e-9)
    return _bins_from_cdf(special.ndtr((edges - loc) / sd))


def laplace_probs(sd: float, edges: FloatArray, loc: float = 0.0) -> FloatArray:
    """Bin probabilities of a Laplace distribution with standard deviation ``sd`` (scale ``sd / sqrt(2)``)."""
    b = max(float(sd), 1e-9) / math.sqrt(2)
    z = (edges - loc) / b
    cdf = np.where(z < 0, 0.5 * np.exp(np.minimum(z, 0.0)), 1 - 0.5 * np.exp(-np.maximum(z, 0.0)))
    return _bins_from_cdf(cdf)


def student_t_probs(sd: float, nu: float, edges: FloatArray, loc: float = 0.0) -> FloatArray:
    """Bin probabilities of a Student-t distribution with ``nu > 2`` degrees of freedom and standard deviation
    ``sd``."""
    nu = max(float(nu), 2.05)
    scale = max(float(sd), 1e-9) * math.sqrt((nu - 2) / nu)
    return _bins_from_cdf(special.stdtr(nu, (edges - loc) / scale))


def mixture(components: Sequence[FloatArray], weights: Sequence[float]) -> FloatArray:
    """Weighted average of bin distributions; weights are normalized."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    out: FloatArray = np.sum(
        [wi * np.asarray(c, dtype=np.float64) for wi, c in zip(w, components, strict=True)], axis=0
    )
    return out


def log_returns(close: FloatArray) -> FloatArray:
    """Hourly log returns of a close-price series (one element shorter)."""
    out: FloatArray = np.diff(np.log(close))
    return out


class Lib:
    """Namespace object handed to programs as ``ctx.lib``."""

    normal_probs = staticmethod(normal_probs)
    laplace_probs = staticmethod(laplace_probs)
    student_t_probs = staticmethod(student_t_probs)
    mixture = staticmethod(mixture)
    log_returns = staticmethod(log_returns)


LIB = Lib()
