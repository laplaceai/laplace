"""The baseline ladder every evolved program is measured against.

- ``naive``: a zero-mean Laplace distribution scaled by the last week's realized volatility (the seed program).
- ``ewma``: an exponentially weighted variance (RiskMetrics) with Student-t tails.
- ``garch``: GARCH(1,1) with Student-t innovations, fitted per asset by maximum likelihood, with exact multi-step
  variance forecasts.
- ``har``: a heterogeneous autoregressive model of future realized variance on 1-, 24-, 168- and 720-hour realized
  variance, fitted by least squares per asset and horizon, with Student-t tails.

Every parameter is fitted on the search window only. Tail shapes (``nu``) are chosen per horizon by the same binned log
score the programs are judged by.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, signal, special

from laplace_research.evolve.feed import HORIZONS, N_BINS, FloatArray, IntArray, edges, finalize

NU_GRID = (2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 20.0, 50.0)
LAMBDA_GRID = (0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998)
HAR_WINDOWS = (1, 24, 168, 720)


def hourly_returns(close: FloatArray) -> FloatArray:
    """``r[a, t] = log(close[a, t] / close[a, t - 1])``, with ``r[a, 0] = 0``."""
    r = np.zeros_like(close)
    r[:, 1:] = np.diff(np.log(close), axis=1)
    return r


def trailing_mean(x: FloatArray, window: int) -> FloatArray:
    """Mean of ``x[..., t - window + 1 : t + 1]`` (shorter at the start)."""
    c = np.cumsum(x, axis=-1)
    out = c.copy()
    out[..., window:] = c[..., window:] - c[..., :-window]
    counts = np.minimum(np.arange(1, x.shape[-1] + 1), window)
    result: FloatArray = out / counts
    return result


def t_bins(sd: FloatArray, nu: float, horizon: int) -> FloatArray:
    """Student-t bin probabilities for an array of standard deviations: shape ``sd.shape + (N_BINS,)``."""
    scale = np.maximum(sd, 1e-9) * math.sqrt((nu - 2) / nu)
    cdf = special.stdtr(nu, edges(horizon) / scale[..., None])
    zeros = np.zeros((*sd.shape, 1))
    out: FloatArray = np.diff(np.concatenate([zeros, cdf, zeros + 1], axis=-1), axis=-1)
    return out


def laplace_bins(sd: FloatArray, horizon: int) -> FloatArray:
    b = np.maximum(sd, 1e-9)[..., None] / math.sqrt(2)
    z = edges(horizon) / b
    cdf = np.where(z < 0, 0.5 * np.exp(np.minimum(z, 0)), 1 - 0.5 * np.exp(-np.maximum(z, 0)))
    zeros = np.zeros((*sd.shape, 1))
    out: FloatArray = np.diff(np.concatenate([zeros, cdf, zeros + 1], axis=-1), axis=-1)
    return out


def _mean_log_score(probs: FloatArray, realized: IntArray) -> float:
    p = finalize(probs)
    return float(np.mean(np.log(np.take_along_axis(p, realized[..., None], axis=-1)[..., 0])))


def _best_nu(sd: FloatArray, realized: IntArray, horizon: int) -> float:
    return max(NU_GRID, key=lambda nu: _mean_log_score(t_bins(sd, nu, horizon), realized))


@dataclass
class Baseline:
    """A fitted baseline: ``sd(times)`` gives ``[asset, horizon, sample]`` standard deviations; ``probs`` returns raw
    bin probabilities (finalize them once before scoring)."""

    name: str
    sd: Callable[[IntArray], FloatArray]
    nu: dict[int, float] = field(default_factory=dict)  # empty for the Laplace baseline

    def probs(self, times: IntArray) -> FloatArray:
        sd = self.sd(times)
        out = np.empty((*sd.shape, N_BINS))
        for j, h in enumerate(HORIZONS):
            out[:, j] = laplace_bins(sd[:, j], h) if not self.nu else t_bins(sd[:, j], self.nu[h], h)
        return out


def naive(close: FloatArray) -> Baseline:
    week_var = trailing_mean(hourly_returns(close) ** 2, 168)

    def sd(times: IntArray) -> FloatArray:
        return np.sqrt(np.stack([week_var[:, times] * h for h in HORIZONS], axis=1))

    return Baseline("naive", sd)


def _fit_nu(sd: Callable[[IntArray], FloatArray], realized: IntArray, fit_times: IntArray) -> dict[int, float]:
    s = sd(fit_times)
    return {h: _best_nu(s[:, j], realized[:, j], h) for j, h in enumerate(HORIZONS)}


def ewma(close: FloatArray, realized_fit: IntArray, fit_times: IntArray) -> Baseline:
    r2 = hourly_returns(close) ** 2

    def variance(lam: float) -> FloatArray:
        init = r2[:, 1:169].mean(axis=1, keepdims=True)
        v, _ = signal.lfilter([1 - lam], [1, -lam], r2, axis=1, zi=lam * init)
        out: FloatArray = v
        return out

    def make_sd(v: FloatArray) -> Callable[[IntArray], FloatArray]:
        return lambda times: np.sqrt(np.stack([v[:, times] * h for h in HORIZONS], axis=1))

    best: tuple[float, float, dict[int, float]] | None = None
    for lam in LAMBDA_GRID:
        sd = make_sd(variance(lam))
        nus = _fit_nu(sd, realized_fit, fit_times)
        score = sum(
            _mean_log_score(t_bins(sd(fit_times)[:, j], nus[h], h), realized_fit[:, j]) for j, h in enumerate(HORIZONS)
        )
        if best is None or score > best[0]:
            best = (score, lam, nus)
    assert best is not None
    _, lam, nus = best
    return Baseline(f"ewma(lambda={lam})", make_sd(variance(lam)), nus)


def _garch_variance(r: FloatArray, omega: float, alpha: float, beta: float) -> FloatArray:
    """Conditional variance of ``r[t + 1]`` given returns up to ``t``: ``v[t] = omega + alpha r[t]^2 + beta v[t-1]``."""
    init = float(np.mean(r[: min(len(r), 720)] ** 2))
    steady = omega / (1 - beta)
    # y[t] = alpha r[t]^2 + beta y[t-1] with y = v - steady reproduces the recursion exactly, from v[-1] = init.
    y, _ = signal.lfilter([alpha], [1, -beta], r**2, zi=[beta * (init - steady)])
    out: FloatArray = y + steady
    return out


def _garch_fit(r: FloatArray) -> tuple[float, float, float]:
    var = float(np.var(r))

    def nll(params: FloatArray) -> float:
        a, b, nu = params
        if a < 0 or b < 0 or a + b >= 0.9999 or nu <= 2.05:
            return 1e12
        omega = var * (1 - a - b)
        v = np.maximum(_garch_variance(r[:-1], omega, a, b), 1e-14)
        z2 = r[1:] ** 2 / (v * (nu - 2) / nu)
        ll = (
            special.gammaln((nu + 1) / 2)
            - special.gammaln(nu / 2)
            - 0.5 * np.log(np.pi * (nu - 2) / nu * v)
            - (nu + 1) / 2 * np.log1p(z2 / nu)
        )
        return float(-np.sum(ll))

    best = optimize.minimize(nll, x0=np.array([0.05, 0.9, 5.0]), method="Nelder-Mead", options={"maxiter": 600})
    a, b, _ = best.x
    return var * (1 - a - b), float(a), float(b)


def garch(close: FloatArray, realized_fit: IntArray, fit_times: IntArray, fit_range: slice) -> Baseline:
    r = hourly_returns(close)
    one_step = np.empty_like(r)
    persistence = np.empty(len(r))
    long_run = np.empty(len(r))
    for a in range(r.shape[0]):
        omega, alpha, beta = _garch_fit(r[a, fit_range])
        one_step[a] = _garch_variance(r[a], omega, alpha, beta)
        persistence[a] = alpha + beta
        long_run[a] = omega / (1 - alpha - beta)

    def sd(times: IntArray) -> FloatArray:
        out = np.empty((r.shape[0], len(HORIZONS), len(times)))
        for j, h in enumerate(HORIZONS):
            # Sum of k-step variance forecasts: V_L + p^(k-1) (v1 - V_L), k = 1..h.
            geometric = np.array([np.sum(p ** np.arange(h)) for p in persistence])
            v1 = one_step[:, times]
            out[:, j] = np.sqrt(
                np.maximum(h * long_run[:, None] + geometric[:, None] * (v1 - long_run[:, None]), 1e-12)
            )
        return out

    return Baseline("garch(1,1)-t", sd, _fit_nu(sd, realized_fit, fit_times))


def har(close: FloatArray, realized_fit: IntArray, fit_times: IntArray) -> Baseline:
    r2 = hourly_returns(close) ** 2
    features = np.stack([trailing_mean(r2, w) for w in HAR_WINDOWS], axis=-1)  # [asset, t, k]
    future = np.cumsum(r2, axis=1)
    coefs: dict[tuple[int, int], FloatArray] = {}
    for a in range(r2.shape[0]):
        for h in HORIZONS:
            t = fit_times
            y = future[a, t + h] - future[a, t]
            x = np.column_stack([np.ones(len(t)), features[a, t] * h])
            coefs[a, h], *_ = np.linalg.lstsq(x, y, rcond=None)

    def sd(times: IntArray) -> FloatArray:
        out = np.empty((r2.shape[0], len(HORIZONS), len(times)))
        for a in range(r2.shape[0]):
            for j, h in enumerate(HORIZONS):
                x = np.column_stack([np.ones(len(times)), features[a, times] * h])
                floor_var = 0.1 * features[a, times, 2] * h  # never below a tenth of the weekly variance
                out[a, j] = np.sqrt(np.maximum(x @ coefs[a, h], floor_var))
        return out

    return Baseline("har-rv", sd, _fit_nu(sd, realized_fit, fit_times))


def ladder(close: FloatArray, realized_fit: IntArray, fit_times: IntArray, fit_range: slice) -> list[Baseline]:
    return [
        naive(close),
        ewma(close, realized_fit, fit_times),
        garch(close, realized_fit, fit_times, fit_range),
        har(close, realized_fit, fit_times),
    ]
