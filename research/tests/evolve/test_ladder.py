from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from laplace_research.evolve import baselines as b
from laplace_research.evolve.evaluate import realized_bins, sample_times
from laplace_research.evolve.feed import HORIZONS, N_BINS
from laplace_research.evolve.history import HistoryReader, history_paths
from laplace_research.evolve.islands import SEED_PROGRAM
from laplace_research.evolve.panel import Panel, utc
from laplace_research.evolve.sandbox import load


def test_trailing_mean_matches_a_loop() -> None:
    x = np.arange(10, dtype=float)
    assert b.trailing_mean(x, 3)[5] == pytest.approx((3 + 4 + 5) / 3)
    assert b.trailing_mean(x, 3)[1] == pytest.approx(0.5)


def test_garch_variance_matches_the_recursion() -> None:
    r = np.random.default_rng(0).normal(0, 0.01, 50)
    omega, alpha, beta = 1e-6, 0.1, 0.85
    v = b._garch_variance(r, omega, alpha, beta)
    prev = float(np.mean(r**2))
    for t in range(len(r)):
        prev = omega + alpha * r[t] ** 2 + beta * prev
        assert v[t] == pytest.approx(prev)


def test_t_log_density_matches_scipy() -> None:
    x = np.linspace(-0.1, 0.1, 41)
    for nu in (2.5, 4.0, 30.0):
        variance = np.full_like(x, 4e-4)
        expected = stats.t.logpdf(x, nu, scale=np.sqrt(4e-4 * (nu - 2) / nu))
        assert np.allclose(b._t_log_density(x, variance, nu), expected)


def test_garch_fit_recovers_simulated_parameters() -> None:
    # A wrong normalizing constant in the likelihood once rewarded nu -> infinity and left alpha and beta arbitrary.
    rng = np.random.default_rng(1)
    omega, alpha, beta, nu, n = 2e-6, 0.08, 0.9, 5.0, 20_000
    var, r = omega / (1 - alpha - beta), np.empty(n)
    for t in range(n):
        r[t] = rng.standard_t(nu) * np.sqrt(var * (nu - 2) / nu)
        var = omega + alpha * r[t] ** 2 + beta * var
    fitted_omega, fitted_alpha, fitted_beta = b._garch_fit(r)
    assert fitted_alpha == pytest.approx(alpha, abs=0.02)
    assert fitted_beta == pytest.approx(beta, abs=0.03)
    assert fitted_omega / (1 - fitted_alpha - fitted_beta) == pytest.approx(np.var(r), rel=1e-9)


def test_seed_program_reproduces_the_naive_baseline(panel: Panel, history: Path) -> None:
    fn = load(SEED_PROGRAM)
    t = 900
    expected = b.naive(panel.close).probs(np.array([t]))
    reader = HistoryReader(*history_paths(history))
    reader.advance_to(t)
    for a in range(len(panel.assets)):
        for j, h in enumerate(HORIZONS):
            assert np.allclose(fn(reader.context(a, t, h)), expected[a, j, 0])


def test_ladder_fits_and_returns_distributions(panel: Panel) -> None:
    fit = sample_times(panel, utc(int(panel.ts[0])), utc(int(panel.ts[-1])), 3)
    rungs = b.ladder(panel.close, realized_bins(panel, fit), fit, slice(0, len(panel.ts)))
    assert [r.name.split("(")[0] for r in rungs] == ["naive", "ewma", "garch", "har-rv"]
    for rung in rungs:
        p = rung.probs(fit[:5])
        assert p.shape == (len(panel.assets), len(HORIZONS), 5, N_BINS)
        assert np.allclose(p.sum(axis=-1), 1)
