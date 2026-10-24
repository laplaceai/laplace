import numpy as np
import pytest

from laplace_research.evolve.baselines import naive
from laplace_research.evolve.evaluate import (
    Evaluator,
    log_scores,
    mean_gain,
    ranked_probability_scores,
    realized_bins,
    sample_times,
)
from laplace_research.evolve.feed import N_BINS
from laplace_research.evolve.islands import SEED_PROGRAM
from laplace_research.evolve.panel import Panel, utc


def test_perfect_forecast_has_zero_rps_and_near_zero_log_loss() -> None:
    realized = np.array([[[3]]])
    probs = np.zeros((1, 1, 1, N_BINS))
    probs[..., 3] = 1
    assert ranked_probability_scores(probs, realized)[0, 0, 0] == pytest.approx(0, abs=1e-6)
    assert log_scores(probs, realized)[0, 0, 0] == pytest.approx(0, abs=2e-3)


def test_sample_times_leave_room_for_lookback_and_horizon(panel: Panel) -> None:
    times = sample_times(panel, utc(int(panel.ts[0])), utc(int(panel.ts[-1])), 7)
    assert times[0] >= 719
    assert times[-1] + 24 < len(panel.ts)


def test_evaluator_scores_the_seed_at_zero_gain_and_reports_failures(panel: Panel, tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "panel.npz"
    panel.save(path)
    times = sample_times(panel, utc(int(panel.ts[0])), utc(int(panel.ts[-1])), 50)
    realized = realized_bins(panel, times)
    reference = log_scores(naive(panel.close).probs(times), realized)
    with Evaluator(path, workers=2) as evaluator:
        seed = evaluator.run(SEED_PROGRAM, times, 30)
        assert seed.error is None
        assert seed.probs is not None
        assert mean_gain(log_scores(seed.probs, realized), reference) == pytest.approx(0, abs=1e-9)

        crash = evaluator.run("def forecast(ctx):\n    return [1] * 3\n", times, 30)
        assert crash.probs is None
        assert crash.error is not None
        assert "bad output" in crash.error

        slow = evaluator.run("def forecast(ctx):\n    while True:\n        pass\n", times, 1)
        assert slow.error is not None
        assert "timed out" in slow.error

        mutate = evaluator.run("def forecast(ctx):\n    ctx.close[0] = 1.0\n    return [1] * 10\n", times, 30)
        assert mutate.error is not None
        assert "read-only" in mutate.error
