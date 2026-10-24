import numpy as np
import pytest

from laplace_engine.baselines import NAIVE, ladder_programs
from laplace_research.evolve.baselines import ladder
from laplace_research.evolve.context import LOOKBACK
from laplace_research.evolve.evaluate import realized_bins, sample_times
from laplace_research.evolve.feed import HORIZONS, edges
from laplace_research.evolve.history import HistoryReader, write_history
from laplace_research.evolve.islands import SEED_PROGRAM
from laplace_research.evolve.lib import student_t_probs
from laplace_research.evolve.panel import Panel, utc
from laplace_research.evolve.sandbox import load


def test_the_naive_baseline_is_the_seed_program() -> None:
    assert NAIVE == SEED_PROGRAM


def test_baseline_programs_reproduce_the_research_ladder(panel: Panel, tmp_path) -> None:  # type: ignore[no-untyped-def]
    fit = sample_times(panel, utc(int(panel.ts[0])), utc(int(panel.ts[800])), 3)
    rungs = ladder(panel.close, realized_bins(panel, fit), fit, slice(0, 800))
    sources = ladder_programs(rungs, panel.assets, "a synthetic window")
    assert list(sources) == ["naive", "ewma", "garch", "har"]

    path = tmp_path / "panel.npz"
    panel.save(path)
    reader = HistoryReader(*write_history(panel, path))
    t = len(panel.ts) - 1
    reader.advance_to(t)
    research = dict(zip(sources, (rung.probs(np.array([t])) for rung in rungs), strict=True))
    har = rungs[3]
    r2 = np.diff(np.log(panel.close[:, t + 1 - LOOKBACK : t + 1])) ** 2
    for name, source in sources.items():
        fn = load(source)
        for a in range(len(panel.assets)):
            for j, h in enumerate(HORIZONS):
                got = np.asarray(fn(reader.context(a, t, h)))
                if name != "har":
                    # Recursions restart from the window; after 719 steps the start no longer matters.
                    assert got == pytest.approx(research[name][a, j, 0], abs=1e-9), (name, a, h)
                    continue
                # HARs longest mean covers the windows 719 returns where research has 720: check the formula exactly.
                means = np.array([r2[a, -w:].mean() for w in har.params["windows"]])
                c = np.asarray(har.params["per_asset"][a][str(h)])
                variance = max(c[0] + float(c[1:] @ means) * h, 0.1 * means[2] * h)
                expected = student_t_probs(np.sqrt(variance), har.nu[h], edges(h))
                assert got == pytest.approx(expected, abs=1e-12), (name, a, h)
