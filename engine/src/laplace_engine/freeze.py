"""Freezing a research run into a release: its champions, its pool of champions and its baseline ladder, exactly as
the run's report judged them."""

from __future__ import annotations

import json
from pathlib import Path

from laplace_engine.baselines import ladder_programs
from laplace_engine.release import Pool, Program, Release, weights_ppm
from laplace_research.evolve.baselines import ladder
from laplace_research.evolve.evaluate import realized_bins
from laplace_research.evolve.run import Run, parse_utc

POOL = "pool(evolved)"


def freeze(run_dir: Path, name: str, *, pool_name: str = "demon") -> Release:
    """The run's champions (private), its validation-fitted pool of champions, and its baselines with parameters
    fitted on the search window (public)."""
    report = json.loads((run_dir / "report.json").read_text())
    run = Run(run_dir)
    config, panel = run.config, run.panel
    fit = config.fit_times(panel)
    start, end, _ = config.search
    fit_range = slice(panel.index_at_or_after(parse_utc(start)), panel.index_at_or_after(parse_utc(end)))
    rungs = ladder(panel.close, realized_bins(panel, fit.times), fit.times, fit_range)
    where = f"{report['run']}'s search window ({start[:10]} to {end[:10]})"

    weights = report["pools"][POOL]["weights"]  # "island#program" -> fitted weight
    champions = []
    for member in sorted(weights):
        island, number = member.split("#")
        source = (run_dir / "champions" / f"{island}-{number}.py").read_bytes().decode()
        champions.append(Program(island, source, public=False, origin=f"{report['run']}#{number}"))
    baselines = [
        Program(n, source, public=True, origin=f"baseline fitted on {where}" if n != "naive" else "baseline")
        for n, source in ladder_programs(rungs, panel.assets, where).items()
    ]
    pool = Pool(pool_name, tuple((member.split("#")[0], w) for member, w in weights_ppm(weights)))
    notes = (
        f"Frozen from research run {report['run']}: its {len(champions)} island champions (chosen on validation), "
        f"the {POOL} pool with weights fitted on validation, and the baseline ladder fitted on {where}, the "
        "configuration its report judged on the holdout."
    )
    return Release(name, notes, tuple(champions + baselines), (pool,))
