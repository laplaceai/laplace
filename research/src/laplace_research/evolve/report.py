"""Judge an evolution honestly.

1. Each island's champion is chosen on the validation window from its three best programs by search score, so the
   search score never decides on its own.
2. Champions, the baseline ladder and two ensembles are then scored once on the holdout window, which played no part
   in any choice. Intervals come from a bootstrap over whole days.
3. The ensembles ("the demon") are linear pools whose weights are fitted on validation by EM. Each member's
   contribution is the drop in the pool's holdout score when it is removed (weights renormalized, not refitted).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from laplace_research.evolve.baselines import ladder
from laplace_research.evolve.db import Program
from laplace_research.evolve.evaluate import (
    Evaluator,
    day_groups,
    log_scores,
    ranked_probability_scores,
    realized_bins,
)
from laplace_research.evolve.feed import HORIZONS, FloatArray, floor
from laplace_research.evolve.run import TIME_LIMIT_S, Run, parse_utc
from laplace_research.experiments.report import spearman
from laplace_research.scoring import block_bootstrap


@dataclass
class Member:
    name: str
    kind: str  # "champion" or "baseline"
    search_gain: float | None
    validation: FloatArray  # probs [asset, horizon, sample, bin]
    holdout: FloatArray
    program: Program | None = None


def em_weights(p: FloatArray, iterations: int = 500, tol: float = 1e-10) -> FloatArray:
    """Mixture weights maximizing the mean log of ``sum_m w_m p[m, s]`` (``p``: members x samples, positive)."""
    m = p.shape[0]
    w = np.full(m, 1.0 / m)
    for _ in range(iterations):
        mix = w @ p
        new = w * np.mean(p / mix, axis=1)
        new /= new.sum()
        if np.max(np.abs(new - w)) < tol:
            return new
        w = new
    return w


def realized_probs(probs: FloatArray, realized: np.ndarray) -> FloatArray:
    """Floored probability of the realized bin, flattened over asset, horizon and sample."""
    out: FloatArray = np.take_along_axis(floor(probs), realized[..., None], axis=-1)[..., 0].reshape(-1)
    return out


def _per_time(x: FloatArray) -> FloatArray:
    """Average ``[asset, horizon, sample]`` over assets and horizons, leaving one value per forecast time."""
    out: FloatArray = x.mean(axis=(0, 1))
    return out


def _ci(diff: FloatArray, groups: Sequence[str]) -> tuple[float, float]:
    d = _per_time(diff)
    return block_bootstrap(groups, lambda idx: float(np.mean(d[list(idx)])))


def report(run_dir: Path, workers: int = 8) -> dict[str, Any]:
    run = Run(run_dir)
    panel, cfg = run.panel, run.config
    val, hold, fit = cfg.split(panel, "validation"), cfg.split(panel, "holdout"), cfg.fit_times(panel)
    rv, rh = realized_bins(panel, val.times), realized_bins(panel, hold.times)
    days = day_groups(panel, hold.times)

    start, end, _ = cfg.search
    fit_range = slice(panel.index_at_or_after(parse_utc(start)), panel.index_at_or_after(parse_utc(end)))
    members = [
        Member(b.name, "baseline", None, b.probs(val.times), b.probs(hold.times))
        for b in ladder(panel.close, realized_bins(panel, fit.times), fit.times, fit_range)
    ]
    naive_val, naive_hold = log_scores(members[0].validation, rv), log_scores(members[0].holdout, rh)

    with Evaluator(run_dir / "panel.npz", workers=workers) as evaluator:
        for island in run.db.islands():
            best: tuple[float, Program, FloatArray] | None = None
            for program in run.db.best(int(island["id"]), 3):
                outcome = evaluator.run(program.code, val.times, TIME_LIMIT_S * 3)
                if outcome.probs is None:
                    continue
                gain = float(np.mean(log_scores(outcome.probs, rv) - naive_val))
                if best is None or gain > best[0]:
                    best = (gain, program, outcome.probs)
            if best is None:
                continue
            _, program, val_probs = best
            hold_outcome = evaluator.run(program.code, hold.times, TIME_LIMIT_S * 3)
            if hold_outcome.probs is None:
                continue
            members.append(
                Member(
                    f"{island['name']}#{program.id}", "champion", program.gain, val_probs, hold_outcome.probs, program
                )
            )

    champions = [m for m in members if m.kind == "champion"]
    by_name = {m.name: m for m in members}
    garch = next(m for m in members if m.name.startswith("garch"))
    har = next(m for m in members if m.name.startswith("har"))

    ensembles: dict[str, dict[str, dict[str, float]]] = {}
    for label, names in (
        ("demon(evolved)", [m.name for m in champions]),
        ("demon(evolved+classical)", [m.name for m in members if m.name != "naive"]),
    ):
        if len(names) < 2:
            continue
        result = linear_pool([by_name[n] for n in names], rv, rh)
        ensembles[label] = {
            "weights": result.weights,
            "contribution_holdout": result.contribution_holdout,
            "contribution_validation": result.contribution_validation,
        }
        members.append(Member(label, "ensemble", None, result.validation, result.holdout))

    rows: list[dict[str, Any]] = []
    for m in members:
        lh = log_scores(m.holdout, rh)
        rows.append(
            {
                "name": m.name,
                "kind": m.kind,
                "search_gain": m.search_gain,
                "validation_gain": float(np.mean(log_scores(m.validation, rv) - naive_val)),
                "holdout_gain": float(np.mean(lh - naive_hold)),
                "vs_garch": float(np.mean(lh - log_scores(garch.holdout, rh))),
                "vs_garch_ci": _ci(lh - log_scores(garch.holdout, rh), days),
                "vs_har": float(np.mean(lh - log_scores(har.holdout, rh))),
                "vs_har_ci": _ci(lh - log_scores(har.holdout, rh), days),
                "holdout_by_horizon": {
                    f"h{h}": float(np.mean((lh - naive_hold)[:, j])) for j, h in enumerate(HORIZONS)
                },
                "rps": float(np.mean(ranked_probability_scores(m.holdout, rh))),
            }
        )
    rows.sort(key=lambda r: -r["holdout_gain"])

    best_single = max((r for r in rows if r["kind"] == "champion"), key=lambda r: r["validation_gain"], default=None)
    persistence = None
    if "demon(evolved)" in ensembles and len(champions) >= 3:
        e = ensembles["demon(evolved)"]
        names = list(e["contribution_holdout"])
        persistence = spearman(
            [e["contribution_validation"][n] for n in names], [e["contribution_holdout"][n] for n in names]
        )

    summary: dict[str, Any] = {
        "run": run_dir.name,
        "config": json.loads((run_dir / "config.json").read_text()),
        "samples": {"validation": len(val.times), "holdout": len(hold.times), "holdout_days": len(set(days))},
        "programs": run.db.meta(),
        "rows": rows,
        "ensembles": ensembles,
        "best_single_by_validation": best_single["name"] if best_single else None,
        "contribution_persistence_spearman": persistence,
        "mutators": mutator_stats(run.db.all_programs()),
    }
    (run_dir / "report.json").write_text(json.dumps(summary, indent=1))
    (run_dir / "report.md").write_text(render(summary))
    champions_dir = run_dir / "champions"
    champions_dir.mkdir(exist_ok=True)
    for m in champions:
        assert m.program is not None
        (champions_dir / f"{m.name.replace('#', '-')}.py").write_text(m.program.code)
    return summary


@dataclass
class PoolResult:
    weights: dict[str, float]
    contribution_holdout: dict[str, float]
    contribution_validation: dict[str, float]
    validation: FloatArray
    holdout: FloatArray


def linear_pool(members: Sequence[Member], realized_val: np.ndarray, realized_hold: np.ndarray) -> PoolResult:
    """Fit EM weights on validation; contribution = pool score minus the score without that member."""
    pv = np.stack([realized_probs(m.validation, realized_val) for m in members])
    ph = np.stack([realized_probs(m.holdout, realized_hold) for m in members])
    w = em_weights(pv)

    def without(p: FloatArray, k: int) -> float:
        keep = np.arange(len(members)) != k
        if w[keep].sum() <= 0:
            return float("nan")
        return float(np.mean(np.log((w[keep] / w[keep].sum()) @ p[keep])))

    full_v, full_h = float(np.mean(np.log(w @ pv))), float(np.mean(np.log(w @ ph)))
    return PoolResult(
        weights={m.name: float(x) for m, x in zip(members, w, strict=True)},
        contribution_holdout={m.name: full_h - without(ph, k) for k, m in enumerate(members)},
        contribution_validation={m.name: full_v - without(pv, k) for k, m in enumerate(members)},
        validation=np.tensordot(w, np.stack([floor(m.validation) for m in members]), axes=1),
        holdout=np.tensordot(w, np.stack([floor(m.holdout) for m in members]), axes=1),
    )


def mutator_stats(programs: Sequence[Program]) -> dict[str, dict[str, float]]:
    by_id = {p.id: p for p in programs}
    out: dict[str, dict[str, float]] = {}
    for name in sorted({p.mutator for p in programs if p.mutator != "seed"}):
        mine = [p for p in programs if p.mutator == name]
        valid = [p for p in mine if p.status == "ok"]
        deltas = [
            p.gain - by_id[p.parent_id].gain  # type: ignore[operator]
            for p in valid
            if p.parent_id is not None and by_id[p.parent_id].gain is not None and p.gain is not None
        ]
        cost = sum(p.cost_usd for p in mine)
        improved = sum(d > 0 for d in deltas)
        out[name] = {
            "calls": len(mine),
            "valid_share": len(valid) / len(mine) if mine else 0.0,
            "improved_share": improved / len(mine) if mine else 0.0,
            "mean_delta": fmean(deltas) if deltas else 0.0,
            "cost_usd": cost,
            "usd_per_improvement": cost / improved if improved else float("inf"),
        }
    return out


def render(s: dict[str, Any]) -> str:
    n = s["samples"]
    lines = [
        f"# Evolution run `{s['run']}`",
        "",
        f"{s['programs']['programs']} programs ({s['programs']['valid']} valid), ${s['programs']['cost_usd']:.2f} of "
        f"mutation calls. Holdout: {n['holdout']} forecast times over {n['holdout_days']} days x 7 assets x 3 "
        f"horizons; validation: {n['validation']} times.",
        "Gains are mean log-likelihood gains in nats per question (positive = better). Champions were chosen on "
        "validation; the holdout window played no part in any choice. 95% intervals bootstrap whole days.",
        "",
        "| forecaster | kind | search | validation | holdout | vs GARCH | 95% CI | vs HAR | 95% CI | 1h | 4h | 24h |",
        "|---|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|",
    ]
    for r in s["rows"]:
        sg = "" if r["search_gain"] is None else f"{r['search_gain']:+.4f}"
        g, gh = r["vs_garch_ci"], r["vs_har_ci"]
        hb = r["holdout_by_horizon"]
        lines.append(
            f"| {r['name']} | {r['kind']} | {sg} | {r['validation_gain']:+.4f} | {r['holdout_gain']:+.4f} | "
            f"{r['vs_garch']:+.4f} | {g[0]:+.4f}…{g[1]:+.4f} | {r['vs_har']:+.4f} | {gh[0]:+.4f}…{gh[1]:+.4f} | "
            f"{hb['h1']:+.4f} | {hb['h4']:+.4f} | {hb['h24']:+.4f} |"
        )
    for label, e in s["ensembles"].items():
        lines += ["", f"**{label}** weights (fitted on validation) and holdout contribution (nats per question):", ""]
        lines += ["| member | weight | holdout contribution | validation contribution |", "|---|---:|---:|---:|"]
        for name, w in sorted(e["weights"].items(), key=lambda kv: -kv[1]):
            hold, val = e["contribution_holdout"][name], e["contribution_validation"][name]
            lines.append(f"| {name} | {w:.3f} | {hold:+.5f} | {val:+.5f} |")
    lines += [
        "",
        "Contribution persistence (Spearman, validation vs holdout, evolved demon): "
        f"{s['contribution_persistence_spearman']}",
        "",
        "| mutator | calls | valid | improved on parent | mean change | cost | $ per improvement |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in s["mutators"].items():
        lines.append(
            f"| {name} | {m['calls']} | {m['valid_share']:.0%} | {m['improved_share']:.0%} | {m['mean_delta']:+.4f} | "
            f"${m['cost_usd']:.2f} | ${m['usd_per_improvement']:.3f} |"
        )
    return "\n".join(lines) + "\n"
