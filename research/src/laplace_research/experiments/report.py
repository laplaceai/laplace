"""Score a Majors run: skill vs baselines with day-block bootstrap intervals, error diversity, ensembles,
persistence and calibration-in-the-large. Writes ``report.md`` and ``summary.json`` next to the run."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from statistics import fmean
from typing import Any

from laplace_research.learning import fit_shrink, hedge
from laplace_research.scoring import binary_log_loss, block_bootstrap, brier, log_loss, skill

REFERENCE = "laplace_vol"


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    mx, my = fmean(xs), fmean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / (sxx * syy) ** 0.5 if sxx and syy else 0.0


def _ranks(xs: Sequence[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    for r, i in enumerate(order):
        ranks[i] = float(r)
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    return _pearson(_ranks(xs), _ranks(ys))


def linear_pool(members: Sequence[dict[str, Any]]) -> dict[str, Any]:
    keys = members[0]["move24"].keys()
    return {
        "move24": {k: fmean(m["move24"][k] for m in members) for k in keys},
        "big4": fmean(m["big4"] for m in members),
    }


def load(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]], int]:
    cards = [json.loads(line) for line in (run_dir / "cards.jsonl").read_text().splitlines()]
    forecasts: dict[str, dict[str, dict[str, Any]]] = {}
    failed: set[tuple[str, str]] = set()
    for line in (run_dir / "forecasts.jsonl").read_text().splitlines():
        row = json.loads(line)
        if "error" in row:
            failed.add((row["agent"], row["card_id"]))
        else:
            forecasts.setdefault(row["agent"], {})[row["card_id"]] = row
    # A failure that a resumed run later answered is not a failure.
    errors = sum(1 for agent, card_id in failed if card_id not in forecasts.get(agent, {}))
    return cards, forecasts, errors


def report(run_dir: Path) -> dict[str, Any]:
    cards, by_agent, errors = load(run_dir)
    agents = sorted(by_agent)
    # Compare everyone on the cards every agent answered.
    common = [c for c in cards if all(c["card"]["card_id"] in by_agent[a] for a in agents)]
    ids = [c["card"]["card_id"] for c in common]
    days = [datetime.fromtimestamp(c["card"]["ts"], UTC).strftime("%Y-%m-%d") for c in common]
    half = len(common) // 2

    forecasters: dict[str, list[dict[str, Any]]] = {
        name: [c["baselines"][name] for c in common] for name in ("uniform", "base_rate", REFERENCE)
    }
    for a in agents:
        forecasters[a] = [by_agent[a][i] for i in ids]
    models = sorted({a.split("/")[0] for a in agents})
    pools = {f"pool:{m}": [a for a in agents if a.startswith(m + "/")] for m in models}
    pools["pool:all"] = agents
    for name, members in pools.items():
        forecasters[name] = [linear_pool([forecasters[m][j] for m in members]) for j in range(len(ids))]

    # Learning layers. Shrinkage is fitted on the first half; hedge weights only use settled outcomes.
    outcomes = [c["outcome"]["move24"] for c in common]
    reference = [f["move24"] for f in forecasters[REFERENCE]]
    shrink_weights: dict[str, float] = {}
    for a in agents:
        moves = [f["move24"] for f in forecasters[a]]
        w = fit_shrink(moves[:half], reference[:half], outcomes[:half], log_loss)
        shrink_weights[a] = w
        forecasters[f"shrunk:{a}"] = [
            {"move24": {k: w * m[k] + (1 - w) * r[k] for k in m}, "big4": f["big4"]}
            for m, r, f in zip(moves, reference, forecasters[a], strict=True)
        ]
    times = [c["card"]["ts"] for c in common]
    for label, names in (("hedge:agents", agents), ("hedge:agents+reference", [*agents, REFERENCE])):
        pooled = hedge({n: [f["move24"] for f in forecasters[n]] for n in names}, outcomes, times, 24 * 3600, log_loss)
        forecasters[label] = [
            {"move24": p, "big4": f["big4"]} for p, f in zip(pooled, forecasters[REFERENCE], strict=True)
        ]

    ll: dict[str, list[float]] = {}
    ll4: dict[str, list[float]] = {}
    rows: list[dict[str, Any]] = []
    for name, fs in forecasters.items():
        ll[name] = [log_loss(f["move24"], c["outcome"]["move24"]) for f, c in zip(fs, common, strict=True)]
        ll4[name] = [binary_log_loss(f["big4"], c["outcome"]["big4"]) for f, c in zip(fs, common, strict=True)]
    ref, ref4 = ll[REFERENCE], ll4[REFERENCE]
    big4_rate = fmean(c["outcome"]["big4"] for c in common)
    for name, fs in forecasters.items():

        def sk(idx: Sequence[int], a: list[float] = ll[name], r: list[float] = ref) -> float:
            return skill(fmean(a[i] for i in idx), fmean(r[i] for i in idx))

        def sk4(idx: Sequence[int], a: list[float] = ll4[name], r: list[float] = ref4) -> float:
            return skill(fmean(a[i] for i in idx), fmean(r[i] for i in idx))

        everything = range(len(common))
        rows.append(
            {
                "forecaster": name,
                "logloss_move24": fmean(ll[name]),
                "skill_move24": sk(everything),
                "skill_move24_ci": block_bootstrap(days, sk),
                "skill_move24_first_half": sk(range(half)),
                "skill_move24_second_half": sk(range(half, len(common))),
                "brier_move24": fmean(
                    brier(f["move24"], c["outcome"]["move24"]) for f, c in zip(fs, common, strict=True)
                ),
                "logloss_big4": fmean(ll4[name]),
                "skill_big4": sk4(everything),
                "skill_big4_ci": block_bootstrap(days, sk4),
                "mean_p_big4": fmean(f["big4"] for f in fs),
            }
        )
    rows.sort(key=lambda r: -r["skill_move24"])

    excess = {a: [x - y for x, y in zip(ll[a], ref, strict=True)] for a in agents}
    within: list[float] = []
    across: list[float] = []
    for a, b in combinations(agents, 2):
        (within if a.split("/")[0] == b.split("/")[0] else across).append(_pearson(excess[a], excess[b]))
    by_name = {r["forecaster"]: r for r in rows}
    summary: dict[str, Any] = {
        "run": run_dir.name,
        "cards": len(cards),
        "cards_scored": len(common),
        "failed_forecasts": errors,
        "days": len(set(days)),
        "big4_rate": big4_rate,
        "reference": REFERENCE,
        "forecasters": rows,
        "error_correlation": {
            "within_model": fmean(within) if within else None,
            "across_models": fmean(across) if across else None,
        },
        "persistence_spearman": spearman(
            [by_name[a]["skill_move24_first_half"] for a in agents],
            [by_name[a]["skill_move24_second_half"] for a in agents],
        ),
        "shrink_weights": shrink_weights,
        "usage_input_tokens": sum(
            int(f.get("usage", {}).get("input_tokens") or f.get("usage", {}).get("prompt_tokens") or 0)
            for a in agents
            for f in by_agent[a].values()
        ),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (run_dir / "report.md").write_text(render(summary))
    return summary


def render(s: dict[str, Any]) -> str:
    lines = [
        f"# Run `{s['run']}`",
        "",
        f"{s['cards_scored']} cards scored of {s['cards']} ({s['days']} days, 3 assets); "
        f"{s['failed_forecasts']} failed forecasts; >2%-in-4h rate {s['big4_rate']:.1%}.",
        f"Skill is relative to `{s['reference']}` (0 = equal, positive = better); 95% day-block bootstrap.",
        "",
        "| forecaster | skill 24h | 95% CI | 1st half | 2nd half | skill 4h | 95% CI | mean p(4h) |",
        "|---|---:|---|---:|---:|---:|---|---:|",
    ]
    for r in s["forecasters"]:
        lo, hi = r["skill_move24_ci"]
        lo4, hi4 = r["skill_big4_ci"]
        lines.append(
            f"| {r['forecaster']} | {r['skill_move24']:+.3f} | {lo:+.3f}…{hi:+.3f} | "
            f"{r['skill_move24_first_half']:+.3f} | {r['skill_move24_second_half']:+.3f} | "
            f"{r['skill_big4']:+.3f} | {lo4:+.3f}…{hi4:+.3f} | {r['mean_p_big4']:.3f} |"
        )
    ec = s["error_correlation"]

    def fmt(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.2f}"

    lines += [
        "",
        f"Mean correlation of per-card excess loss: within a model {fmt(ec['within_model'])}, "
        f"across models {fmt(ec['across_models'])}.",
        f"Persistence (Spearman, agent skill 1st half vs 2nd half): {s['persistence_spearman']:+.2f}.",
        f"Input tokens: {s['usage_input_tokens']:,}.",
        "",
        "Shrinkage weight on each agent (fitted on the 1st half; 0 = only the reference survives): "
        + ", ".join(f"{a} {w:.2f}" for a, w in sorted(s["shrink_weights"].items(), key=lambda x: -x[1]))
        + ". Judge the shrunk:* rows on the 2nd-half column only.",
        "",
    ]
    return "\n".join(lines)
