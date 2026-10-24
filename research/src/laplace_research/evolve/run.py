"""Set up and run an evolution: panel, splits, islands, and the generation loop."""

from __future__ import annotations

import asyncio
import json
import random
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path

from laplace_research.evolve import panel as panel_mod
from laplace_research.evolve.baselines import naive
from laplace_research.evolve.db import Database, Program
from laplace_research.evolve.evaluate import (
    Evaluator,
    Split,
    breakdown,
    log_scores,
    mean_gain,
    realized_bins,
    sample_times,
)
from laplace_research.evolve.feed import finalize
from laplace_research.evolve.islands import ISLANDS, SEED_PROGRAM
from laplace_research.evolve.mutate import MUTATORS, NoProgram, Scored, build_prompt, propose
from laplace_research.evolve.panel import Panel
from laplace_research.inference import AuthError, Credentials, InferenceError, WorkersAI

TIME_LIMIT_S = 90
MODEL_CONCURRENCY = {
    "@cf/deepseek-ai/deepseek-v4.1-flash-1017": 4,
    "@cf/zai-org/glm-5.3": 6,
    "@cf/zai-org/glm-5.3-flash": 8,
}


def parse_utc(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


@dataclass(frozen=True)
class RunConfig:
    """Time windows (UTC, ``[start, end)``) and sampling steps. Steps of 23 and 5 hours rotate through the clock."""

    data_start: str
    data_end: str
    search: tuple[str, str, int]
    validation: tuple[str, str, int]
    holdout: tuple[str, str, int]
    baseline_fit_step: int = 2
    seed: int = 0

    @classmethod
    def default(cls, now: datetime) -> RunConfig:
        end = now.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:00")
        return cls(
            data_start="2025-01-01T00:00",
            data_end=end,
            search=("2025-02-01T00:00", "2026-06-01T00:00", 23),
            validation=("2026-06-01T00:00", "2026-08-16T00:00", 5),
            holdout=("2026-08-16T00:00", end, 3),
        )

    def split(self, panel: Panel, name: str) -> Split:
        start, end, step = getattr(self, name)
        return Split(name, sample_times(panel, parse_utc(start), parse_utc(end), step))

    def fit_times(self, panel: Panel) -> Split:
        start, end, _ = self.search
        return Split("fit", sample_times(panel, parse_utc(start), parse_utc(end), self.baseline_fit_step))


@dataclass(frozen=True)
class Score:
    """A program's result on the search window: either a gain over the naive baseline or an error."""

    gain: float | None = None
    breakdown: dict[str, float] | None = None
    error: str | None = None
    eval_seconds: float | None = None


class Run:
    def __init__(self, run_dir: Path) -> None:
        self.dir = run_dir
        self.config = RunConfig(**json.loads((run_dir / "config.json").read_text()))
        self.panel = Panel.load(run_dir / "panel.npz")
        self.db = Database(run_dir / "programs.sqlite")
        self.search = self.config.split(self.panel, "search")
        self.search_realized = realized_bins(self.panel, self.search.times)
        self.search_naive = log_scores(finalize(naive(self.panel.close).probs(self.search.times)), self.search_realized)

    @staticmethod
    def init(run_dir: Path, cache_dir: Path) -> Run:
        if (run_dir / "config.json").exists():
            raise FileExistsError(f"{run_dir} already initialized")
        run_dir.mkdir(parents=True)
        config = RunConfig.default(datetime.now(UTC) - timedelta(hours=1))
        panel = panel_mod.build(parse_utc(config.data_start), parse_utc(config.data_end), cache_dir)
        panel.save(run_dir / "panel.npz")
        (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=1))
        run = Run(run_dir)
        with Evaluator(run_dir / "panel.npz", workers=1) as evaluator:
            seed = run.score(evaluator, SEED_PROGRAM)
        if seed.error is not None:
            raise RuntimeError(f"the seed program failed: {seed.error}")
        for name, directive in ISLANDS.items():
            run.db.add_program(
                island_id=run.db.add_island(name, directive),
                parent_id=None,
                generation=0,
                mutator="seed",
                code=SEED_PROGRAM,
                gain=seed.gain,
                breakdown=seed.breakdown,
                eval_seconds=seed.eval_seconds,
            )
        return run

    def score(self, evaluator: Evaluator, code: str) -> Score:
        outcome = evaluator.run(code, self.search.times, TIME_LIMIT_S)
        if outcome.probs is None:
            return Score(error=outcome.error, eval_seconds=outcome.seconds)
        log_p = log_scores(finalize(outcome.probs), self.search_realized)
        return Score(
            gain=mean_gain(log_p, self.search_naive),
            breakdown=breakdown(log_p, self.search_naive, self.panel.assets),
            eval_seconds=outcome.seconds,
        )

    async def evolve(self, generations: int, children: int, workers: int) -> None:
        islands = self.db.islands()
        start = self.db.last_generation() + 1
        with Evaluator(self.dir / "panel.npz", workers=workers) as evaluator:
            async with WorkersAI(Credentials.from_env(), concurrency=16, model_limits=MODEL_CONCURRENCY) as ai:
                for g in range(start, start + generations):
                    jobs = [
                        self._child(ai, evaluator, island, index, g, c)
                        for index, island in enumerate(islands)
                        for c in range(children)
                    ]
                    gains = [x for x in await asyncio.gather(*jobs) if x is not None]
                    meta = self.db.meta()
                    best = f"{max(gains):+.4f}" if gains else "none"
                    print(
                        f"generation {g}: {len(gains)}/{len(jobs)} valid, best child {best}, "
                        f"{meta['programs']} programs, ${meta['cost_usd']:.2f} spent",
                        file=sys.stderr,
                        flush=True,
                    )

    async def _child(
        self, ai: WorkersAI, evaluator: Evaluator, island: sqlite3.Row, index: int, g: int, c: int
    ) -> float | None:
        rng = random.Random(self.config.seed * 1_000_003 + g * 1009 + index * 31 + c)
        island_id = int(island["id"])
        top = self.db.best(island_id, 4)
        parent = top[0] if rng.random() < 0.6 or len(top) == 1 else rng.choice(top)
        inspirations: list[Program] = []
        others = [i for i in self.db.islands() if i["id"] != island_id]
        if others and rng.random() < 0.5:
            other_best = self.db.best(int(rng.choice(others)["id"]), 1)
            inspirations = [p for p in other_best if p.code != parent.code]
        mutator = MUTATORS[_pick_mutator(g, c, index)]
        prompt = build_prompt(
            island["directive"],
            _scored(parent),
            [_scored(p) for p in inspirations],
            self.db.recent_failures(island_id, 3),
        )
        record = partial(
            self.db.add_program, island_id=island_id, parent_id=parent.id, generation=g, mutator=mutator.name
        )
        try:
            proposal = await propose(ai, mutator, prompt)
        except AuthError:
            raise
        except NoProgram as exc:  # answered but unusable: billed, so its cost is recorded
            record(code="", error=f"mutation: {exc}"[:400], cost_usd=exc.cost_usd)
            return None
        except (InferenceError, KeyError) as exc:  # no usable answer; any billing is unknown here
            record(code="", error=f"mutation: {exc}"[:400])
            return None
        if self.db.has_code(island_id, proposal.code):
            record(code=proposal.code, title=proposal.title, error="mutation: duplicate", cost_usd=proposal.cost_usd)
            return None
        score = await asyncio.to_thread(self.score, evaluator, proposal.code)
        record(
            code=proposal.code,
            title=proposal.title,
            why=proposal.why,
            cost_usd=proposal.cost_usd,
            gain=score.gain,
            breakdown=score.breakdown,
            error=score.error,
            eval_seconds=score.eval_seconds,
        )
        return score.gain


def _scored(p: Program) -> Scored:
    assert p.gain is not None
    keep = {k: v for k, v in p.breakdown.items() if k.startswith("h")}
    return Scored(p.code, p.gain, keep)


def _pick_mutator(g: int, c: int, island_index: int) -> str:
    """Every fourth generation the first child of each island gets the deep model; the rest alternate cheap ones."""
    if c == 0 and g % 4 == 3:
        return "glm-5.3"
    return "glm-5.3-flash" if (island_index + c + g) % 2 == 0 else "deepseek-v4.1-flash"
