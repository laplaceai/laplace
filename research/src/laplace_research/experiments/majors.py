"""PoC 1: blinded Majors backtest.

For BTC, ETH and SOL, a card every ``step_hours`` over the last ``days`` days asks two questions: the 24-hour move bin
and whether the price moves more than 2% within 4 hours. Agents (model x directive) answer every card; baselines and
outcomes are computed from the same candles. Forecasts are appended to ``forecasts.jsonl`` as they arrive, so an
interrupted run resumes where it stopped.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from laplace_research import baselines
from laplace_research.agents import Agent, ForecastParseError, forecast
from laplace_research.cards import WEEK, Card, Outcome, build_card, settle
from laplace_research.data.candles import HOUR, Candle, fetch_hourly
from laplace_research.inference import AuthError, Credentials, InferenceError, WorkersAI

ASSETS = {"BTC-USD": "A", "ETH-USD": "B", "SOL-USD": "C"}
BASE_RATE_DAYS = 30


def default_agents() -> list[Agent]:
    directives = ["none", "plain", "volatility", "momentum", "reversion", "fat_tails"]
    agents = [Agent(model, d) for model in ("clef-flash", "clef") for d in directives]
    agents += [Agent("deepseek", d) for d in ("plain", "volatility")]
    agents.append(Agent("deepseek-fast", "plain"))
    return agents


def build_cards(candles: dict[str, list[Candle]], start: datetime, step_hours: int) -> list[dict[str, Any]]:
    """Cards with their outcome and baseline forecasts, in time order."""
    rows: list[dict[str, Any]] = []
    for asset, series in candles.items():
        index = {c.ts: i for i, c in enumerate(series)}
        settled: list[tuple[int, Outcome]] = []
        for i in range(WEEK, len(series) - 24):
            settled.append((series[i].ts, settle(series, i)))
        for i, candle in enumerate(series):
            ts = candle.ts
            if ts < start.timestamp() or i + 24 >= len(series) or (ts // HOUR) % step_hours:
                continue
            card = build_card(asset, ASSETS[asset], series, i)
            # Only outcomes fully settled by card time feed the base rate.
            history = [o for t, o in settled if ts - (BASE_RATE_DAYS * 24 + 24) * HOUR <= t <= ts - 24 * HOUR]
            assert index[ts] == i
            uni, uni4 = baselines.uniform()
            br, br4 = baselines.base_rate(history)
            lv, lv4 = baselines.laplace_vol(series, i)
            rows.append(
                {
                    "card": asdict(card),
                    "outcome": asdict(settle(series, i)),
                    "baselines": {
                        "uniform": {"move24": uni, "big4": uni4},
                        "base_rate": {"move24": br, "big4": br4},
                        "laplace_vol": {"move24": lv, "big4": lv4},
                    },
                }
            )
    rows.sort(key=lambda r: (r["card"]["ts"], r["card"]["asset"]))
    return rows


def load_candles(days: int, cache_dir: Path) -> dict[str, list[Candle]]:
    end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days + BASE_RATE_DAYS + 9)
    with httpx.Client(timeout=30.0, headers={"User-Agent": "laplace-research"}) as client:
        return {asset: fetch_hourly(asset, start, end, client=client, cache_dir=cache_dir) for asset in ASSETS}


def _done(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if "error" not in row:
            done.add((row["agent"], row["card_id"]))
    return done


async def _run_agents(rows: Sequence[dict[str, Any]], agents: Iterable[Agent], out: Path, concurrency: int) -> None:
    done = _done(out)
    jobs = [
        (agent, Card(**row["card"]))
        for agent in agents
        for row in rows
        if (agent.agent_id, row["card"]["card_id"]) not in done
    ]
    print(f"{len(jobs)} forecasts to run ({len(done)} already done)", file=sys.stderr)
    async with WorkersAI(Credentials.from_env(), concurrency=concurrency) as ai:
        with out.open("a") as fh:

            async def one(agent: Agent, card: Card) -> None:
                row: dict[str, Any] = {"agent": agent.agent_id, "card_id": card.card_id}
                try:
                    f = await forecast(ai, agent, card)
                    row |= {"move24": f.move24, "big4": f.big4, "usage": f.usage}
                except AuthError:
                    raise
                except (InferenceError, ForecastParseError) as exc:
                    row["error"] = str(exc)[:500]
                fh.write(json.dumps(row) + "\n")
                fh.flush()

            for n, chunk_start in enumerate(range(0, len(jobs), 200)):
                await asyncio.gather(*(one(a, c) for a, c in jobs[chunk_start : chunk_start + 200]))
                print(f"  {min(len(jobs), (n + 1) * 200)}/{len(jobs)}", file=sys.stderr)


def run(run_dir: Path, *, days: int, step_hours: int, concurrency: int, agents: Sequence[Agent]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    cards_path = run_dir / "cards.jsonl"
    if cards_path.exists():
        rows = [json.loads(line) for line in cards_path.read_text().splitlines()]
    else:
        candles = load_candles(days, run_dir.parent.parent / ".cache" / "candles")
        start = max(s[-1].ts for s in candles.values()) - days * 24 * HOUR
        rows = build_cards(candles, datetime.fromtimestamp(start, UTC), step_hours)
        cards_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} cards in {run_dir}", file=sys.stderr)
    asyncio.run(_run_agents(rows, agents, run_dir / "forecasts.jsonl", concurrency))
