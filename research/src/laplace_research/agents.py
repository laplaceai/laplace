"""Forecasting agents: a model plus a directive.

Decision models (Clef, Clef-flash) take the card as ``state`` and answer typed questions with probabilities.
Chat models (DeepSeek) get the same card and must reply with a JSON object of probabilities; replies that don't
parse are recorded as failures, never patched with a default.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from laplace_research.cards import BIG_MOVE_4H, MOVE_BINS, MOVE_LABELS, Card
from laplace_research.inference import WorkersAI
from laplace_research.scoring import normalize

MODELS = {
    "clef-flash": "@cf/cloudflare/clef-flash",
    "clef": "@cf/cloudflare/clef",
    "deepseek": "@cf/deepseek-ai/deepseek-v4.1-flash-1017",
}

DIRECTIVES = {
    "none": "",
    "plain": "Forecast as accurately as you can and report honest probabilities.",
    "volatility": (
        "Direction over these horizons is close to a coin flip. Derive the probabilities from recent realized "
        "volatility and keep the up and down sides symmetric unless the data clearly says otherwise."
    ),
    "momentum": (
        "Recent trends in this asset tend to persist over the next day. Lean the probabilities in the "
        "direction of the last 24 hours and 7 days."
    ),
    "reversion": (
        "Large recent moves in this asset tend to partly reverse over the next day. Lean against the last "
        "24 hours when it was a big move."
    ),
    "fat_tails": (
        "Crypto returns have fat tails: moves several times the typical size happen more often than a normal "
        "distribution implies. Do not be overconfident in the middle bin."
    ),
}

MOVE_QUESTION = "Where will this asset's price be 24 hours from now, relative to now?"
BIG4_QUESTION = (
    f"Will this asset's price move more than {BIG_MOVE_4H:.0%} in either direction between now and 4 hours from now?"
)
MOVE_KEYS = [bin_id for bin_id, _, _ in MOVE_BINS]


@dataclass(frozen=True, slots=True)
class Agent:
    model: str
    directive: str

    @property
    def agent_id(self) -> str:
        return f"{self.model}/{self.directive}"


@dataclass(frozen=True, slots=True)
class Forecast:
    move24: dict[str, float]
    big4: float
    usage: dict[str, Any]


class ForecastParseError(ValueError):
    pass


def _state(card: Card, directive: str) -> dict[str, Any]:
    state: dict[str, Any] = {"card": card.features}
    if DIRECTIVES[directive]:
        state["forecaster_directive"] = DIRECTIVES[directive]
    return state


async def forecast(ai: WorkersAI, agent: Agent, card: Card) -> Forecast:
    if agent.model.startswith("clef"):
        return await _forecast_clef(ai, agent, card)
    return await _forecast_chat(ai, agent, card)


async def _forecast_clef(ai: WorkersAI, agent: Agent, card: Card) -> Forecast:
    result = await ai.run(
        MODELS[agent.model],
        {
            "model": agent.model,
            "state": _state(card, agent.directive),
            "questions": {
                "move24": {"type": "choice", "instructions": MOVE_QUESTION, "criteria": MOVE_LABELS},
                "big4": {"type": "noul", "instructions": BIG4_QUESTION},
            },
        },
    )
    answers = result["answers"]
    return Forecast(
        move24=normalize(answers["move24"]["probabilities"], MOVE_KEYS),
        big4=float(answers["big4"]["noul"]),
        usage=result.get("usage", {}),
    )


CHAT_FORMAT = (
    "Reply with only a JSON object, no prose: "
    '{"move24": {' + ", ".join(f'"{k}": p' for k in MOVE_KEYS) + '}, "big4": p}. '
    "move24 holds the probability of each 24-hour outcome ("
    + "; ".join(f"{k} = {v}" for k, v in MOVE_LABELS.items())
    + ") and must sum to 1. big4 is the probability of a yes to: "
    + BIG4_QUESTION
)


async def _forecast_chat(ai: WorkersAI, agent: Agent, card: Card) -> Forecast:
    system = "You are a calibrated quantitative forecaster."
    if DIRECTIVES[agent.directive]:
        system += " " + DIRECTIVES[agent.directive]
    result = await ai.run(
        MODELS[agent.model],
        {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps({"card": card.features}) + "\n\n" + CHAT_FORMAT},
            ],
            "max_completion_tokens": 2000,
            "temperature": 0.2,
        },
    )
    content = result["choices"][0]["message"]["content"] or ""
    return parse_chat_forecast(content, result.get("usage", {}))


def parse_chat_forecast(content: str, usage: dict[str, Any]) -> Forecast:
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match is None:
        raise ForecastParseError(f"no JSON object in reply: {content[:200]!r}")
    try:
        data = json.loads(match.group(0))
        move = normalize({k: float(v) for k, v in data["move24"].items()}, MOVE_KEYS)
        big4 = float(data["big4"])
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ForecastParseError(f"unusable reply: {content[:200]!r}") from exc
    if not 0.0 <= big4 <= 1.0:
        raise ForecastParseError(f"big4 out of range: {big4}")
    return Forecast(move24=move, big4=big4, usage=usage)
