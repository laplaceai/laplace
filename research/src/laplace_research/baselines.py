"""Reference forecasters every agent is measured against.

- ``uniform``: equal odds per bin, the score of knowing nothing.
- ``base_rate``: the asset's own outcome frequencies over the previous 30 days, add-one smoothed (Laplace's rule of
  succession), using only outcomes already settled at card time.
- ``laplace_vol``: a zero-mean Laplace distribution of log returns whose scale comes from the last week's realized
  volatility. Crypto returns are fat-tailed, which a Laplace distribution captures better than a normal one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from laplace_research.cards import BIG_MOVE_4H, MOVE_BINS, WEEK, Outcome, hourly_log_returns
from laplace_research.data.candles import Candle

Forecast = dict[str, float]
BINARY_BIN_COUNT = 2


def uniform() -> tuple[Forecast, float]:
    return {bin_id: 1 / len(MOVE_BINS) for bin_id, _, _ in MOVE_BINS}, 0.5


def base_rate(history: Sequence[Outcome]) -> tuple[Forecast, float]:
    counts = {bin_id: 1 for bin_id, _, _ in MOVE_BINS}
    big = 1
    for outcome in history:
        counts[outcome.move24] += 1
        big += outcome.big4
    total = sum(counts.values())
    return (
        {k: v / total for k, v in counts.items()},
        big / (len(history) + BINARY_BIN_COUNT),
    )


def laplace_cdf(x: float, scale: float) -> float:
    if x < 0:
        return 0.5 * math.exp(x / scale)
    return 1 - 0.5 * math.exp(-x / scale)


def laplace_vol(candles: Sequence[Candle], i: int) -> tuple[Forecast, float]:
    closes = [c.close for c in candles[i - WEEK : i + 1]]
    sigma_hour = math.sqrt(sum(r * r for r in hourly_log_returns(closes)) / WEEK)
    # A Laplace distribution with scale b has standard deviation b * sqrt(2).
    b24 = sigma_hour * math.sqrt(24) / math.sqrt(2)
    b4 = sigma_hour * math.sqrt(4) / math.sqrt(2)

    def cdf(simple_return: float) -> float:
        if math.isinf(simple_return):
            return 0.0 if simple_return < 0 else 1.0
        return laplace_cdf(math.log1p(simple_return), b24)

    move = {bin_id: cdf(hi) - cdf(lo) for bin_id, lo, hi in MOVE_BINS}
    k = math.log1p(BIG_MOVE_4H)
    big4 = 0.5 * math.exp(-k / b4) + 0.5 * math.exp(math.log1p(-BIG_MOVE_4H) / b4)
    return move, big4
