import math

import pytest

from laplace_research.data.candles import HOUR, Candle


@pytest.fixture
def candles() -> list[Candle]:
    """Ten days of deterministic hourly candles with a slow wave and a small hourly wiggle."""
    out = []
    for i in range(240):
        close = 100 * math.exp(0.05 * math.sin(i / 20) + 0.004 * math.sin(i * 1.7))
        out.append(
            Candle(
                ts=1_700_000_000 - 1_700_000_000 % HOUR + i * HOUR,
                low=close * 0.995,
                high=close * 1.005,
                open=close,
                close=close,
                volume=10 + i % 7,
            )
        )
    return out
