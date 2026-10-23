import itertools
import json

import pytest

from laplace_research.cards import MOVE_BINS, WEEK, build_card, move_bin, settle
from laplace_research.data.candles import Candle


@pytest.mark.parametrize(
    ("ret", "expected"),
    [(-0.05, "lt_m3"), (-0.03, "m3_m1"), (-0.0099, "m1_p1"), (0.01, "p1_p3"), (0.03, "gt_p3")],
)
def test_move_bin_edges(ret: float, expected: str) -> None:
    assert move_bin(ret) == expected


def test_bins_cover_the_real_line_without_overlap() -> None:
    for (_, _, hi), (_, lo, _) in itertools.pairwise(MOVE_BINS):
        assert hi == lo


def test_card_is_blinded(candles: list[Candle]) -> None:
    card = build_card("BTC-USD", "A", candles, WEEK + 10)
    text = json.dumps(card.features)
    assert "BTC" not in text
    assert str(candles[WEEK + 10].ts) not in text
    assert f"{candles[WEEK + 10].close:.2f}" not in text


def test_card_uses_no_future_candles(candles: list[Candle]) -> None:
    i = WEEK + 10
    future_changed = [*candles[: i + 1], *(c.__class__(c.ts, 1, 1, 1, 1, 1) for c in candles[i + 1 :])]
    assert build_card("X", "A", candles, i).features == build_card("X", "A", future_changed, i).features


def test_card_needs_a_week_of_history(candles: list[Candle]) -> None:
    with pytest.raises(ValueError, match="week"):
        build_card("X", "A", candles, WEEK - 1)


def test_settle_reads_closes_4_and_24_hours_ahead(candles: list[Candle]) -> None:
    i = WEEK
    outcome = settle(candles, i)
    assert outcome.move24 == move_bin(candles[i + 24].close / candles[i].close - 1)
    assert outcome.big4 == (abs(candles[i + 4].close / candles[i].close - 1) > 0.02)
