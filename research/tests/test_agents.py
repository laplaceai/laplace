import pytest

from laplace_research.agents import CHAT_FORMAT, MOVE_KEYS, ForecastParseError, parse_chat_forecast


def test_parse_chat_forecast_normalizes_and_tolerates_surrounding_text() -> None:
    reply = 'Here you go: {"move24": {"lt_m3": 1, "m3_m1": 1, "m1_p1": 2, "p1_p3": 0, "gt_p3": 0}, "big4": 0.1}'
    f = parse_chat_forecast(reply, {})
    assert f.move24["m1_p1"] == pytest.approx(0.5)
    assert f.big4 == pytest.approx(0.1)


@pytest.mark.parametrize(
    "reply",
    ["no json here", '{"move24": "x", "big4": 0.1}', '{"move24": {"m1_p1": 1}, "big4": 2}'],
)
def test_parse_chat_forecast_rejects_unusable_replies(reply: str) -> None:
    with pytest.raises(ForecastParseError):
        parse_chat_forecast(reply, {})


def test_chat_format_names_every_bin() -> None:
    assert all(k in CHAT_FORMAT for k in MOVE_KEYS)
