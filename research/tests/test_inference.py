import asyncio
import json

import httpx
import pytest

from laplace_research import inference
from laplace_research.inference import AuthError, BudgetExceeded, Credentials, InferenceError, WorkersAI

OK = {"success": True, "result": {"response": "ok"}}


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant(_: float) -> None:
        return None

    monkeypatch.setattr(inference.asyncio, "sleep", instant)


def _client(credentials: Credentials, responses: list[httpx.Response], seen: list[httpx.Request]) -> WorkersAI:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return responses.pop(0)

    return WorkersAI(credentials, transport=httpx.MockTransport(handler))


async def _run(ai: WorkersAI, metadata: dict[str, str | int] | None = None) -> dict[str, object]:
    async with ai:
        return await ai.run("@cf/test/model", {"prompt": "hi"}, metadata)


def test_gateway_requests_carry_both_tokens_and_the_metadata() -> None:
    seen: list[httpx.Request] = []
    ai = _client(
        Credentials("acct", "model-token", gateway="lab", gateway_token="gw-token"),
        [httpx.Response(200, json=OK)],
        seen,
    )
    assert asyncio.run(_run(ai, {"run": "r1", "generation": 3})) == {"response": "ok"}
    request = seen[0]
    assert str(request.url) == "https://gateway.ai.cloudflare.com/v1/acct/lab/workers-ai/@cf/test/model"
    assert request.headers["authorization"] == "Bearer model-token"
    assert request.headers["cf-aig-authorization"] == "Bearer gw-token"
    assert json.loads(request.headers["cf-aig-metadata"]) == {"run": "r1", "generation": 3}


def test_without_a_gateway_requests_go_to_the_rest_api_and_ignore_metadata() -> None:
    seen: list[httpx.Request] = []
    ai = _client(Credentials("acct", "model-token"), [httpx.Response(200, json=OK)], seen)
    asyncio.run(_run(ai, {"run": "r1"}))
    assert str(seen[0].url) == "https://api.cloudflare.com/client/v4/accounts/acct/ai/run/@cf/test/model"
    assert "cf-aig-authorization" not in seen[0].headers
    assert "cf-aig-metadata" not in seen[0].headers


def test_spend_limit_stops_without_retrying() -> None:
    seen: list[httpx.Request] = []
    limited = httpx.Response(429, text="Gateway spend limit exceeded")
    ai = _client(Credentials("acct", "t", gateway="lab", gateway_token="g"), [limited], seen)
    with pytest.raises(BudgetExceeded):
        asyncio.run(_run(ai))
    assert len(seen) == 1


def test_rate_limits_are_retried() -> None:
    seen: list[httpx.Request] = []
    ai = _client(Credentials("acct", "t"), [httpx.Response(429, text="slow down"), httpx.Response(200, json=OK)], seen)
    assert asyncio.run(_run(ai)) == {"response": "ok"}
    assert len(seen) == 2


def test_a_rejected_model_token_is_refreshed_once() -> None:
    seen: list[httpx.Request] = []
    responses = [httpx.Response(401, text="no"), httpx.Response(200, json=OK)]
    ai = _client(Credentials("acct", "old", refresh_command="echo new"), responses, seen)
    assert asyncio.run(_run(ai)) == {"response": "ok"}
    assert [r.headers["authorization"] for r in seen] == ["Bearer old", "Bearer new"]


def test_rejected_credentials_without_a_refresh_command_stop() -> None:
    ai = _client(Credentials("acct", "t"), [httpx.Response(403, text="no")], [])
    with pytest.raises(AuthError):
        asyncio.run(_run(ai))


def test_gateway_without_its_token_is_a_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAPLACE_AI_ACCOUNT_ID", "acct")
    monkeypatch.setenv("LAPLACE_AI_TOKEN", "t")
    monkeypatch.setenv("LAPLACE_AI_GATEWAY", "lab")
    monkeypatch.delenv("LAPLACE_AI_GATEWAY_TOKEN", raising=False)
    with pytest.raises(InferenceError, match="GATEWAY_TOKEN"):
        Credentials.from_env()


def test_too_much_metadata_is_refused() -> None:
    ai = _client(Credentials("acct", "t", gateway="lab", gateway_token="g"), [], [])
    with pytest.raises(ValueError, match="metadata"):
        asyncio.run(_run(ai, {str(i): i for i in range(6)}))
