"""Minimal async client for Workers AI, directly over REST or through an authenticated AI Gateway.

Credentials come from the environment only:
- ``LAPLACE_AI_ACCOUNT_ID`` and ``LAPLACE_AI_TOKEN`` (a token allowed to run Workers AI models);
- optionally ``LAPLACE_AI_GATEWAY`` and ``LAPLACE_AI_GATEWAY_TOKEN``: requests then go through that gateway, which
  logs them, prices them and enforces its rate and spend limits;
- optionally ``LAPLACE_AI_TOKEN_CMD``: a rejected model token is replaced by that command's output once and the request
  retried, so long runs survive short-lived tokens.

Rate limits and transient server errors are retried with exponential backoff and jitter. Rejected credentials and an
exhausted gateway budget stop the caller instead, because retrying cannot fix either.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

RETRY_STATUS = {408, 429, 500, 502, 503, 504}
API = "https://api.cloudflare.com/client/v4"
GATEWAY = "https://gateway.ai.cloudflare.com/v1"
MAX_METADATA_ENTRIES = 5  # AI Gateway accepts at most five custom metadata entries per request

Metadata = Mapping[str, str | int | float | bool]


class InferenceError(RuntimeError):
    pass


class AuthError(InferenceError):
    """The credentials were rejected. Retrying or continuing the run would only fail again."""


class BudgetExceeded(InferenceError):
    """The gateway's spend limit is reached. Raise the limit or wait; retrying now only adds load."""


@dataclass(frozen=True, slots=True)
class Credentials:
    account_id: str
    token: str
    refresh_command: str | None = None
    gateway: str | None = None
    gateway_token: str | None = None

    @classmethod
    def from_env(cls) -> Credentials:
        try:
            credentials = cls(
                os.environ["LAPLACE_AI_ACCOUNT_ID"],
                os.environ["LAPLACE_AI_TOKEN"],
                os.environ.get("LAPLACE_AI_TOKEN_CMD") or None,
                os.environ.get("LAPLACE_AI_GATEWAY") or None,
                os.environ.get("LAPLACE_AI_GATEWAY_TOKEN") or None,
            )
        except KeyError as missing:
            raise InferenceError(f"missing environment variable {missing}") from None
        if credentials.gateway and not credentials.gateway_token:
            raise InferenceError("LAPLACE_AI_GATEWAY is set but LAPLACE_AI_GATEWAY_TOKEN is not")
        return credentials


def _is_spend_limit(response: httpx.Response) -> bool:
    text = response.text.lower()
    return response.status_code in {402, 429} and any(word in text for word in ("spend", "budget", "cost limit"))


class WorkersAI:
    def __init__(
        self,
        credentials: Credentials,
        *,
        concurrency: int = 16,
        model_limits: Mapping[str, int] | None = None,
        max_attempts: int = 6,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if credentials.gateway:
            self._url = f"{GATEWAY}/{credentials.account_id}/{credentials.gateway}/workers-ai"
            self._gateway_headers = {"cf-aig-authorization": f"Bearer {credentials.gateway_token}"}
        else:
            self._url = f"{API}/accounts/{credentials.account_id}/ai/run"
            self._gateway_headers = {}
        self._gateway = credentials.gateway is not None
        self._model_token = credentials.token
        self._refresh_command = credentials.refresh_command
        self._refresh_lock = asyncio.Lock()
        self._token_version = 0
        self._semaphore = asyncio.Semaphore(concurrency)
        # Some hosted models rate-limit far below others; cap them separately.
        self._model_semaphores = {m: asyncio.Semaphore(n) for m, n in (model_limits or {}).items()}
        self._max_attempts = max_attempts
        # Reasoning models can think for several minutes before answering.
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=20.0), transport=transport)

    async def __aenter__(self) -> WorkersAI:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._client.aclose()

    async def run(self, model: str, payload: dict[str, Any], metadata: Metadata | None = None) -> dict[str, Any]:
        """Run ``model`` (e.g. ``@cf/cloudflare/clef-flash``) and return its ``result`` object.

        ``metadata`` (at most five entries) is attached to the gateway's log of the request, so cost can be attributed
        to a run, island or mutator; it is ignored without a gateway.
        """
        if metadata is not None and len(metadata) > MAX_METADATA_ENTRIES:
            raise ValueError(f"at most {MAX_METADATA_ENTRIES} metadata entries")
        model_semaphore = self._model_semaphores.get(model)
        if model_semaphore is None:
            async with self._semaphore:
                return await self._run_authenticated(model, payload, metadata)
        async with model_semaphore, self._semaphore:
            return await self._run_authenticated(model, payload, metadata)

    async def _run_authenticated(
        self, model: str, payload: dict[str, Any], metadata: Metadata | None
    ) -> dict[str, Any]:
        version = self._token_version
        try:
            return await self._run(model, payload, metadata)
        except AuthError:
            if self._refresh_command is None:
                raise
            await self._refresh(version)
            return await self._run(model, payload, metadata)

    async def _refresh(self, seen_version: int) -> None:
        async with self._refresh_lock:
            if self._token_version != seen_version:
                return  # another request already refreshed the token
            assert self._refresh_command is not None
            process = await asyncio.create_subprocess_shell(
                self._refresh_command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
            )
            stdout, _ = await process.communicate()
            token = stdout.decode().strip()
            if process.returncode or not token:
                raise AuthError("token refresh command failed")
            self._model_token = token
            self._token_version += 1

    def _headers(self, metadata: Metadata | None) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self._model_token}", **self._gateway_headers}
        if metadata and self._gateway:
            headers["cf-aig-metadata"] = json.dumps(dict(metadata))
        return headers

    async def _run(self, model: str, payload: dict[str, Any], metadata: Metadata | None) -> dict[str, Any]:
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = await self._client.post(
                    f"{self._url}/{model}", json=payload, headers=self._headers(metadata)
                )
            except httpx.TransportError as exc:
                if attempt == self._max_attempts:
                    raise InferenceError(f"{model}: {exc!r}") from exc
            else:
                if response.status_code == httpx.codes.OK:
                    body: dict[str, Any] = response.json()
                    if not body.get("success", False):
                        raise InferenceError(f"{model}: {body.get('errors')}")
                    result: dict[str, Any] = body["result"]
                    return result
                if response.status_code in {httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN}:
                    raise AuthError(f"{model}: HTTP {response.status_code}, credentials rejected")
                if _is_spend_limit(response):
                    raise BudgetExceeded(f"{model}: gateway spend limit reached ({response.text[:200]})")
                if response.status_code not in RETRY_STATUS or attempt == self._max_attempts:
                    raise InferenceError(f"{model}: HTTP {response.status_code} {response.text[:300]}")
            await asyncio.sleep(min(30.0, 2**attempt) * (0.5 + random.random()))
        raise AssertionError("unreachable")
