"""Minimal async client for Workers AI over REST.

Credentials come from the environment only (``LAPLACE_AI_ACCOUNT_ID``, ``LAPLACE_AI_TOKEN``). Rate limits and
transient server errors are retried with exponential backoff and jitter.
"""

from __future__ import annotations

import asyncio
import os
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

RETRY_STATUS = {408, 429, 500, 502, 503, 504}


class InferenceError(RuntimeError):
    pass


class AuthError(InferenceError):
    """The credentials were rejected. Retrying or continuing the run would only fail again."""


@dataclass(frozen=True, slots=True)
class Credentials:
    account_id: str
    token: str

    @classmethod
    def from_env(cls) -> Credentials:
        try:
            return cls(os.environ["LAPLACE_AI_ACCOUNT_ID"], os.environ["LAPLACE_AI_TOKEN"])
        except KeyError as missing:
            raise InferenceError(f"missing environment variable {missing}") from None


class WorkersAI:
    def __init__(
        self,
        credentials: Credentials,
        *,
        concurrency: int = 16,
        model_limits: Mapping[str, int] | None = None,
        max_attempts: int = 6,
    ) -> None:
        self._url = f"https://api.cloudflare.com/client/v4/accounts/{credentials.account_id}/ai/run"
        self._headers = {"Authorization": f"Bearer {credentials.token}"}
        self._semaphore = asyncio.Semaphore(concurrency)
        # Some hosted models rate-limit far below others; cap them separately.
        self._model_semaphores = {m: asyncio.Semaphore(n) for m, n in (model_limits or {}).items()}
        self._max_attempts = max_attempts
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))

    async def __aenter__(self) -> WorkersAI:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._client.aclose()

    async def run(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Run ``model`` (e.g. ``@cf/cloudflare/clef-flash``) and return its ``result`` object."""
        model_semaphore = self._model_semaphores.get(model)
        if model_semaphore is None:
            async with self._semaphore:
                return await self._run(model, payload)
        async with model_semaphore, self._semaphore:
            return await self._run(model, payload)

    async def _run(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = await self._client.post(f"{self._url}/{model}", json=payload, headers=self._headers)
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
                if response.status_code not in RETRY_STATUS or attempt == self._max_attempts:
                    raise InferenceError(f"{model}: HTTP {response.status_code} {response.text[:300]}")
            await asyncio.sleep(min(30.0, 2**attempt) * (0.5 + random.random()))
        raise AssertionError("unreachable")
