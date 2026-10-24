"""Minimal async client for Workers AI over REST.

Credentials come from the environment only (``LAPLACE_AI_ACCOUNT_ID``, ``LAPLACE_AI_TOKEN``). If
``LAPLACE_AI_TOKEN_CMD`` is set, a rejected token is replaced by that command's output once and the request retried,
so long runs survive short-lived tokens. Rate limits and transient server errors are retried with exponential backoff
and jitter.
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
    refresh_command: str | None = None

    @classmethod
    def from_env(cls) -> Credentials:
        try:
            return cls(
                os.environ["LAPLACE_AI_ACCOUNT_ID"],
                os.environ["LAPLACE_AI_TOKEN"],
                os.environ.get("LAPLACE_AI_TOKEN_CMD") or None,
            )
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
        self._refresh_command = credentials.refresh_command
        self._refresh_lock = asyncio.Lock()
        self._token_version = 0
        self._semaphore = asyncio.Semaphore(concurrency)
        # Some hosted models rate-limit far below others; cap them separately.
        self._model_semaphores = {m: asyncio.Semaphore(n) for m, n in (model_limits or {}).items()}
        self._max_attempts = max_attempts
        # Reasoning models can think for several minutes before answering.
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=20.0))

    async def __aenter__(self) -> WorkersAI:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._client.aclose()

    async def run(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Run ``model`` (e.g. ``@cf/cloudflare/clef-flash``) and return its ``result`` object."""
        model_semaphore = self._model_semaphores.get(model)
        if model_semaphore is None:
            async with self._semaphore:
                return await self._run_authenticated(model, payload)
        async with model_semaphore, self._semaphore:
            return await self._run_authenticated(model, payload)

    async def _run_authenticated(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        version = self._token_version
        try:
            return await self._run(model, payload)
        except AuthError:
            if self._refresh_command is None:
                raise
            await self._refresh(version)
            return await self._run(model, payload)

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
            self._headers = {"Authorization": f"Bearer {token}"}
            self._token_version += 1

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
