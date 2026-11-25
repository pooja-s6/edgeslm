"""Async Ollama REST client with real timing extraction from /api/generate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import OLLAMA_BASE_URL, OLLAMA_REQUEST_TIMEOUT_S


@dataclass(frozen=True)
class GenerateTelemetry:
    """Timing metrics parsed from a non-streaming Ollama generate response."""

    response: str
    load_time_ms: float
    ttft_ms: float
    decode_time_ms: float
    tokens_per_sec: float
    total_time_ms: float
    eval_count: int
    prompt_eval_count: int
    raw: dict[str, Any]


def parse_generate_telemetry(payload: dict[str, Any]) -> GenerateTelemetry:
    """
    Convert Ollama nanosecond duration fields into ms / tokens-per-sec.

    Uses only fields present in the real `/api/generate` JSON response.
    """
    load_duration_ns = float(payload.get("load_duration") or 0)
    prompt_eval_duration_ns = float(payload.get("prompt_eval_duration") or 0)
    eval_duration_ns = float(payload.get("eval_duration") or 0)
    total_duration_ns = float(payload.get("total_duration") or 0)
    eval_count = int(payload.get("eval_count") or 0)
    prompt_eval_count = int(payload.get("prompt_eval_count") or 0)

    load_time_ms = load_duration_ns / 1e6
    ttft_ms = prompt_eval_duration_ns / 1e6
    decode_time_ms = eval_duration_ns / 1e6
    total_time_ms = total_duration_ns / 1e6

    if eval_duration_ns > 0 and eval_count > 0:
        tokens_per_sec = eval_count / (eval_duration_ns / 1e9)
    else:
        tokens_per_sec = 0.0

    return GenerateTelemetry(
        response=str(payload.get("response") or ""),
        load_time_ms=load_time_ms,
        ttft_ms=ttft_ms,
        decode_time_ms=decode_time_ms,
        tokens_per_sec=tokens_per_sec,
        total_time_ms=total_time_ms,
        eval_count=eval_count,
        prompt_eval_count=prompt_eval_count,
        raw=payload,
    )


class OllamaClient:
    """Thin async wrapper around the local Ollama HTTP API."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        timeout: float = OLLAMA_REQUEST_TIMEOUT_S,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = client
        self._owns_client = client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Return True if Ollama responds to a lightweight tags request."""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            return response.status_code == 200
        except (httpx.HTTPError, OSError):
            return False

    async def list_local_models(self) -> list[dict[str, Any]]:
        """List models installed locally via `/api/tags`."""
        client = await self._get_client()
        response = await client.get("/api/tags")
        response.raise_for_status()
        data = response.json()
        return list(data.get("models") or [])

    async def get_loaded_models(self) -> list[dict[str, Any]]:
        """List currently loaded runners via `/api/ps`."""
        client = await self._get_client()
        response = await client.get("/api/ps")
        response.raise_for_status()
        data = response.json()
        return list(data.get("models") or [])

    async def generate_with_telemetry(
        self,
        model: str,
        prompt: str,
        options: dict[str, Any] | None = None,
    ) -> GenerateTelemetry:
        """
        Call `/api/generate` with stream=False and parse real timing metrics.
        """
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if options:
            body["options"] = options

        client = await self._get_client()
        response = await client.post("/api/generate", json=body)
        response.raise_for_status()
        payload = response.json()
        return parse_generate_telemetry(payload)
