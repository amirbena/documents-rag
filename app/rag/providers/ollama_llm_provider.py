"""Streaming Ollama-backed implementation of LLMProvider.

Calls only Ollama's `POST /api/generate` with `stream=true` and yields text chunks as they
arrive — no chat endpoint, no SSE, no ingestion, no Qdrant writes. Internal provider only;
not yet wired to any API route.
"""

import json
from collections.abc import AsyncIterator

import httpx

from app.core.config import Settings, get_settings
from app.core.correlation import correlation_headers
from app.rag.providers.llm_provider import LLMProvider

# Category (Phase 2.10, see app/core/errors.py): ProviderError.


class OllamaLLMError(Exception):
    """Raised when Ollama is unreachable, returns an error, or the stream is malformed."""


class OllamaLLMProvider(LLMProvider):
    """LLMProvider that streams completions from Ollama's /api/generate for the configured model.

    Uses Settings.resolved_llm_model — LLM_MODEL if set, else OLLAMA_CHAT_MODEL — so the model
    can be changed independently of LLM_PROVIDER.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._transport = transport

    async def generate(self, prompt: str) -> str:
        """Return the full completion by collecting all streamed chunks."""
        return "".join([chunk async for chunk in self.stream_generate(prompt)])

    async def stream_generate(self, prompt: str) -> AsyncIterator[str]:
        """Yield text chunks as they stream from Ollama, stopping once `done: true` arrives."""
        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty")

        async with httpx.AsyncClient(
            base_url=self._settings.ollama_base_url,
            timeout=self._settings.ollama_llm_timeout_seconds,
            transport=self._transport,
        ) as client:
            try:
                async with client.stream(
                    "POST",
                    "/api/generate",
                    json={
                        "model": self._settings.resolved_llm_model,
                        "prompt": prompt,
                        "stream": True,
                    },
                    headers=correlation_headers(),
                ) as response:
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        raise OllamaLLMError(
                            f"Ollama returned {exc.response.status_code} for /api/generate"
                        ) from exc

                    done = False
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue

                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise OllamaLLMError(f"Malformed JSON line from Ollama: {line!r}") from exc

                        chunk = payload.get("response")
                        if chunk:
                            yield chunk

                        if payload.get("done"):
                            done = True
                            break

                    if not done:
                        raise OllamaLLMError("Ollama stream ended before done=true")
            except httpx.HTTPError as exc:
                raise OllamaLLMError(f"Ollama unreachable at /api/generate: {exc}") from exc
