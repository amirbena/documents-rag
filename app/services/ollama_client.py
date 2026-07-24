"""Async client for Ollama health and model-availability checks only.

Deliberately does not call `/api/generate` or `/api/embeddings` — this milestone only
verifies that Ollama is reachable and that the configured models are pulled, so it stays
usable by a health endpoint without pulling generation/embedding logic forward.
"""

from dataclasses import dataclass

import httpx

from app.core.config import Settings, get_settings


@dataclass
class OllamaHealthResult:
    """Outcome of a single Ollama reachability + model-availability check."""

    reachable: bool
    chat_model_available: bool
    embedding_model_available: bool
    error: str | None = None


class OllamaClient:
    """Thin async HTTP client that talks to OLLAMA_BASE_URL for status checks."""

    def __init__(
        self,
        settings: Settings | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._transport = transport

    async def check_health(self) -> OllamaHealthResult:
        """Check Ollama reachability and whether the configured chat/embedding models exist."""
        try:
            async with httpx.AsyncClient(
                base_url=self._settings.ollama_base_url,
                timeout=self._settings.ollama_health_timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.get("/api/tags")
                response.raise_for_status()
        except httpx.HTTPError as exc:
            return OllamaHealthResult(
                reachable=False,
                chat_model_available=False,
                embedding_model_available=False,
                error=str(exc),
            )

        available_models = [model.get("name", "") for model in response.json().get("models", [])]
        return OllamaHealthResult(
            reachable=True,
            chat_model_available=self._model_present(self._settings.ollama_chat_model, available_models),
            embedding_model_available=self._model_present(
                self._settings.ollama_embedding_model, available_models
            ),
        )

    @staticmethod
    def _model_present(model_name: str, available_models: list[str]) -> bool:
        """Match a configured model name against Ollama tags, ignoring the `:tag` suffix."""
        return any(name == model_name or name.startswith(f"{model_name}:") for name in available_models)
