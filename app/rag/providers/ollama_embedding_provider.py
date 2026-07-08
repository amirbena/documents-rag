"""Ollama-backed implementation of EmbeddingProvider.

Calls only Ollama's `POST /api/embeddings` with OLLAMA_EMBEDDING_MODEL — no generation calls,
no ingestion, no Qdrant writes. This is the embedding half of the RAG pipeline in isolation.
"""

import httpx

from app.core.config import Settings, get_settings
from app.rag.providers.embedding_provider import EmbeddingProvider


class OllamaEmbeddingError(Exception):
    """Raised when Ollama is unreachable, returns an error, or responds unexpectedly."""


class OllamaEmbeddingProvider(EmbeddingProvider):
    """EmbeddingProvider that calls Ollama's /api/embeddings for OLLAMA_EMBEDDING_MODEL."""

    def __init__(
        self,
        settings: Settings | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._transport = transport

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in the same order."""
        return await self.embed_texts(texts)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed each text in turn (Ollama's /api/embeddings takes one prompt per call)."""
        return [await self.embed_text(text) for text in texts]

    async def embed_text(self, text: str) -> list[float]:
        """Return the embedding vector for a single piece of text."""
        if not text or not text.strip():
            raise ValueError("text must not be empty")

        try:
            async with httpx.AsyncClient(
                base_url=self._settings.ollama_base_url,
                timeout=30.0,
                transport=self._transport,
            ) as client:
                response = await client.post(
                    "/api/embeddings",
                    json={"model": self._settings.ollama_embedding_model, "prompt": text},
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise OllamaEmbeddingError(
                f"Ollama returned {exc.response.status_code} for /api/embeddings"
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaEmbeddingError(f"Ollama unreachable at /api/embeddings: {exc}") from exc

        try:
            embedding = response.json()["embedding"]
        except (ValueError, KeyError, TypeError) as exc:
            raise OllamaEmbeddingError("Malformed embedding response from Ollama") from exc

        if not isinstance(embedding, list) or not all(isinstance(v, int | float) for v in embedding):
            raise OllamaEmbeddingError("Malformed embedding response from Ollama")

        return embedding
