"""Tests for OllamaEmbeddingProvider with a mocked Ollama HTTP transport."""

import json

import httpx
import pytest

from app.core.config import Settings, get_settings
from app.rag.providers.ollama_embedding_provider import OllamaEmbeddingError, OllamaEmbeddingProvider


def _provider(transport: httpx.MockTransport) -> OllamaEmbeddingProvider:
    return OllamaEmbeddingProvider(settings=get_settings(), transport=transport)


def _success_transport(embedding: list[float]) -> httpx.MockTransport:
    """Build a mock transport whose /api/embeddings response returns the given vector."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embedding": embedding})

    return httpx.MockTransport(handler)


async def test_embed_text_returns_vector_on_success() -> None:
    """A successful response should return the embedding vector unchanged."""
    provider = _provider(_success_transport([0.1, 0.2, 0.3]))

    result = await provider.embed_text("hello world")

    assert result == [0.1, 0.2, 0.3]


async def test_embed_texts_embeds_each_text_in_order() -> None:
    """embed_texts/embed should call the API once per text and preserve order."""
    provider = _provider(_success_transport([1.0, 2.0]))

    result = await provider.embed_texts(["a", "b", "c"])

    assert result == [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]


async def test_embed_unreachable_raises_ollama_embedding_error() -> None:
    """A connection failure should raise OllamaEmbeddingError."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = _provider(httpx.MockTransport(handler))

    with pytest.raises(OllamaEmbeddingError):
        await provider.embed_text("hello")


async def test_embed_non_200_response_raises_ollama_embedding_error() -> None:
    """A non-200 response should raise OllamaEmbeddingError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "internal error"})

    provider = _provider(httpx.MockTransport(handler))

    with pytest.raises(OllamaEmbeddingError):
        await provider.embed_text("hello")


async def test_embed_malformed_response_raises_ollama_embedding_error() -> None:
    """A response missing the expected `embedding` field should raise OllamaEmbeddingError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    provider = _provider(httpx.MockTransport(handler))

    with pytest.raises(OllamaEmbeddingError):
        await provider.embed_text("hello")


async def test_embed_empty_text_raises_value_error() -> None:
    """Empty or whitespace-only text should be rejected before any HTTP call is made."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not make an HTTP request for empty text")

    provider = _provider(httpx.MockTransport(handler))

    with pytest.raises(ValueError):
        await provider.embed_text("")

    with pytest.raises(ValueError):
        await provider.embed_text("   ")


async def test_embed_uses_ollama_embedding_model_unaffected_by_llm_model() -> None:
    """Embedding requests must always use OLLAMA_EMBEDDING_MODEL, never LLM_MODEL."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"embedding": [0.1]})

    settings = Settings(LLM_MODEL="some-other-chat-model")
    provider = OllamaEmbeddingProvider(settings=settings, transport=httpx.MockTransport(handler))

    await provider.embed_text("hello")

    assert captured["body"]["model"] == settings.ollama_embedding_model
    assert captured["body"]["model"] != "some-other-chat-model"
