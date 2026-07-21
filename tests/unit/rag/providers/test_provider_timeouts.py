"""Tests that provider timeouts come from Settings, not hardcoded literals (Phase 2.10)."""

import httpx

from app.core.config import Settings
from app.rag.providers.ollama_embedding_provider import OllamaEmbeddingProvider
from app.rag.providers.ollama_llm_provider import OllamaLLMProvider
from app.rag.providers.qdrant_vector_store import QdrantVectorStore
from app.services.ollama_client import OllamaClient


def _dummy_transport() -> httpx.MockTransport:
    return httpx.MockTransport(lambda request: httpx.Response(200, json={}))


async def test_ollama_embedding_provider_uses_configured_timeout() -> None:
    captured: dict[str, object] = {}
    original_init = httpx.AsyncClient.__init__

    def _capture_init(self, *args, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        return original_init(self, *args, **kwargs)

    import unittest.mock

    settings = Settings(OLLAMA_EMBEDDING_TIMEOUT_SECONDS=17.0)
    provider = OllamaEmbeddingProvider(settings=settings, transport=_dummy_transport())
    with unittest.mock.patch.object(httpx.AsyncClient, "__init__", _capture_init):
        try:
            await provider.embed_text("hello")
        except Exception:
            pass

    assert captured["timeout"] == 17.0


async def test_ollama_llm_provider_uses_configured_timeout() -> None:
    import unittest.mock

    captured: dict[str, object] = {}
    original_init = httpx.AsyncClient.__init__

    def _capture_init(self, *args, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        return original_init(self, *args, **kwargs)

    settings = Settings(OLLAMA_LLM_TIMEOUT_SECONDS=42.0)
    provider = OllamaLLMProvider(settings=settings, transport=_dummy_transport())
    with unittest.mock.patch.object(httpx.AsyncClient, "__init__", _capture_init):
        try:
            async for _ in provider.stream_generate("hello"):
                pass
        except Exception:
            pass

    assert captured["timeout"] == 42.0


async def test_ollama_health_client_uses_configured_timeout() -> None:
    import unittest.mock

    captured: dict[str, object] = {}
    original_init = httpx.AsyncClient.__init__

    def _capture_init(self, *args, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        return original_init(self, *args, **kwargs)

    settings = Settings(OLLAMA_HEALTH_TIMEOUT_SECONDS=1.5)
    client = OllamaClient(settings=settings, transport=_dummy_transport())
    with unittest.mock.patch.object(httpx.AsyncClient, "__init__", _capture_init):
        await client.check_health()

    assert captured["timeout"] == 1.5


async def test_qdrant_vector_store_uses_configured_timeout() -> None:
    settings = Settings(QDRANT_TIMEOUT_SECONDS=9.0)
    store = QdrantVectorStore(settings=settings, transport=_dummy_transport())

    client = store._client()
    try:
        assert client.timeout == httpx.Timeout(9.0)
    finally:
        await client.aclose()
