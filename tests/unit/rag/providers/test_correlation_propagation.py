"""Tests that outbound Ollama/Qdrant requests carry the current correlation ID (Phase 2.10).

MinIO is deliberately not covered here — the `minio` SDK does not support per-request custom
headers cleanly (see app/storage/minio_storage.py's module docstring), so correlation is not
propagated there; this is a documented limitation, not an oversight.
"""

import httpx

from app.core.config import get_settings
from app.core.correlation import CORRELATION_ID_HEADER, set_correlation_id
from app.rag.providers.ollama_embedding_provider import OllamaEmbeddingProvider
from app.rag.providers.ollama_llm_provider import OllamaLLMProvider
from app.rag.providers.qdrant_vector_store import QdrantVectorStore


async def test_ollama_embedding_request_carries_the_correlation_id() -> None:
    set_correlation_id("corr-embedding-test")
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["header"] = request.headers.get(CORRELATION_ID_HEADER, "")
        return httpx.Response(200, json={"embedding": [0.1]})

    provider = OllamaEmbeddingProvider(settings=get_settings(), transport=httpx.MockTransport(handler))
    await provider.embed_text("hello")

    assert captured["header"] == "corr-embedding-test"


async def test_ollama_llm_request_carries_the_correlation_id() -> None:
    set_correlation_id("corr-llm-test")
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["header"] = request.headers.get(CORRELATION_ID_HEADER, "")
        return httpx.Response(200, text='{"response": "hi", "done": true}\n')

    provider = OllamaLLMProvider(settings=get_settings(), transport=httpx.MockTransport(handler))
    async for _ in provider.stream_generate("hello"):
        pass

    assert captured["header"] == "corr-llm-test"


async def test_qdrant_request_carries_the_correlation_id() -> None:
    set_correlation_id("corr-qdrant-test")
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["header"] = request.headers.get(CORRELATION_ID_HEADER, "")
        return httpx.Response(200, json={"status": "ok"})

    store = QdrantVectorStore(settings=get_settings(), transport=httpx.MockTransport(handler))
    await store.create_collection_if_not_exists("docs", vector_size=4)

    assert captured["header"] == "corr-qdrant-test"
