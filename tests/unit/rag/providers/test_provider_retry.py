"""Tests that Ollama/Qdrant/MinIO providers retry transient failures and surface their own
error type (unchanged) after exhaustion (Phase 2.10).

The unit-tier `asyncio.sleep` no-op patch (tests/unit/conftest.py) keeps these fast — no real
backoff delay is ever waited on.
"""

import httpx
import pytest
from minio.error import S3Error
from urllib3.exceptions import MaxRetryError

from app.core.config import Settings
from app.rag.providers.ollama_embedding_provider import OllamaEmbeddingError, OllamaEmbeddingProvider
from app.rag.providers.qdrant_vector_store import QdrantVectorStore, QdrantVectorStoreError
from app.storage.minio_storage import MinioFileStorage


def _settings(**overrides) -> Settings:
    fields = {"PROVIDER_RETRY_MAX_ATTEMPTS": 3, "PROVIDER_RETRY_BASE_DELAY_SECONDS": 0.01}
    fields.update(overrides)
    return Settings(**fields)


async def test_ollama_embedding_retries_502_then_succeeds() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] < 3:
            return httpx.Response(502)
        return httpx.Response(200, json={"embedding": [0.1]})

    provider = OllamaEmbeddingProvider(settings=_settings(), transport=httpx.MockTransport(handler))
    result = await provider.embed_text("hello")

    assert result == [0.1]
    assert calls["count"] == 3


async def test_ollama_embedding_never_retries_a_400() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(400)

    provider = OllamaEmbeddingProvider(settings=_settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(OllamaEmbeddingError):
        await provider.embed_text("hello")

    assert calls["count"] == 1


async def test_ollama_embedding_exhausts_retries_and_raises_its_own_error_type() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(503)

    provider = OllamaEmbeddingProvider(
        settings=_settings(PROVIDER_RETRY_MAX_ATTEMPTS=3), transport=httpx.MockTransport(handler)
    )
    with pytest.raises(OllamaEmbeddingError, match="503"):
        await provider.embed_text("hello")

    assert calls["count"] == 3


async def test_qdrant_retries_connection_error_then_succeeds() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] < 2:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, json={"result": True, "status": "ok"})

    store = QdrantVectorStore(settings=_settings(), transport=httpx.MockTransport(handler))
    await store.create_collection_if_not_exists("docs", vector_size=4)

    assert calls["count"] == 2


async def test_qdrant_never_retries_a_404_during_existence_check() -> None:
    """A 404 on the existence pre-check is meaningful data (collection absent), not an error —
    it must never be retried or misclassified as transient."""
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(404)

    store = QdrantVectorStore(settings=_settings(), transport=httpx.MockTransport(handler))
    result = await store.get_collection_vector_size("missing")

    assert result is None
    assert calls["count"] == 1


async def test_qdrant_exhausts_retries_and_raises_its_own_error_type() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(503)

    store = QdrantVectorStore(
        settings=_settings(PROVIDER_RETRY_MAX_ATTEMPTS=3), transport=httpx.MockTransport(handler)
    )
    with pytest.raises(QdrantVectorStoreError):
        await store.get_collection_vector_size("docs")

    assert calls["count"] == 3


async def test_minio_retries_max_retry_error_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def _fake_stat_object(bucket, key):
        calls["count"] += 1
        if calls["count"] < 2:
            raise MaxRetryError(pool=None, url="http://minio/x")
        return object()

    storage = MinioFileStorage(
        settings=Settings(
            FILE_STORAGE_PROVIDER="minio",
            MINIO_ENDPOINT="localhost:9000",
            MINIO_ACCESS_KEY="key",
            MINIO_SECRET_KEY="secret",
            MINIO_BUCKET="documents",
            PROVIDER_RETRY_MAX_ATTEMPTS=3,
            PROVIDER_RETRY_BASE_DELAY_SECONDS=0.01,
        )
    )
    monkeypatch.setattr(storage._client, "stat_object", _fake_stat_object)

    result = await storage.exists("some/key")

    assert result is True
    assert calls["count"] == 2


async def test_minio_never_retries_a_not_found_s3_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def _fake_stat_object(bucket, key):
        calls["count"] += 1
        raise S3Error(
            code="NoSuchKey",
            message="not found",
            resource="x",
            request_id="1",
            host_id="1",
            response=None,
        )

    storage = MinioFileStorage(
        settings=Settings(
            FILE_STORAGE_PROVIDER="minio",
            MINIO_ENDPOINT="localhost:9000",
            MINIO_ACCESS_KEY="key",
            MINIO_SECRET_KEY="secret",
            MINIO_BUCKET="documents",
        )
    )
    monkeypatch.setattr(storage._client, "stat_object", _fake_stat_object)

    result = await storage.exists("some/key")

    assert result is False
    assert calls["count"] == 1
