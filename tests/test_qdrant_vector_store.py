"""Tests for QdrantVectorStore with a mocked Qdrant HTTP transport."""

import json

import httpx
import pytest

from app.core.config import get_settings
from app.rag.providers.qdrant_vector_store import QdrantVectorStore, QdrantVectorStoreError
from app.rag.providers.vector_store import VectorPoint


def _store(transport: httpx.MockTransport) -> QdrantVectorStore:
    return QdrantVectorStore(settings=get_settings(), transport=transport)


def _sample_point(point_id: str = "point-1") -> VectorPoint:
    return VectorPoint(
        id=point_id,
        vector=[0.1, 0.2, 0.3],
        document_id="doc-1",
        chunk_id="chunk-1",
        text="hello world",
        source="handbook.pdf",
        page_number=2,
    )


async def test_create_collection_if_not_exists_creates_when_missing() -> None:
    """A 404 on GET /collections/{name} should trigger a PUT to create the collection."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if request.method == "GET":
            return httpx.Response(404, json={"status": "not found"})
        return httpx.Response(200, json={"result": True, "status": "ok"})

    store = _store(httpx.MockTransport(handler))

    await store.create_collection_if_not_exists("docs", vector_size=768)

    assert calls == ["GET", "PUT"]


async def test_create_collection_if_not_exists_skips_when_present() -> None:
    """A 200 on GET /collections/{name} should skip creation entirely."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        return httpx.Response(200, json={"result": {"status": "green"}, "status": "ok"})

    store = _store(httpx.MockTransport(handler))

    await store.create_collection_if_not_exists("docs", vector_size=768)

    assert calls == ["GET"]


async def test_create_collection_unreachable_raises_error() -> None:
    """A connection failure should raise QdrantVectorStoreError."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    store = _store(httpx.MockTransport(handler))

    with pytest.raises(QdrantVectorStoreError):
        await store.create_collection_if_not_exists("docs", vector_size=768)


async def test_upsert_vectors_sends_expected_payload() -> None:
    """upsert_vectors should PUT points with id/vector/payload metadata fields."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"result": {"status": "acknowledged"}, "status": "ok"})

    store = _store(httpx.MockTransport(handler))

    await store.upsert_vectors("docs", [_sample_point()])

    body = captured["body"]
    point = body["points"][0]
    assert point["id"] == "point-1"
    assert point["vector"] == [0.1, 0.2, 0.3]
    assert point["payload"] == {
        "document_id": "doc-1",
        "chunk_id": "chunk-1",
        "text": "hello world",
        "source": "handbook.pdf",
        "page_number": 2,
    }


async def test_upsert_vectors_omits_page_number_when_absent() -> None:
    """A point without page_number should omit it from the payload entirely."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"result": {"status": "acknowledged"}, "status": "ok"})

    store = _store(httpx.MockTransport(handler))
    point = VectorPoint(
        id="point-2",
        vector=[0.5],
        document_id="doc-2",
        chunk_id="chunk-2",
        text="no page number here",
        source="notes.txt",
    )

    await store.upsert_vectors("docs", [point])

    payload = captured["body"]["points"][0]["payload"]
    assert "page_number" not in payload


async def test_upsert_vectors_non_200_response_raises_error() -> None:
    """A non-200 response should raise QdrantVectorStoreError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"status": {"error": "boom"}})

    store = _store(httpx.MockTransport(handler))

    with pytest.raises(QdrantVectorStoreError):
        await store.upsert_vectors("docs", [_sample_point()])


async def test_search_similar_returns_parsed_results() -> None:
    """search_similar should parse Qdrant's result list into VectorSearchResult objects."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "result": [
                    {
                        "id": "point-1",
                        "score": 0.987,
                        "payload": {
                            "document_id": "doc-1",
                            "chunk_id": "chunk-1",
                            "text": "hello world",
                            "source": "handbook.pdf",
                            "page_number": 2,
                        },
                    }
                ],
                "status": "ok",
            },
        )

    store = _store(httpx.MockTransport(handler))

    results = await store.search_similar("docs", query_vector=[0.1, 0.2, 0.3], limit=5)

    assert len(results) == 1
    result = results[0]
    assert result.id == "point-1"
    assert result.score == 0.987
    assert result.document_id == "doc-1"
    assert result.chunk_id == "chunk-1"
    assert result.text == "hello world"
    assert result.source == "handbook.pdf"
    assert result.page_number == 2


async def test_search_similar_non_200_response_raises_error() -> None:
    """A non-200 response should raise QdrantVectorStoreError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"status": {"error": "boom"}})

    store = _store(httpx.MockTransport(handler))

    with pytest.raises(QdrantVectorStoreError):
        await store.search_similar("docs", query_vector=[0.1], limit=5)


async def test_search_similar_malformed_response_raises_error() -> None:
    """A response missing the expected payload fields should raise QdrantVectorStoreError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": [{"id": "point-1", "score": 0.5, "payload": {}}]})

    store = _store(httpx.MockTransport(handler))

    with pytest.raises(QdrantVectorStoreError):
        await store.search_similar("docs", query_vector=[0.1], limit=5)


async def test_search_similar_unreachable_raises_error() -> None:
    """A connection failure should raise QdrantVectorStoreError."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    store = _store(httpx.MockTransport(handler))

    with pytest.raises(QdrantVectorStoreError):
        await store.search_similar("docs", query_vector=[0.1], limit=5)
