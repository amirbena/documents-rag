"""Integration tests for QdrantVectorStore against a real, ephemeral Qdrant container.

Exercises the real Qdrant HTTP contract — collection creation, upsert, similarity ranking, full
metadata round-trip, and a real dimension-mismatch error — none of which a mocked httpx
transport can guarantee still matches Qdrant's actual behavior.
"""

import uuid

import pytest

from app.core.config import get_settings
from app.rag.providers.qdrant_vector_store import QdrantVectorStore, QdrantVectorStoreError
from app.rag.providers.vector_store import VectorPoint


def _unique_collection_name() -> str:
    return f"integration-test-{uuid.uuid4().hex}"


@pytest.fixture
def vector_store(qdrant_url: str) -> QdrantVectorStore:
    """A QdrantVectorStore pointed at the ephemeral Qdrant container."""
    return QdrantVectorStore(settings=get_settings())


async def test_create_collection_is_idempotent(vector_store: QdrantVectorStore) -> None:
    """Creating the same collection twice should not raise on the second call."""
    collection_name = _unique_collection_name()

    await vector_store.create_collection_if_not_exists(collection_name, vector_size=4)
    await vector_store.create_collection_if_not_exists(collection_name, vector_size=4)


async def test_upsert_and_search_returns_ranked_results(vector_store: QdrantVectorStore) -> None:
    """search_similar should rank real upserted points by similarity to the query vector."""
    collection_name = _unique_collection_name()
    await vector_store.create_collection_if_not_exists(collection_name, vector_size=3)

    close_point = VectorPoint(
        id=str(uuid.uuid4()),
        vector=[1.0, 0.0, 0.0],
        document_id="doc-close",
        chunk_id="chunk-close",
        text="closest chunk",
        source="close.pdf",
    )
    far_point = VectorPoint(
        id=str(uuid.uuid4()),
        vector=[0.0, 1.0, 0.0],
        document_id="doc-far",
        chunk_id="chunk-far",
        text="farthest chunk",
        source="far.pdf",
    )
    mid_point = VectorPoint(
        id=str(uuid.uuid4()),
        vector=[0.7, 0.3, 0.0],
        document_id="doc-mid",
        chunk_id="chunk-mid",
        text="middling chunk",
        source="mid.pdf",
    )
    await vector_store.upsert_vectors(collection_name, [far_point, close_point, mid_point])

    results = await vector_store.search_similar(collection_name, query_vector=[1.0, 0.0, 0.0], limit=3)

    assert len(results) == 3
    assert [result.chunk_id for result in results] == ["chunk-close", "chunk-mid", "chunk-far"]
    assert results[0].score > results[1].score > results[2].score


async def test_metadata_round_trips_through_real_qdrant(vector_store: QdrantVectorStore) -> None:
    """All payload metadata should survive a real upsert + search unchanged, plus a real score."""
    collection_name = _unique_collection_name()
    await vector_store.create_collection_if_not_exists(collection_name, vector_size=2)

    point = VectorPoint(
        id=str(uuid.uuid4()),
        vector=[0.5, 0.5],
        document_id="doc-42",
        chunk_id="doc-42-chunk-3",
        text="refunds are processed within 14 days",
        source="handbook.pdf",
        page_number=7,
        sheet_name=None,
    )
    await vector_store.upsert_vectors(collection_name, [point])

    results = await vector_store.search_similar(collection_name, query_vector=[0.5, 0.5], limit=1)

    assert len(results) == 1
    result = results[0]
    assert result.document_id == "doc-42"
    assert result.chunk_id == "doc-42-chunk-3"
    assert result.text == "refunds are processed within 14 days"
    assert result.source == "handbook.pdf"
    assert result.page_number == 7
    assert result.sheet_name is None
    assert isinstance(result.score, float)


async def test_metadata_round_trip_with_sheet_name(vector_store: QdrantVectorStore) -> None:
    """sheet_name (XLSX-sourced chunks) should round-trip too, alongside a null page_number."""
    collection_name = _unique_collection_name()
    await vector_store.create_collection_if_not_exists(collection_name, vector_size=2)

    point = VectorPoint(
        id=str(uuid.uuid4()),
        vector=[0.1, 0.9],
        document_id="doc-99",
        chunk_id="doc-99-chunk-0",
        text="Q1 revenue: 120000",
        source="report.xlsx",
        sheet_name="Summary",
    )
    await vector_store.upsert_vectors(collection_name, [point])

    results = await vector_store.search_similar(collection_name, query_vector=[0.1, 0.9], limit=1)

    assert results[0].sheet_name == "Summary"
    assert results[0].page_number is None


async def test_vector_dimension_mismatch_fails_clearly(vector_store: QdrantVectorStore) -> None:
    """Upserting a vector whose length doesn't match the collection's size should raise clearly."""
    collection_name = _unique_collection_name()
    await vector_store.create_collection_if_not_exists(collection_name, vector_size=4)

    wrong_size_point = VectorPoint(
        id=str(uuid.uuid4()),
        vector=[0.1, 0.2],
        document_id="doc-1",
        chunk_id="chunk-1",
        text="mismatched vector",
        source="notes.txt",
    )

    with pytest.raises(QdrantVectorStoreError):
        await vector_store.upsert_vectors(collection_name, [wrong_size_point])
