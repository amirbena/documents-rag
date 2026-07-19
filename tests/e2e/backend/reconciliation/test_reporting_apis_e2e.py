"""Backend E2E: the single-document audit and collection reconciliation report endpoints
(Phase 2.8.7, subtask 5), over the real public HTTP boundary, real Testcontainers Postgres +
Qdrant.

Documents/collections are seeded directly via a raw DB session plus a real QdrantVectorStore
(mirroring tests/integration/reconciliation/test_document_audit_qdrant_real.py's exact pattern)
rather than the full upload/ingestion HTTP pipeline — this file's purpose is the HTTP boundary and
representative healthy/inconsistent/missing scenarios, not re-proving the single-document
auditor's own finding matrix (already covered by test_document_audit_*.py) or the collection
report service's own classification matrix (already covered by
test_collection_reconciliation_report_service.py). Object Storage is faked (always "exists") so
seeded documents audit as healthy/consistent unless a scenario deliberately withholds vectors.
"""

import uuid
from collections.abc import AsyncIterator

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import app.api.v1.routes.reconciliation as reconciliation_route_module
from app.core.config import get_settings
from app.main import app
from app.models.document import Document
from app.models.index_collection import IndexCollection, IndexCollectionStatus
from app.rag.providers.qdrant_vector_store import QdrantVectorStore
from app.rag.providers.vector_store import VectorPoint

pytestmark = pytest.mark.e2e


class _AlwaysExistsFileStorage:
    async def exists(self, key: str) -> bool:
        return True


@pytest.fixture(autouse=True)
def _override_file_storage() -> AsyncIterator[None]:
    app.dependency_overrides[reconciliation_route_module.get_file_storage] = _AlwaysExistsFileStorage
    yield
    app.dependency_overrides.pop(reconciliation_route_module.get_file_storage, None)


async def _seed_index_collection(
    session_factory: async_sessionmaker[AsyncSession], collection_name: str, **overrides: object
) -> None:
    fields: dict[str, object] = dict(
        collection_name=collection_name,
        embedding_provider="ollama",
        embedding_model="test-model",
        embedding_dimension=4,
        embedding_version="v1",
        chunking_version="v1",
        status=IndexCollectionStatus.ACTIVE,
    )
    fields.update(overrides)
    async with session_factory() as session:
        existing = await session.get(IndexCollection, collection_name)
        if existing is None:
            session.add(IndexCollection(**fields))  # type: ignore[arg-type]
            await session.commit()


async def _seed_document(
    session_factory: async_sessionmaker[AsyncSession], collection_name: str | None, **overrides: object
) -> Document:
    fields: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        original_filename="a.txt",
        stored_filename=f"{uuid.uuid4().hex}.txt",
        content_type="text/plain",
        file_size=5,
        stored_path="documents/a/a.txt",
        storage_provider="local",
        storage_key="documents/a/a.txt",
        collection_name=collection_name,
    )
    if collection_name is not None:
        fields.update(
            embedding_provider="ollama",
            embedding_model="test-model",
            embedding_dimension=4,
            embedding_version="v1",
            chunking_version="v1",
        )
    fields.update(overrides)
    document = Document(**fields)  # type: ignore[arg-type]
    async with session_factory() as session:
        session.add(document)
        await session.commit()
    return document


async def _row_counts(session_factory: async_sessionmaker[AsyncSession]) -> dict[str, int]:
    async with session_factory() as session:
        counts = {}
        tables = (
            "documents",
            "ingestion_jobs",
            "document_deletion_jobs",
            "reindex_jobs",
            "vector_cleanup_jobs",
            "index_collections",
        )
        for table in tables:
            result = await session.execute(text(f"SELECT count(*) FROM {table}"))
            counts[table] = result.scalar_one()
        return counts


# --- Part A: single-document audit ---------------------------------------------------------------


async def test_single_healthy_document_report(
    app_client: httpx.AsyncClient, e2e_session_factory: async_sessionmaker[AsyncSession]
) -> None:
    settings = get_settings()
    vector_store = QdrantVectorStore(settings=settings)
    collection_name = f"e2e-audit-healthy-{uuid.uuid4().hex}"

    await _seed_index_collection(e2e_session_factory, collection_name)
    document = await _seed_document(e2e_session_factory, collection_name)

    await vector_store.create_collection_if_not_exists(collection_name, 4)
    await vector_store.upsert_vectors(
        collection_name,
        [
            VectorPoint(
                id=str(uuid.uuid4()),
                vector=[0.1, 0.2, 0.3, 0.4],
                document_id=document.id,
                chunk_id="chunk-1",
                text="hello",
                source="a.txt",
            )
        ],
    )

    response = await app_client.get(f"/api/v1/reconciliation/documents/{document.id}/audit")

    assert response.status_code == 200
    body = response.json()
    assert body["classification"] == "consistent"
    assert body["overall_status"] == "consistent"
    assert body["issues"] == []
    assert body["database"]["document_exists"] is True
    assert body["vector_store"]["collection_name"] == collection_name
    assert body["vector_store"]["has_vectors"] is True
    assert body["vector_store"]["vector_count"] == 1


async def test_single_inconsistent_document_report(
    app_client: httpx.AsyncClient, e2e_session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """A document whose active collection exists but has zero vectors for it — a representative,
    already-supported mismatch (ACTIVE_VECTORS_MISSING)."""
    settings = get_settings()
    vector_store = QdrantVectorStore(settings=settings)
    collection_name = f"e2e-audit-inconsistent-{uuid.uuid4().hex}"

    await _seed_index_collection(e2e_session_factory, collection_name)
    document = await _seed_document(e2e_session_factory, collection_name)
    await vector_store.create_collection_if_not_exists(collection_name, 4)
    # No vectors upserted for this document — collection exists, but is empty.

    response = await app_client.get(f"/api/v1/reconciliation/documents/{document.id}/audit")

    assert response.status_code == 200
    body = response.json()
    assert body["classification"] == "inconsistent"
    assert body["overall_status"] == "inconsistent"
    codes = {issue["code"] for issue in body["issues"]}
    assert "active_vectors_missing" in codes
    assert body["vector_store"]["has_vectors"] is False


async def test_missing_document_returns_200_with_not_found_classification(
    app_client: httpx.AsyncClient,
) -> None:
    response = await app_client.get(f"/api/v1/reconciliation/documents/{uuid.uuid4()}/audit")

    assert response.status_code == 200
    body = response.json()
    assert body["classification"] == "not_found"
    assert body["database"]["document_exists"] is False


# --- Part B: collection reconciliation report -----------------------------------------------------


async def test_healthy_collection_report(
    app_client: httpx.AsyncClient, e2e_session_factory: async_sessionmaker[AsyncSession]
) -> None:
    settings = get_settings()
    vector_store = QdrantVectorStore(settings=settings)
    collection_name = f"e2e-report-healthy-{uuid.uuid4().hex}"

    await _seed_index_collection(e2e_session_factory, collection_name)
    doc_a = await _seed_document(e2e_session_factory, collection_name)
    doc_b = await _seed_document(e2e_session_factory, collection_name)

    await vector_store.create_collection_if_not_exists(collection_name, 4)
    await vector_store.upsert_vectors(
        collection_name,
        [
            VectorPoint(
                id=str(uuid.uuid4()),
                vector=[0.1, 0.2, 0.3, 0.4],
                document_id=doc_a.id,
                chunk_id="chunk-1",
                text="a",
                source="a.txt",
            ),
            VectorPoint(
                id=str(uuid.uuid4()),
                vector=[0.5, 0.6, 0.7, 0.8],
                document_id=doc_b.id,
                chunk_id="chunk-1",
                text="b",
                source="b.txt",
            ),
        ],
    )

    response = await app_client.get(f"/api/v1/reconciliation/collections/{collection_name}/report")

    assert response.status_code == 200
    body = response.json()
    assert body["exists"] is True
    assert body["classification"] == "healthy"
    assert body["document_count"] == 2
    assert body["expected_vector_count"] == 2
    assert body["actual_vector_count"] == 2
    assert body["difference"] == 0
    assert body["index_collection_status"] == "active"


async def test_collection_count_mismatch_report(
    app_client: httpx.AsyncClient, e2e_session_factory: async_sessionmaker[AsyncSession]
) -> None:
    settings = get_settings()
    vector_store = QdrantVectorStore(settings=settings)
    collection_name = f"e2e-report-mismatch-{uuid.uuid4().hex}"

    await _seed_index_collection(e2e_session_factory, collection_name)
    await _seed_document(e2e_session_factory, collection_name)
    await _seed_document(e2e_session_factory, collection_name)
    await _seed_document(e2e_session_factory, collection_name)  # 3 documents claim this collection

    await vector_store.create_collection_if_not_exists(collection_name, 4)
    # Only one vector actually exists — a deliberate deficit.
    await vector_store.upsert_vectors(
        collection_name,
        [
            VectorPoint(
                id=str(uuid.uuid4()),
                vector=[0.1, 0.2, 0.3, 0.4],
                document_id=str(uuid.uuid4()),
                chunk_id="chunk-1",
                text="a",
                source="a.txt",
            )
        ],
    )
    before_count = await vector_store.count_collection_vectors(collection_name)

    response = await app_client.get(f"/api/v1/reconciliation/collections/{collection_name}/report")

    assert response.status_code == 200
    body = response.json()
    assert body["classification"] == "inconsistent"
    assert body["expected_vector_count"] == 3
    assert body["actual_vector_count"] == 1
    assert body["difference"] == -2

    after_count = await vector_store.count_collection_vectors(collection_name)
    assert after_count == before_count  # no vectors added or deleted by the report itself


async def test_missing_collection_report(app_client: httpx.AsyncClient) -> None:
    collection_name = f"e2e-report-missing-{uuid.uuid4().hex}"

    response = await app_client.get(f"/api/v1/reconciliation/collections/{collection_name}/report")

    assert response.status_code == 200
    body = response.json()
    assert body["exists"] is False
    assert body["classification"] == "missing"
    assert body["actual_vector_count"] == 0


async def test_inactive_known_collection_report_distinguishes_active_from_health(
    app_client: httpx.AsyncClient, e2e_session_factory: async_sessionmaker[AsyncSession]
) -> None:
    settings = get_settings()
    vector_store = QdrantVectorStore(settings=settings)
    collection_name = f"e2e-report-inactive-{uuid.uuid4().hex}"

    await _seed_index_collection(
        e2e_session_factory, collection_name, status=IndexCollectionStatus.RETIRED
    )
    document = await _seed_document(e2e_session_factory, collection_name)
    await vector_store.create_collection_if_not_exists(collection_name, 4)
    await vector_store.upsert_vectors(
        collection_name,
        [
            VectorPoint(
                id=str(uuid.uuid4()),
                vector=[0.1, 0.2, 0.3, 0.4],
                document_id=document.id,
                chunk_id="chunk-1",
                text="a",
                source="a.txt",
            )
        ],
    )

    response = await app_client.get(f"/api/v1/reconciliation/collections/{collection_name}/report")

    assert response.status_code == 200
    body = response.json()
    assert body["is_active"] is False  # not the platform's current desired collection
    assert body["index_collection_status"] == "retired"
    assert body["classification"] == "healthy"  # inactive, but internally consistent


# --- read-only guarantee ---------------------------------------------------------------------------


async def test_both_endpoints_are_read_only(
    app_client: httpx.AsyncClient, e2e_session_factory: async_sessionmaker[AsyncSession]
) -> None:
    settings = get_settings()
    vector_store = QdrantVectorStore(settings=settings)
    collection_name = f"e2e-report-readonly-{uuid.uuid4().hex}"

    await _seed_index_collection(e2e_session_factory, collection_name)
    document = await _seed_document(e2e_session_factory, collection_name)
    await vector_store.create_collection_if_not_exists(collection_name, 4)
    await vector_store.upsert_vectors(
        collection_name,
        [
            VectorPoint(
                id=str(uuid.uuid4()),
                vector=[0.1, 0.2, 0.3, 0.4],
                document_id=document.id,
                chunk_id="chunk-1",
                text="a",
                source="a.txt",
            )
        ],
    )

    before_rows = await _row_counts(e2e_session_factory)
    before_vectors = await vector_store.count_collection_vectors(collection_name)

    audit_response = await app_client.get(f"/api/v1/reconciliation/documents/{document.id}/audit")
    assert audit_response.status_code == 200
    report_response = await app_client.get(f"/api/v1/reconciliation/collections/{collection_name}/report")
    assert report_response.status_code == 200

    after_rows = await _row_counts(e2e_session_factory)
    after_vectors = await vector_store.count_collection_vectors(collection_name)

    assert after_rows == before_rows
    assert after_vectors == before_vectors

    async with e2e_session_factory() as session:
        fresh_document = await session.get(Document, document.id)
    assert fresh_document is not None
    assert fresh_document.collection_name == collection_name
