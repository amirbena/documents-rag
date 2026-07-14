"""Backend E2E: document upload content-hash deduplication, through the real HTTP boundary.

Same real-HTTP, real-Postgres/Qdrant, fake-AI-provider setup as
test_upload_to_streaming_chat.py/test_backend_failure_paths.py (see tests/e2e/backend/
conftest.py). Covers the single-request-at-a-time reuse outcomes (CREATED baseline,
REUSED_INDEXED, REUSED_FAILED, filename independence, response safety) — sequential-active and
concurrent races live in test_concurrent_uploads.py, deletion conflicts in
test_deletion_conflicts.py, and hash release in test_reupload_after_deletion.py.
"""

from collections.abc import Awaitable, Callable

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.rag.embedding_config import get_active_embedding_config
from app.rag.providers.qdrant_vector_store import QdrantVectorStore
from tests.e2e.backend.fakes import FakeEmbeddingProvider

pytestmark = pytest.mark.e2e

_NEW_UPLOAD_CONTENT = b"Topic: quarterly budget review and headcount planning.\n"
_INDEXED_REUSE_CONTENT = b"Topic: distinctivemarker vector search deduplication check.\n"
_FILENAME_INDEPENDENCE_CONTENT = b"Topic: renamed filename must not affect content identity.\n"
_INVALID_PDF_CONTENT = b"this is not a valid pdf file"
_RESPONSE_SAFETY_CONTENT = b"Topic: response safety fields check.\n"

_FORBIDDEN_RESPONSE_FIELDS = (
    "content_hash",
    "storage_key",
    "storage_bucket",
    "storage_etag",
    "storage_provider",
    "error_message",
    "safe_message",
)


async def _table_count(session_factory: async_sessionmaker[AsyncSession], table: str) -> int:
    async with session_factory() as session:
        result = await session.execute(text(f"SELECT count(*) FROM {table}"))
        return result.scalar_one()


async def test_new_upload_creates_one_document_lifecycle_with_searchable_vectors(
    app_client: httpx.AsyncClient,
    process_pending_job: Callable[[], Awaitable[IngestionJob | None]],
    fake_embedding_provider: FakeEmbeddingProvider,
    e2e_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Baseline: a genuinely new upload is 202/CREATED, one document, one job, indexed vectors."""
    response = await app_client.post(
        "/api/v1/documents",
        files={"file": ("budget.txt", _NEW_UPLOAD_CONTENT, "text/plain")},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["outcome"] == "CREATED"
    assert body["original_filename"] == "budget.txt"
    document_id = body["document_id"]
    job_id = body["job_id"]

    result = await process_pending_job()
    assert result is not None
    assert result.id == job_id
    assert result.status == IngestionStatus.COMPLETED

    assert await _table_count(e2e_session_factory, "documents") == 1
    assert await _table_count(e2e_session_factory, "ingestion_jobs") == 1

    settings = get_settings()
    active_config = get_active_embedding_config(settings)
    vector_store = QdrantVectorStore(settings=settings)
    query_vector = (await fake_embedding_provider.embed(["budget"]))[0]
    search_results = await vector_store.search_similar(active_config.collection_name, query_vector, limit=10)
    assert any(result.document_id == document_id for result in search_results)


async def test_sequential_reuse_after_indexing_returns_reused_indexed_with_no_duplicate_vectors(
    app_client: httpx.AsyncClient,
    process_pending_job: Callable[[], Awaitable[IngestionJob | None]],
    fake_embedding_provider: FakeEmbeddingProvider,
    e2e_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Uploading identical bytes after indexing reuses the document; no re-indexing occurs."""
    first = await app_client.post(
        "/api/v1/documents",
        files={"file": ("original.txt", _INDEXED_REUSE_CONTENT, "text/plain")},
    )
    assert first.status_code == 202
    document_id = first.json()["document_id"]
    job_id = first.json()["job_id"]

    result = await process_pending_job()
    assert result is not None
    assert result.status == IngestionStatus.COMPLETED

    settings = get_settings()
    active_config = get_active_embedding_config(settings)
    vector_store = QdrantVectorStore(settings=settings)
    query_vector = (await fake_embedding_provider.embed(["distinctivemarker"]))[0]
    before = await vector_store.search_similar(active_config.collection_name, query_vector, limit=50)
    chunk_ids_before = {result.chunk_id for result in before if result.document_id == document_id}
    assert chunk_ids_before

    second = await app_client.post(
        "/api/v1/documents",
        files={"file": ("duplicate.txt", _INDEXED_REUSE_CONTENT, "text/plain")},
    )

    assert second.status_code == 200
    body = second.json()
    assert body["outcome"] == "REUSED_INDEXED"
    assert body["document_id"] == document_id
    assert body["job_id"] == job_id

    # No new ingestion job was scheduled — the only proof needed that no re-indexing happened.
    assert await process_pending_job() is None

    assert await _table_count(e2e_session_factory, "documents") == 1
    assert await _table_count(e2e_session_factory, "ingestion_jobs") == 1

    after = await vector_store.search_similar(active_config.collection_name, query_vector, limit=50)
    chunk_ids_after = {result.chunk_id for result in after if result.document_id == document_id}
    assert chunk_ids_after == chunk_ids_before, "reuse must never add a second logical vector set"


async def test_reuse_preserves_original_filename_regardless_of_incoming_filename(
    app_client: httpx.AsyncClient,
    process_pending_job: Callable[[], Awaitable[IngestionJob | None]],
    e2e_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Same bytes under a different filename must return the authoritative existing filename."""
    first = await app_client.post(
        "/api/v1/documents",
        files={"file": ("original-document.pdf", _FILENAME_INDEPENDENCE_CONTENT, "text/plain")},
    )
    assert first.status_code == 202
    document_id = first.json()["document_id"]
    await process_pending_job()

    second = await app_client.post(
        "/api/v1/documents",
        files={"file": ("renamed-copy.pdf", _FILENAME_INDEPENDENCE_CONTENT, "text/plain")},
    )

    assert second.status_code == 200
    body = second.json()
    assert body["document_id"] == document_id
    assert body["original_filename"] == "original-document.pdf"

    # The existing document's own metadata was never overwritten by the second request's filename.
    detail = await app_client.get(f"/api/v1/documents/{document_id}")
    assert detail.json()["original_filename"] == "original-document.pdf"

    assert await process_pending_job() is None
    assert await _table_count(e2e_session_factory, "documents") == 1


async def test_reuse_of_failed_ingestion_returns_reused_failed_without_automatic_retry(
    app_client: httpx.AsyncClient,
    process_pending_job: Callable[[], Awaitable[IngestionJob | None]],
    e2e_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Re-uploading content whose ingestion already failed must not silently reschedule it."""
    first = await app_client.post(
        "/api/v1/documents",
        files={"file": ("broken.pdf", _INVALID_PDF_CONTENT, "application/pdf")},
    )
    assert first.status_code == 202
    document_id = first.json()["document_id"]
    job_id = first.json()["job_id"]

    result = await process_pending_job()
    assert result is not None
    assert result.status == IngestionStatus.FAILED

    second = await app_client.post(
        "/api/v1/documents",
        files={"file": ("broken-copy.pdf", _INVALID_PDF_CONTENT, "application/pdf")},
    )

    assert second.status_code == 200
    body = second.json()
    assert body["outcome"] == "REUSED_FAILED"
    assert body["document_id"] == document_id
    assert body["job_id"] == job_id
    assert body["status"] == "failed"

    # No new ingestion job — proves the duplicate upload never triggers an implicit retry.
    assert await process_pending_job() is None

    assert await _table_count(e2e_session_factory, "documents") == 1
    assert await _table_count(e2e_session_factory, "ingestion_jobs") == 1


async def test_upload_responses_never_expose_hash_or_storage_internals(
    app_client: httpx.AsyncClient, process_pending_job: Callable[[], Awaitable[IngestionJob | None]]
) -> None:
    """Neither a CREATED nor a reused response body may leak internal hash/storage/error detail."""
    first = await app_client.post(
        "/api/v1/documents",
        files={"file": ("safety.txt", _RESPONSE_SAFETY_CONTENT, "text/plain")},
    )
    await process_pending_job()

    second = await app_client.post(
        "/api/v1/documents",
        files={"file": ("safety-copy.txt", _RESPONSE_SAFETY_CONTENT, "text/plain")},
    )

    for response in (first, second):
        body = response.json()
        for forbidden_field in _FORBIDDEN_RESPONSE_FIELDS:
            assert forbidden_field not in body
