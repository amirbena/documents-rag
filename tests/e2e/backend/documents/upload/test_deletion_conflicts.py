"""Backend E2E: uploading identical content while a matching document's deletion is unresolved.

Same real-HTTP, real-Postgres/Qdrant, fake-AI-provider setup as
documents/deletion/test_partial_failures.py (see tests/e2e/backend/conftest.py). Both scenarios
here must never reach `upload_document()`'s normal reuse path — Subtask 4's typed
`DeletionActiveError`/`DeletionIncompleteError` -> 409 mapping is what's under test, using the
stable public response shape, never raw internal exception text.

The PENDING/active case needs no fault injection at all: `DELETE` a document and simply never
call `process_pending_deletion()` — the job stays PENDING deterministically. The PARTIALLY_FAILED
case reuses test_partial_failures.py's `_FailingVectorStore` wrapper pattern locally (a small,
test-only double around the real `QdrantVectorStore`) rather than sharing it across feature
packages for a single second use.
"""

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.rag.embedding_config import get_active_embedding_config
from app.rag.providers.qdrant_vector_store import QdrantVectorStore
from app.services.documents.deletion_worker import DocumentDeletionWorker
from app.storage.local_storage import LocalFileStorage
from tests.e2e.backend.documents.deletion.support import upload_and_ingest

pytestmark = pytest.mark.e2e

_DUPLICATE_CONTENT = b"Plain text content for the document deletion E2E test.\n"


class _FailingVectorStore:
    """Wraps a real QdrantVectorStore but fails delete_by_document_id() for one target collection.

    Mirrors documents/deletion/test_partial_failures.py's identically-named class — kept local
    here rather than shared, since only these two modules need it.
    """

    def __init__(self, delegate: QdrantVectorStore, fail_for_collection: str) -> None:
        self._delegate = delegate
        self._fail_for_collection = fail_for_collection

    async def delete_by_document_id(self, collection_name: str, document_id: str) -> None:
        if collection_name == self._fail_for_collection:
            raise RuntimeError("simulated Qdrant delete failure")
        await self._delegate.delete_by_document_id(collection_name, document_id)

    def __getattr__(self, name: str):
        return getattr(self._delegate, name)


@pytest.fixture
def process_pending_deletion_with_failing_vector_store(
    e2e_session_factory: async_sessionmaker[AsyncSession], tmp_path, isolated_test_state: None
):
    """Drive one deletion job to PARTIALLY_FAILED via a vector-cleanup failure, deterministically."""
    settings = get_settings()
    file_storage = LocalFileStorage(root=tmp_path)
    real_vector_store = QdrantVectorStore(settings=settings)
    active_config = get_active_embedding_config(settings)
    failing_vector_store = _FailingVectorStore(real_vector_store, active_config.collection_name)

    async def _process():
        async with e2e_session_factory() as session:
            worker = DocumentDeletionWorker(vector_store=failing_vector_store, file_storage=file_storage)
            return await worker.process_next_job(session)

    return _process


async def _table_count(session_factory: async_sessionmaker[AsyncSession], table: str) -> int:
    async with session_factory() as session:
        result = await session.execute(text(f"SELECT count(*) FROM {table}"))
        return result.scalar_one()


async def test_upload_during_active_deletion_returns_sanitized_409(
    app_client: httpx.AsyncClient,
    process_pending_job,
    e2e_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Re-uploading identical bytes while a PENDING deletion is unresolved must be a 409, never a reuse."""
    document_id = await upload_and_ingest(app_client, process_pending_job)

    delete_response = await app_client.delete(f"/api/v1/documents/{document_id}")
    assert delete_response.status_code == 202
    # Deliberately never call process_pending_deletion() — the job stays PENDING deterministically.

    reupload = await app_client.post(
        "/api/v1/documents", files={"file": ("duplicate.txt", _DUPLICATE_CONTENT, "text/plain")}
    )

    assert reupload.status_code == 409
    detail = reupload.json()["detail"]
    assert "RuntimeError" not in detail
    assert "Traceback" not in detail

    # Nothing new was created, and the original deletion lifecycle is untouched.
    assert await _table_count(e2e_session_factory, "documents") == 1
    assert await _table_count(e2e_session_factory, "ingestion_jobs") == 1

    deletion_status = await app_client.get(f"/api/v1/documents/{document_id}/deletion")
    assert deletion_status.json()["status"] == "pending"


async def test_upload_during_partially_failed_deletion_returns_sanitized_409(
    app_client: httpx.AsyncClient,
    process_pending_job,
    process_pending_deletion_with_failing_vector_store,
    e2e_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Re-uploading identical bytes while a deletion is PARTIALLY_FAILED must be a 409, never a reuse."""
    document_id = await upload_and_ingest(app_client, process_pending_job)

    delete_response = await app_client.delete(f"/api/v1/documents/{document_id}")
    assert delete_response.status_code == 202

    deletion_result = await process_pending_deletion_with_failing_vector_store()
    assert deletion_result is not None
    assert deletion_result.status.value == "partially_failed"

    reupload = await app_client.post(
        "/api/v1/documents", files={"file": ("duplicate.txt", _DUPLICATE_CONTENT, "text/plain")}
    )

    assert reupload.status_code == 409
    detail = reupload.json()["detail"]
    assert "PARTIALLY_FAILED" in detail
    assert "RuntimeError" not in detail

    # The existing document remains authoritative — no new document/ingestion lifecycle was created,
    # which is the observable proof that its content hash is still reserved (never released early).
    assert await _table_count(e2e_session_factory, "documents") == 1
    assert await _table_count(e2e_session_factory, "ingestion_jobs") == 1

    detail_response = await app_client.get(f"/api/v1/documents/{document_id}")
    assert detail_response.status_code == 200
    assert detail_response.json()["status"] == "deletion_failed"
