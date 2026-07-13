"""Backend E2E: full document deletion through the real public HTTP boundary (Phase 2.8.4).

Same real-HTTP, real-Postgres/Qdrant, fake-AI-provider setup as
test_upload_to_streaming_chat.py/test_ingestion_retry_recovery.py (see conftest.py). Drives
DELETE /api/v1/documents/{id} and GET .../deletion over real HTTP, and executes the actual
cross-system cleanup via a real DocumentDeletionWorker (real Qdrant, real tmp_path-rooted
LocalFileStorage) — never a mock. Covers Part 10.5's five mandatory scenarios: successful
deletion, vector-cleanup failure, storage-cleanup failure + retry-to-success, concurrent delete
requests, and deleted-document ingestion-retry rejection.
"""

import asyncio
from pathlib import Path

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.models.ingestion_job import IngestionStatus
from app.rag.embedding_config import get_active_embedding_config
from app.rag.providers.qdrant_vector_store import QdrantVectorStore
from app.services.document_deletion_service import DocumentDeletionWorker
from app.storage.errors import StorageUnavailableError
from app.storage.local_storage import LocalFileStorage
from tests.e2e.backend.fakes import FakeEmbeddingProvider

pytestmark = pytest.mark.e2e

_VALID_CONTENT = b"Plain text content for the document deletion E2E test.\n"


class _FailingVectorStore:
    """Wraps a real QdrantVectorStore but fails delete_by_document_id() for one target collection."""

    def __init__(self, delegate: QdrantVectorStore, fail_for_collection: str) -> None:
        self._delegate = delegate
        self._fail_for_collection = fail_for_collection

    async def delete_by_document_id(self, collection_name: str, document_id: str) -> None:
        if collection_name == self._fail_for_collection:
            raise RuntimeError("simulated Qdrant delete failure")
        await self._delegate.delete_by_document_id(collection_name, document_id)

    def __getattr__(self, name: str):
        return getattr(self._delegate, name)


class _FailOnceThenSucceedFileStorage:
    """Wraps a real LocalFileStorage but raises on the first delete() call, succeeds afterward."""

    def __init__(self, delegate: LocalFileStorage) -> None:
        self._delegate = delegate
        self._delete_calls = 0

    async def delete(self, key: str) -> None:
        self._delete_calls += 1
        if self._delete_calls == 1:
            raise StorageUnavailableError("simulated transient storage outage")
        await self._delegate.delete(key)

    async def save(self, key: str, content: bytes):
        return await self._delegate.save(key, content)

    async def read(self, key: str) -> bytes:
        return await self._delegate.read(key)

    async def exists(self, key: str) -> bool:
        return await self._delegate.exists(key)


@pytest.fixture
def process_pending_deletion(
    e2e_session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
    isolated_test_state: None,
):
    """Run a real DocumentDeletionWorker against one pending deletion job (real Qdrant + storage)."""
    settings = get_settings()
    file_storage = LocalFileStorage(root=tmp_path)
    vector_store = QdrantVectorStore(settings=settings)

    async def _process():
        async with e2e_session_factory() as session:
            worker = DocumentDeletionWorker(vector_store=vector_store, file_storage=file_storage)
            return await worker.process_next_job(session)

    return _process


@pytest.fixture
def process_pending_deletion_with_failing_vector_store(
    e2e_session_factory: async_sessionmaker[AsyncSession], tmp_path: Path, isolated_test_state: None
):
    """Like process_pending_deletion, but vector cleanup fails for the active collection."""
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


@pytest.fixture
def process_pending_deletion_with_flaky_storage(
    e2e_session_factory: async_sessionmaker[AsyncSession], tmp_path: Path, isolated_test_state: None
):
    """Like process_pending_deletion, but the first storage delete() call fails, then succeeds."""
    settings = get_settings()
    real_file_storage = LocalFileStorage(root=tmp_path)
    flaky_file_storage = _FailOnceThenSucceedFileStorage(real_file_storage)
    vector_store = QdrantVectorStore(settings=settings)

    async def _process():
        async with e2e_session_factory() as session:
            worker = DocumentDeletionWorker(vector_store=vector_store, file_storage=flaky_file_storage)
            return await worker.process_next_job(session)

    return _process


async def _upload_and_ingest(app_client: httpx.AsyncClient, process_pending_job) -> str:
    """Upload one document and drive it to COMPLETED through the real IngestionWorker."""
    upload = await app_client.post(
        "/api/v1/documents", files={"file": ("notes.txt", _VALID_CONTENT, "text/plain")}
    )
    assert upload.status_code == 202
    document_id = upload.json()["document_id"]

    result = await process_pending_job()
    assert result is not None
    assert result.status == IngestionStatus.COMPLETED
    return document_id


# --- Scenario 1: successful deletion -------------------------------------------------------------


async def test_successful_deletion_removes_content_and_makes_it_unsearchable(
    app_client: httpx.AsyncClient,
    process_pending_job,
    process_pending_deletion,
    fake_embedding_provider: FakeEmbeddingProvider,
) -> None:
    """Upload -> ingest -> searchable -> DELETE -> worker runs -> deleted, 410, not searchable."""
    document_id = await _upload_and_ingest(app_client, process_pending_job)

    settings = get_settings()
    active_config = get_active_embedding_config(settings)
    vector_store = QdrantVectorStore(settings=settings)
    query_vector = (await fake_embedding_provider.embed(["notes"]))[0]

    before = await vector_store.search_similar(active_config.collection_name, query_vector, limit=10)
    assert any(result.document_id == document_id for result in before)

    delete_response = await app_client.delete(f"/api/v1/documents/{document_id}")
    assert delete_response.status_code == 202
    assert delete_response.json()["created"] is True

    deletion_result = await process_pending_deletion()
    assert deletion_result is not None
    assert deletion_result.status.value == "completed"

    detail = await app_client.get(f"/api/v1/documents/{document_id}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "deleted"

    download = await app_client.get(f"/api/v1/documents/{document_id}/download")
    assert download.status_code == 410

    after = await vector_store.search_similar(active_config.collection_name, query_vector, limit=10)
    assert all(result.document_id != document_id for result in after)

    status_response = await app_client.get(f"/api/v1/documents/{document_id}/deletion")
    assert status_response.status_code == 200
    body = status_response.json()
    assert body["status"] == "completed"
    assert body["vector_cleanup_completed"] is True
    assert body["storage_cleanup_completed"] is True


# --- Scenario 2: vector cleanup failure ------------------------------------------------------------


async def test_vector_cleanup_failure_leaves_document_available_not_deleted(
    app_client: httpx.AsyncClient,
    process_pending_job,
    process_pending_deletion_with_failing_vector_store,
) -> None:
    """A partial vector-cleanup failure -> deletion_failed lifecycle; object stays available."""
    document_id = await _upload_and_ingest(app_client, process_pending_job)

    delete_response = await app_client.delete(f"/api/v1/documents/{document_id}")
    assert delete_response.status_code == 202

    deletion_result = await process_pending_deletion_with_failing_vector_store()
    assert deletion_result is not None
    assert deletion_result.status.value == "partially_failed"
    assert deletion_result.vector_cleanup_completed is False
    assert deletion_result.storage_cleanup_completed is False

    detail = await app_client.get(f"/api/v1/documents/{document_id}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "deletion_failed"

    # The object was never touched — vector cleanup failed before storage cleanup was attempted.
    download = await app_client.get(f"/api/v1/documents/{document_id}/download")
    assert download.status_code == 200

    status_response = await app_client.get(f"/api/v1/documents/{document_id}/deletion")
    assert status_response.status_code == 200
    body = status_response.json()
    assert body["status"] == "partially_failed"
    assert body["safe_message"] is not None
    # No raw provider exception text ever appears in the public response.
    assert "RuntimeError" not in body["safe_message"]
    assert "simulated" not in body["safe_message"]


# --- Scenario 3: storage cleanup failure, then successful retry ------------------------------------


async def test_storage_failure_then_retry_succeeds(
    app_client: httpx.AsyncClient,
    process_pending_job,
    process_pending_deletion_with_flaky_storage,
    process_pending_deletion,
) -> None:
    """Vectors removed, storage fails -> deletion_failed; retry -> storage succeeds -> deleted."""
    document_id = await _upload_and_ingest(app_client, process_pending_job)

    delete_response = await app_client.delete(f"/api/v1/documents/{document_id}")
    assert delete_response.status_code == 202

    first_attempt = await process_pending_deletion_with_flaky_storage()
    assert first_attempt is not None
    assert first_attempt.status.value == "partially_failed"
    assert first_attempt.vector_cleanup_completed is True
    assert first_attempt.storage_cleanup_completed is False

    detail_after_failure = await app_client.get(f"/api/v1/documents/{document_id}")
    assert detail_after_failure.json()["status"] == "deletion_failed"

    retry_response = await app_client.delete(f"/api/v1/documents/{document_id}")
    assert retry_response.status_code == 202
    assert retry_response.json()["created"] is True
    assert retry_response.json()["deletion_job_id"] != first_attempt.id

    second_attempt = await process_pending_deletion()
    assert second_attempt is not None
    assert second_attempt.status.value == "completed"

    detail_after_success = await app_client.get(f"/api/v1/documents/{document_id}")
    assert detail_after_success.json()["status"] == "deleted"

    download = await app_client.get(f"/api/v1/documents/{document_id}/download")
    assert download.status_code == 410


# --- Scenario 4: concurrent delete requests ---------------------------------------------------------


async def test_concurrent_delete_requests_reference_the_same_deletion_job(
    app_client: httpx.AsyncClient, process_pending_job
) -> None:
    """Two DELETE calls for the same document must reference exactly one active deletion job."""
    document_id = await _upload_and_ingest(app_client, process_pending_job)

    first, second = await asyncio.gather(
        app_client.delete(f"/api/v1/documents/{document_id}"),
        app_client.delete(f"/api/v1/documents/{document_id}"),
    )

    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["deletion_job_id"] == second.json()["deletion_job_id"]
    # Exactly one of the two responses reports having created the job.
    assert sorted([first.json()["created"], second.json()["created"]]) == [False, True]


# --- Scenario 5: deleted document rejects ingestion retry -------------------------------------------


async def test_deleted_document_rejects_ingestion_retry(
    app_client: httpx.AsyncClient, process_pending_job, process_pending_deletion
) -> None:
    """A fully deleted document must reject POST .../ingestion/retry with 409, never resurrect it."""
    document_id = await _upload_and_ingest(app_client, process_pending_job)

    delete_response = await app_client.delete(f"/api/v1/documents/{document_id}")
    assert delete_response.status_code == 202

    deletion_result = await process_pending_deletion()
    assert deletion_result is not None
    assert deletion_result.status.value == "completed"

    retry_response = await app_client.post(f"/api/v1/documents/{document_id}/ingestion/retry")
    assert retry_response.status_code == 409
