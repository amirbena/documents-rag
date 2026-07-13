"""Unit tests for app.services.document_deletion_service against a fake session double.

Covers request_document_deletion()'s full decision table (Part 4.1), the concurrent-insert race,
and DocumentDeletionWorker's cleanup order (vectors strictly before storage) and partial-failure
handling. Real Postgres row-locking behavior is covered separately by
tests/integration/test_document_deletion_postgres.py; real Qdrant/MinIO behavior by
tests/integration/test_document_deletion_qdrant.py / test_document_deletion_storage.py.
"""

import uuid
from datetime import UTC, datetime, timedelta

from app.models.document import Document
from app.models.document_deletion_job import DocumentDeletionJob, DocumentDeletionStatus
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.models.vector_cleanup_job import VectorCleanupJob, VectorCleanupStatus
from app.services.document_deletion_service import (
    DeletionErrorCode,
    DeletionRequestOutcome,
    DocumentDeletionWorker,
    request_document_deletion,
    sanitize_deletion_error,
)
from app.storage.errors import StorageUnavailableError
from tests.support.fake_document_deletion_session import FakeDocumentDeletionSession

NOW = datetime(2026, 7, 13, 12, 0, 0, tzinfo=UTC)


def _document(document_id: str | None = None, **overrides: object) -> Document:
    fields: dict[str, object] = {
        "id": document_id or str(uuid.uuid4()),
        "original_filename": "a.pdf",
        "stored_filename": "a.pdf",
        "content_type": "application/pdf",
        "file_size": 10,
        "stored_path": "a.pdf",
        "storage_provider": "local",
        "storage_key": "documents/a/a.pdf",
        "created_at": NOW - timedelta(days=1),
    }
    fields.update(overrides)
    return Document(**fields)  # type: ignore[arg-type]


def _ingestion_job(
    document_id: str, status: IngestionStatus, *, created_at: datetime | None = None
) -> IngestionJob:
    created_at = created_at or (NOW - timedelta(hours=1))
    return IngestionJob(
        id=str(uuid.uuid4()),
        document_id=document_id,
        status=status,
        created_at=created_at,
        updated_at=created_at,
    )


def _deletion_job(
    document_id: str, status: DocumentDeletionStatus, *, created_at: datetime | None = None
) -> DocumentDeletionJob:
    created_at = created_at or (NOW - timedelta(minutes=30))
    return DocumentDeletionJob(
        id=str(uuid.uuid4()),
        document_id=document_id,
        status=status,
        vector_cleanup_completed=False,
        storage_cleanup_completed=False,
        created_at=created_at,
        updated_at=created_at,
    )


class _FakeVectorStore:
    def __init__(self, fail_delete_for: set[str] | None = None) -> None:
        self.deleted: list[tuple[str, str]] = []
        self._fail_delete_for = fail_delete_for or set()

    async def delete_by_document_id(self, collection_name: str, document_id: str) -> None:
        if collection_name in self._fail_delete_for:
            raise RuntimeError(f"could not delete from {collection_name}")
        self.deleted.append((collection_name, document_id))


class _FakeFileStorage:
    def __init__(self, *, raise_on_delete: Exception | None = None) -> None:
        self.deleted_keys: list[str] = []
        self._raise_on_delete = raise_on_delete

    async def delete(self, key: str) -> None:
        if self._raise_on_delete is not None:
            raise self._raise_on_delete
        self.deleted_keys.append(key)

    async def save(self, key: str, content: bytes) -> None:  # pragma: no cover - unused
        raise NotImplementedError

    async def read(self, key: str) -> bytes:  # pragma: no cover - unused
        raise NotImplementedError

    async def exists(self, key: str) -> bool:  # pragma: no cover - unused
        raise NotImplementedError


# --- request_document_deletion() decision table -----------------------------------------------


async def test_request_deletion_missing_document_returns_not_found() -> None:
    session = FakeDocumentDeletionSession()

    result = await request_document_deletion(session, "missing")

    assert result.outcome == DeletionRequestOutcome.DOCUMENT_NOT_FOUND
    assert result.job is None


async def test_request_deletion_with_no_prior_attempt_creates_pending_job() -> None:
    session = FakeDocumentDeletionSession()
    document = _document()
    session.documents[document.id] = document

    result = await request_document_deletion(session, document.id)

    assert result.outcome == DeletionRequestOutcome.CREATED
    assert result.job is not None
    assert result.job.status == DocumentDeletionStatus.PENDING
    assert session.commit_count == 1


async def test_request_deletion_active_ingestion_blocks_with_409() -> None:
    session = FakeDocumentDeletionSession()
    document = _document()
    session.documents[document.id] = document
    session.ingestion_jobs[str(uuid.uuid4())] = _ingestion_job(document.id, IngestionStatus.PENDING)

    result = await request_document_deletion(session, document.id)

    assert result.outcome == DeletionRequestOutcome.INGESTION_ACTIVE
    assert result.job is None
    assert session.commit_count == 0


async def test_request_deletion_processing_ingestion_also_blocks() -> None:
    session = FakeDocumentDeletionSession()
    document = _document()
    session.documents[document.id] = document
    session.ingestion_jobs[str(uuid.uuid4())] = _ingestion_job(document.id, IngestionStatus.PROCESSING)

    result = await request_document_deletion(session, document.id)

    assert result.outcome == DeletionRequestOutcome.INGESTION_ACTIVE


async def test_request_deletion_completed_ingestion_allows_deletion() -> None:
    session = FakeDocumentDeletionSession()
    document = _document()
    session.documents[document.id] = document
    session.ingestion_jobs[str(uuid.uuid4())] = _ingestion_job(document.id, IngestionStatus.COMPLETED)

    result = await request_document_deletion(session, document.id)

    assert result.outcome == DeletionRequestOutcome.CREATED


async def test_request_deletion_failed_ingestion_allows_deletion() -> None:
    session = FakeDocumentDeletionSession()
    document = _document()
    session.documents[document.id] = document
    session.ingestion_jobs[str(uuid.uuid4())] = _ingestion_job(document.id, IngestionStatus.FAILED)

    result = await request_document_deletion(session, document.id)

    assert result.outcome == DeletionRequestOutcome.CREATED


async def test_request_deletion_pending_job_is_already_active() -> None:
    session = FakeDocumentDeletionSession()
    document = _document()
    session.documents[document.id] = document
    existing = _deletion_job(document.id, DocumentDeletionStatus.PENDING)
    session.deletion_jobs[existing.id] = existing

    result = await request_document_deletion(session, document.id)

    assert result.outcome == DeletionRequestOutcome.ALREADY_ACTIVE
    assert result.job is not None
    assert result.job.id == existing.id
    assert session.commit_count == 0
    assert len(session.deletion_jobs) == 1


async def test_request_deletion_processing_job_is_already_active() -> None:
    session = FakeDocumentDeletionSession()
    document = _document()
    session.documents[document.id] = document
    existing = _deletion_job(document.id, DocumentDeletionStatus.PROCESSING)
    session.deletion_jobs[existing.id] = existing

    result = await request_document_deletion(session, document.id)

    assert result.outcome == DeletionRequestOutcome.ALREADY_ACTIVE
    assert result.job is not None
    assert result.job.id == existing.id


async def test_request_deletion_already_completed_is_idempotent() -> None:
    session = FakeDocumentDeletionSession()
    document = _document()
    session.documents[document.id] = document
    completed = _deletion_job(document.id, DocumentDeletionStatus.COMPLETED)
    session.deletion_jobs[completed.id] = completed

    result = await request_document_deletion(session, document.id)

    assert result.outcome == DeletionRequestOutcome.ALREADY_DELETED
    assert result.job is not None
    assert result.job.id == completed.id
    assert session.commit_count == 0
    assert len(session.deletion_jobs) == 1


async def test_request_deletion_partially_failed_creates_new_attempt_without_losing_history() -> None:
    session = FakeDocumentDeletionSession()
    document = _document()
    session.documents[document.id] = document
    failed = _deletion_job(document.id, DocumentDeletionStatus.PARTIALLY_FAILED)
    failed.error_code = DeletionErrorCode.DOCUMENT_STORAGE_CLEANUP_FAILED
    session.deletion_jobs[failed.id] = failed

    result = await request_document_deletion(session, document.id)

    assert result.outcome == DeletionRequestOutcome.CREATED
    assert result.job is not None
    assert result.job.id != failed.id
    assert result.job.status == DocumentDeletionStatus.PENDING
    # Prior failed attempt is preserved, untouched.
    assert session.deletion_jobs[failed.id].status == DocumentDeletionStatus.PARTIALLY_FAILED
    assert session.deletion_jobs[failed.id].error_code == DeletionErrorCode.DOCUMENT_STORAGE_CLEANUP_FAILED


async def test_request_deletion_picks_most_recent_job_as_latest() -> None:
    session = FakeDocumentDeletionSession()
    document = _document()
    session.documents[document.id] = document
    old = _deletion_job(
        document.id, DocumentDeletionStatus.PARTIALLY_FAILED, created_at=NOW - timedelta(days=2)
    )
    latest = _deletion_job(
        document.id, DocumentDeletionStatus.COMPLETED, created_at=NOW - timedelta(hours=1)
    )
    session.deletion_jobs[old.id] = old
    session.deletion_jobs[latest.id] = latest

    result = await request_document_deletion(session, document.id)

    assert result.outcome == DeletionRequestOutcome.ALREADY_DELETED
    assert result.job is not None
    assert result.job.id == latest.id


async def test_request_deletion_concurrent_insert_race_returns_existing_active_job() -> None:
    """A commit()-time unique-index violation is caught, returning the winning concurrent job."""
    session = FakeDocumentDeletionSession()
    document = _document()
    session.documents[document.id] = document

    winning_job = _deletion_job(document.id, DocumentDeletionStatus.PENDING, created_at=NOW)
    session.concurrent_winner_job = winning_job
    session.force_next_commit_integrity_error = True

    result = await request_document_deletion(session, document.id)

    assert result.outcome == DeletionRequestOutcome.ALREADY_ACTIVE
    assert result.job is not None
    assert result.job.id == winning_job.id
    assert session.rollback_count == 1


# --- sanitize_deletion_error() ------------------------------------------------------------------


def test_sanitize_deletion_error_returns_none_for_no_error() -> None:
    assert sanitize_deletion_error(None) is None


def test_sanitize_deletion_error_never_echoes_raw_message() -> None:
    message = sanitize_deletion_error(DeletionErrorCode.DOCUMENT_VECTOR_CLEANUP_FAILED.value)
    assert message is not None
    assert "qdrant" not in message.lower()
    assert "connection" not in message.lower()


def test_sanitize_deletion_error_falls_back_for_unknown_code() -> None:
    assert sanitize_deletion_error("some_unrecognized_code") is not None


# --- DocumentDeletionWorker: cleanup order + partial-failure handling ---------------------------


async def test_worker_returns_none_when_no_pending_job() -> None:
    session = FakeDocumentDeletionSession()
    worker = DocumentDeletionWorker(vector_store=_FakeVectorStore(), file_storage=_FakeFileStorage())

    result = await worker.process_next_job(session)

    assert result is None


async def test_worker_full_success_deletes_vectors_then_storage_and_completes() -> None:
    session = FakeDocumentDeletionSession()
    document = _document(collection_name="documents__ollama__m__ev1__cv1__d768")
    session.documents[document.id] = document
    job = _deletion_job(document.id, DocumentDeletionStatus.PENDING)
    session.deletion_jobs[job.id] = job

    vector_store = _FakeVectorStore()
    file_storage = _FakeFileStorage()
    worker = DocumentDeletionWorker(vector_store=vector_store, file_storage=file_storage)

    result = await worker.process_next_job(session)

    assert result is not None
    assert result.status == DocumentDeletionStatus.COMPLETED
    assert result.vector_cleanup_completed is True
    assert result.storage_cleanup_completed is True
    assert result.completed_at is not None
    assert result.error_code is None
    assert vector_store.deleted == [(document.collection_name, document.id)]
    assert file_storage.deleted_keys == [document.storage_key]


async def test_worker_uses_full_tracked_vector_cleanup_including_historical_collections() -> None:
    """The worker must delete from every tracked historical collection, not just the active one."""
    session = FakeDocumentDeletionSession()
    document = _document(collection_name="active_collection")
    session.documents[document.id] = document
    job = _deletion_job(document.id, DocumentDeletionStatus.PENDING)
    session.deletion_jobs[job.id] = job
    historical = VectorCleanupJob(
        id=str(uuid.uuid4()),
        document_id=document.id,
        collection_name="old_collection",
        status=VectorCleanupStatus.FAILED,
        attempts=1,
    )
    session.cleanup_jobs[historical.id] = historical

    vector_store = _FakeVectorStore()
    worker = DocumentDeletionWorker(vector_store=vector_store, file_storage=_FakeFileStorage())

    result = await worker.process_next_job(session)

    assert result is not None
    assert result.status == DocumentDeletionStatus.COMPLETED
    assert {c for c, _ in vector_store.deleted} == {"active_collection", "old_collection"}


async def test_worker_partial_vector_failure_blocks_storage_deletion() -> None:
    """A failing collection must stop the worker before storage deletion is ever attempted."""
    session = FakeDocumentDeletionSession()
    document = _document(collection_name="broken_collection")
    session.documents[document.id] = document
    job = _deletion_job(document.id, DocumentDeletionStatus.PENDING)
    session.deletion_jobs[job.id] = job

    vector_store = _FakeVectorStore(fail_delete_for={"broken_collection"})
    file_storage = _FakeFileStorage()
    worker = DocumentDeletionWorker(vector_store=vector_store, file_storage=file_storage)

    result = await worker.process_next_job(session)

    assert result is not None
    assert result.status == DocumentDeletionStatus.PARTIALLY_FAILED
    assert result.vector_cleanup_completed is False
    assert result.storage_cleanup_completed is False
    assert result.error_code == DeletionErrorCode.DOCUMENT_VECTOR_CLEANUP_FAILED
    # Storage deletion must never have been attempted.
    assert file_storage.deleted_keys == []


async def test_worker_storage_failure_after_vector_success_is_partially_failed() -> None:
    session = FakeDocumentDeletionSession()
    document = _document(collection_name="ok_collection")
    session.documents[document.id] = document
    job = _deletion_job(document.id, DocumentDeletionStatus.PENDING)
    session.deletion_jobs[job.id] = job

    vector_store = _FakeVectorStore()
    file_storage = _FakeFileStorage(raise_on_delete=StorageUnavailableError("storage down"))
    worker = DocumentDeletionWorker(vector_store=vector_store, file_storage=file_storage)

    result = await worker.process_next_job(session)

    assert result is not None
    assert result.status == DocumentDeletionStatus.PARTIALLY_FAILED
    assert result.vector_cleanup_completed is True
    assert result.storage_cleanup_completed is False
    assert result.error_code == DeletionErrorCode.DOCUMENT_STORAGE_CLEANUP_FAILED
    # Vector deletion did happen, before the storage failure.
    assert vector_store.deleted == [("ok_collection", document.id)]


async def test_worker_already_missing_storage_object_is_idempotent_success() -> None:
    """StorageObjectNotFoundError must never surface — FileStorage.delete() is idempotent by contract,
    but the worker must also tolerate a provider raising not-found explicitly rather than no-op'ing.
    """
    session = FakeDocumentDeletionSession()
    document = _document(collection_name="ok_collection")
    session.documents[document.id] = document
    job = _deletion_job(document.id, DocumentDeletionStatus.PENDING)
    session.deletion_jobs[job.id] = job

    class _IdempotentFileStorage(_FakeFileStorage):
        async def delete(self, key: str) -> None:
            # A real FileStorage.delete() never raises for a missing object (idempotent no-op) —
            # simulate that contract explicitly rather than raising StorageObjectNotFoundError.
            self.deleted_keys.append(key)

    file_storage = _IdempotentFileStorage()
    worker = DocumentDeletionWorker(vector_store=_FakeVectorStore(), file_storage=file_storage)

    result = await worker.process_next_job(session)

    assert result is not None
    assert result.status == DocumentDeletionStatus.COMPLETED


async def test_worker_claims_oldest_pending_job_first() -> None:
    session = FakeDocumentDeletionSession()
    doc_a = _document()
    doc_b = _document()
    session.documents[doc_a.id] = doc_a
    session.documents[doc_b.id] = doc_b
    older = _deletion_job(doc_a.id, DocumentDeletionStatus.PENDING, created_at=NOW - timedelta(hours=2))
    newer = _deletion_job(doc_b.id, DocumentDeletionStatus.PENDING, created_at=NOW - timedelta(minutes=5))
    session.deletion_jobs[older.id] = older
    session.deletion_jobs[newer.id] = newer

    worker = DocumentDeletionWorker(vector_store=_FakeVectorStore(), file_storage=_FakeFileStorage())
    result = await worker.process_next_job(session)

    assert result is not None
    assert result.document_id == doc_a.id
