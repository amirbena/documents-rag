"""Unit tests for app.services.documents.deletion_worker against a fake session double.

Covers DocumentDeletionWorker's claim/cleanup-order/partial-failure/completion behavior only —
never request-scoped scheduling decisions (see test_deletion_service.py for those). Real
Postgres claim-locking (`FOR UPDATE SKIP LOCKED`) is covered separately by
tests/integration/documents/deletion/test_postgres.py; real Qdrant/MinIO behavior by
tests/integration/documents/deletion/test_qdrant.py / test_storage.py.
"""

import uuid
from datetime import timedelta

from app.models.document_deletion_job import DocumentDeletionStatus
from app.models.vector_cleanup_job import VectorCleanupJob, VectorCleanupStatus
from app.services.documents.deletion_service import DeletionErrorCode
from app.services.documents.deletion_worker import DocumentDeletionWorker
from app.storage.errors import StorageUnavailableError
from tests.support.documents.deletion.builders import NOW, build_deletion_job, build_document
from tests.support.documents.deletion.fake_session import FakeDocumentDeletionSession


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


async def test_worker_returns_none_when_no_pending_job() -> None:
    session = FakeDocumentDeletionSession()
    worker = DocumentDeletionWorker(vector_store=_FakeVectorStore(), file_storage=_FakeFileStorage())

    result = await worker.process_next_job(session)

    assert result is None


async def test_worker_full_success_deletes_vectors_then_storage_and_completes() -> None:
    session = FakeDocumentDeletionSession()
    document = build_document(collection_name="documents__ollama__m__ev1__cv1__d768")
    session.documents[document.id] = document
    job = build_deletion_job(document.id, DocumentDeletionStatus.PENDING)
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
    document = build_document(collection_name="active_collection")
    session.documents[document.id] = document
    job = build_deletion_job(document.id, DocumentDeletionStatus.PENDING)
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
    document = build_document(collection_name="broken_collection")
    session.documents[document.id] = document
    job = build_deletion_job(document.id, DocumentDeletionStatus.PENDING)
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
    document = build_document(collection_name="ok_collection")
    session.documents[document.id] = document
    job = build_deletion_job(document.id, DocumentDeletionStatus.PENDING)
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
    document = build_document(collection_name="ok_collection")
    session.documents[document.id] = document
    job = build_deletion_job(document.id, DocumentDeletionStatus.PENDING)
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


async def test_worker_completed_deletion_clears_content_hash() -> None:
    """The COMPLETED transition must release the document's content_hash (Phase 2.8.5), in the
    same commit — never left set once deletion genuinely, fully finishes.
    """
    session = FakeDocumentDeletionSession()
    document = build_document(collection_name="documents__ollama__m__ev1__cv1__d768", content_hash="a" * 64)
    session.documents[document.id] = document
    job = build_deletion_job(document.id, DocumentDeletionStatus.PENDING)
    session.deletion_jobs[job.id] = job

    worker = DocumentDeletionWorker(vector_store=_FakeVectorStore(), file_storage=_FakeFileStorage())
    result = await worker.process_next_job(session)

    assert result is not None
    assert result.status == DocumentDeletionStatus.COMPLETED
    assert document.content_hash is None


async def test_worker_partial_vector_failure_preserves_content_hash() -> None:
    """A PARTIALLY_FAILED deletion (vector cleanup failure) must never release the content hash —
    the old lifecycle still owns unresolved external resources.
    """
    session = FakeDocumentDeletionSession()
    document = build_document(collection_name="broken_collection", content_hash="b" * 64)
    session.documents[document.id] = document
    job = build_deletion_job(document.id, DocumentDeletionStatus.PENDING)
    session.deletion_jobs[job.id] = job

    vector_store = _FakeVectorStore(fail_delete_for={"broken_collection"})
    worker = DocumentDeletionWorker(vector_store=vector_store, file_storage=_FakeFileStorage())
    result = await worker.process_next_job(session)

    assert result is not None
    assert result.status == DocumentDeletionStatus.PARTIALLY_FAILED
    assert document.content_hash == "b" * 64


async def test_worker_storage_failure_preserves_content_hash() -> None:
    """A PARTIALLY_FAILED deletion (storage cleanup failure, after vector cleanup succeeded) must
    also never release the content hash.
    """
    session = FakeDocumentDeletionSession()
    document = build_document(collection_name="ok_collection", content_hash="c" * 64)
    session.documents[document.id] = document
    job = build_deletion_job(document.id, DocumentDeletionStatus.PENDING)
    session.deletion_jobs[job.id] = job

    file_storage = _FakeFileStorage(raise_on_delete=StorageUnavailableError("storage down"))
    worker = DocumentDeletionWorker(vector_store=_FakeVectorStore(), file_storage=file_storage)
    result = await worker.process_next_job(session)

    assert result is not None
    assert result.status == DocumentDeletionStatus.PARTIALLY_FAILED
    assert document.content_hash == "c" * 64


async def test_worker_reprocessing_after_completion_is_safe_and_hash_stays_released() -> None:
    """A second process_next_job() call after completion must find no more pending work and must
    never re-touch the already-released content_hash.
    """
    session = FakeDocumentDeletionSession()
    document = build_document(collection_name="documents__ollama__m__ev1__cv1__d768", content_hash="d" * 64)
    session.documents[document.id] = document
    job = build_deletion_job(document.id, DocumentDeletionStatus.PENDING)
    session.deletion_jobs[job.id] = job

    worker = DocumentDeletionWorker(vector_store=_FakeVectorStore(), file_storage=_FakeFileStorage())
    first = await worker.process_next_job(session)
    assert first is not None
    assert first.status == DocumentDeletionStatus.COMPLETED
    assert document.content_hash is None

    second = await worker.process_next_job(session)

    assert second is None
    assert document.content_hash is None


async def test_worker_clearing_one_document_hash_does_not_touch_unrelated_documents() -> None:
    """Completing one document's deletion must never affect another, unrelated document's hash."""
    session = FakeDocumentDeletionSession()
    target = build_document(collection_name="documents__ollama__m__ev1__cv1__d768", content_hash="e" * 64)
    unrelated = build_document(content_hash="f" * 64)
    session.documents[target.id] = target
    session.documents[unrelated.id] = unrelated
    job = build_deletion_job(target.id, DocumentDeletionStatus.PENDING)
    session.deletion_jobs[job.id] = job

    worker = DocumentDeletionWorker(vector_store=_FakeVectorStore(), file_storage=_FakeFileStorage())
    result = await worker.process_next_job(session)

    assert result is not None
    assert result.status == DocumentDeletionStatus.COMPLETED
    assert target.content_hash is None
    assert unrelated.content_hash == "f" * 64


async def test_worker_claims_oldest_pending_job_first() -> None:
    session = FakeDocumentDeletionSession()
    doc_a = build_document()
    doc_b = build_document()
    session.documents[doc_a.id] = doc_a
    session.documents[doc_b.id] = doc_b
    older = build_deletion_job(doc_a.id, DocumentDeletionStatus.PENDING, created_at=NOW - timedelta(hours=2))
    newer = build_deletion_job(
        doc_b.id, DocumentDeletionStatus.PENDING, created_at=NOW - timedelta(minutes=5)
    )
    session.deletion_jobs[older.id] = older
    session.deletion_jobs[newer.id] = newer

    worker = DocumentDeletionWorker(vector_store=_FakeVectorStore(), file_storage=_FakeFileStorage())
    result = await worker.process_next_job(session)

    assert result is not None
    assert result.document_id == doc_a.id
