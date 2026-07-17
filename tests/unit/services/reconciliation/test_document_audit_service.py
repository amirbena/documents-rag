"""Unit tests for app.services.reconciliation.document_audit_service against fake doubles
(Phase 2.8.7, subtask 1).

Covers audit classification/orchestration only — narrow fakes for Document/job rows, Object
Storage, and Qdrant. Does not repeat the full unit matrices of ingestion, deletion, re-indexing, or
cleanup already covered by their own dedicated test files.
"""

import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from app.core.config import get_settings
from app.models.document import Document
from app.models.document_deletion_job import DocumentDeletionJob, DocumentDeletionStatus
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.models.reindex_job import ReindexJob, ReindexJobStatus
from app.models.vector_cleanup_job import VectorCleanupJob, VectorCleanupStatus
from app.rag.providers.qdrant_vector_store import QdrantVectorStoreError
from app.services.reconciliation.document_audit_service import (
    AuditOverallStatus,
    DocumentLifecycleFindingCode,
    FindingSeverity,
    audit_document_lifecycle,
)
from app.storage.errors import StorageUnavailableError

_BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)
_SETTINGS = get_settings()


# --- fakes --------------------------------------------------------------------------------------


class _Scalars:
    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def all(self) -> list[Any]:
        return self._items

    def first(self) -> Any | None:
        return self._items[0] if self._items else None


class _ListResult:
    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def scalars(self) -> _Scalars:
        return _Scalars(self._items)


class _FakeAuditSession:
    """In-memory AsyncSession double for audit_document_lifecycle()."""

    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.ingestion_jobs: dict[str, IngestionJob] = {}
        self.deletion_jobs: dict[str, DocumentDeletionJob] = {}
        self.reindex_jobs: dict[str, ReindexJob] = {}
        self.cleanup_jobs: dict[str, VectorCleanupJob] = {}

    def add(self, instance: object) -> None:
        if isinstance(instance, Document):
            self.documents[instance.id] = instance
        elif isinstance(instance, IngestionJob):
            self.ingestion_jobs[instance.id] = instance
        elif isinstance(instance, DocumentDeletionJob):
            self.deletion_jobs[instance.id] = instance
        elif isinstance(instance, ReindexJob):
            self.reindex_jobs[instance.id] = instance
        elif isinstance(instance, VectorCleanupJob):
            self.cleanup_jobs[instance.id] = instance

    async def get(self, model: type, instance_id: str) -> object | None:
        if model is Document:
            return self.documents.get(instance_id)
        return None

    def _latest(self, jobs: list[Any], document_id: str, table: str, compiled: str) -> _ListResult:
        matching = [job for job in jobs if job.document_id == document_id]
        matching.sort(key=lambda job: (job.created_at, job.id), reverse=True)
        limit_match = re.search(r"LIMIT (\d+)", compiled)
        if limit_match:
            matching = matching[: int(limit_match.group(1))]
        return _ListResult(matching)

    async def execute(self, stmt: Any) -> _ListResult:
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))

        if "ingestion_jobs" in compiled:
            eq_match = re.search(r"ingestion_jobs\.document_id = '([^']*)'", compiled)
            document_id = eq_match.group(1) if eq_match else ""
            return self._latest(list(self.ingestion_jobs.values()), document_id, "ingestion_jobs", compiled)

        if "document_deletion_jobs" in compiled:
            eq_match = re.search(r"document_deletion_jobs\.document_id = '([^']*)'", compiled)
            document_id = eq_match.group(1) if eq_match else ""
            return self._latest(
                list(self.deletion_jobs.values()), document_id, "document_deletion_jobs", compiled
            )

        if "reindex_jobs" in compiled:
            eq_match = re.search(r"reindex_jobs\.document_id = '([^']*)'", compiled)
            document_id = eq_match.group(1) if eq_match else ""
            return self._latest(list(self.reindex_jobs.values()), document_id, "reindex_jobs", compiled)

        if "vector_cleanup_jobs" in compiled:
            jobs = [
                job
                for job in self.cleanup_jobs.values()
                if job.status in (VectorCleanupStatus.PENDING, VectorCleanupStatus.FAILED)
            ]
            return _ListResult(jobs)

        return _ListResult([])


class _FakeFileStorage:
    def __init__(self, *, existing_keys: set[str] | None = None, unavailable: bool = False) -> None:
        self._existing_keys = existing_keys or set()
        self._unavailable = unavailable

    async def exists(self, key: str) -> bool:
        if self._unavailable:
            raise StorageUnavailableError("storage backend unreachable")
        return key in self._existing_keys


class _FakeVectorStore:
    def __init__(
        self,
        *,
        collection_sizes: dict[str, int] | None = None,
        vector_counts: dict[tuple[str, str], int] | None = None,
        unavailable_collections: set[str] | None = None,
        unavailable_counts: set[str] | None = None,
    ) -> None:
        self._collection_sizes = collection_sizes or {}
        self._vector_counts = vector_counts or {}
        self._unavailable_collections = unavailable_collections or set()
        self._unavailable_counts = unavailable_counts or set()

    async def get_collection_vector_size(self, collection_name: str) -> int | None:
        if collection_name in self._unavailable_collections:
            raise QdrantVectorStoreError("qdrant unreachable")
        return self._collection_sizes.get(collection_name)

    async def count_document_vectors(self, collection_name: str, document_id: str) -> int:
        if collection_name in self._unavailable_counts:
            raise QdrantVectorStoreError("qdrant unreachable")
        return self._vector_counts.get((collection_name, document_id), 0)


# --- builders -------------------------------------------------------------------------------


def _document(**overrides: object) -> Document:
    fields: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        original_filename="report.pdf",
        stored_filename=f"{uuid.uuid4().hex}.pdf",
        content_type="application/pdf",
        file_size=10,
        stored_path="storage/documents/report.pdf",
        storage_key=f"storage/documents/{uuid.uuid4().hex}.pdf",
        collection_name=None,
        embedding_provider=None,
        embedding_model=None,
        embedding_dimension=None,
        embedding_version=None,
        chunking_version=None,
        indexed_at=None,
    )
    fields.update(overrides)
    return Document(**fields)  # type: ignore[arg-type]


def _indexed_document(**overrides: object) -> Document:
    fields: dict[str, object] = dict(
        collection_name="documents__ollama__model__ev1__cv1__d768",
        embedding_provider="ollama",
        embedding_model="model",
        embedding_dimension=768,
        embedding_version="v1",
        chunking_version="v1",
        indexed_at=_BASE_TIME,
    )
    fields.update(overrides)
    return _document(**fields)


def _ingestion_job(document_id: str, status: IngestionStatus, **overrides: object) -> IngestionJob:
    fields: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        document_id=document_id,
        status=status,
        created_at=_BASE_TIME,
        updated_at=_BASE_TIME,
    )
    fields.update(overrides)
    return IngestionJob(**fields)  # type: ignore[arg-type]


def _deletion_job(
    document_id: str, status: DocumentDeletionStatus, **overrides: object
) -> DocumentDeletionJob:
    fields: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        document_id=document_id,
        status=status,
        vector_cleanup_completed=False,
        storage_cleanup_completed=False,
        created_at=_BASE_TIME,
        updated_at=_BASE_TIME,
    )
    fields.update(overrides)
    return DocumentDeletionJob(**fields)  # type: ignore[arg-type]


def _reindex_job(document_id: str, **overrides: object) -> ReindexJob:
    fields: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        document_id=document_id,
        source_collection_name="documents__ollama__old__ev0__cv0__d768",
        target_collection_name="documents__ollama__model__ev1__cv1__d768",
        target_chunk_size=500,
        target_chunk_overlap=50,
        status=ReindexJobStatus.COMPLETED,
        created_at=_BASE_TIME,
        completed_at=_BASE_TIME,
        activated_at=None,
    )
    fields.update(overrides)
    return ReindexJob(**fields)  # type: ignore[arg-type]


def _cleanup_job(document_id: str, **overrides: object) -> VectorCleanupJob:
    fields: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        document_id=document_id,
        collection_name="old-collection",
        status=VectorCleanupStatus.PENDING,
        attempts=0,
    )
    fields.update(overrides)
    return VectorCleanupJob(**fields)  # type: ignore[arg-type]


def _seed(session: _FakeAuditSession, document: Document, **kwargs: object) -> None:
    session.documents[document.id] = document


async def _run(
    session: _FakeAuditSession,
    document_id: str,
    *,
    file_storage: Any = None,
    vector_store: Any = None,
):
    return await audit_document_lifecycle(
        session,
        document_id,
        _SETTINGS,
        file_storage or _FakeFileStorage(),
        vector_store or _FakeVectorStore(),
    )


def _codes(result) -> set[DocumentLifecycleFindingCode]:
    return {f.code for f in result.findings}


# --- 1: missing document ------------------------------------------------------------------------


async def test_missing_document_returns_not_found() -> None:
    session = _FakeAuditSession()

    result = await _run(session, str(uuid.uuid4()))

    assert result.overall_status == AuditOverallStatus.NOT_FOUND
    assert _codes(result) == {DocumentLifecycleFindingCode.DOCUMENT_MISSING}
    assert result.postgres_state is None
    assert result.storage_state is None
    assert result.vector_state is None


# --- 2: consistent indexed document --------------------------------------------------------------


async def test_consistent_indexed_document_returns_consistent() -> None:
    session = _FakeAuditSession()
    document = _indexed_document()
    _seed(session, document)
    session.ingestion_jobs[str(uuid.uuid4())] = _ingestion_job(document.id, IngestionStatus.COMPLETED)

    file_storage = _FakeFileStorage(existing_keys={document.storage_key})
    vector_store = _FakeVectorStore(
        collection_sizes={document.collection_name: 768},
        vector_counts={(document.collection_name, document.id): 3},
    )

    result = await _run(session, document.id, file_storage=file_storage, vector_store=vector_store)

    assert result.overall_status == AuditOverallStatus.CONSISTENT
    assert result.findings == ()
    assert result.storage_state.exists is True
    assert result.vector_state.collection_exists is True
    assert result.vector_state.has_vectors is True


# --- 3/4: object storage -----------------------------------------------------------------------


async def test_missing_object_produces_object_missing() -> None:
    session = _FakeAuditSession()
    document = _indexed_document()
    _seed(session, document)
    session.ingestion_jobs[str(uuid.uuid4())] = _ingestion_job(document.id, IngestionStatus.COMPLETED)

    file_storage = _FakeFileStorage(existing_keys=set())
    vector_store = _FakeVectorStore(
        collection_sizes={document.collection_name: 768},
        vector_counts={(document.collection_name, document.id): 1},
    )

    result = await _run(session, document.id, file_storage=file_storage, vector_store=vector_store)

    assert DocumentLifecycleFindingCode.OBJECT_MISSING in _codes(result)
    assert result.overall_status == AuditOverallStatus.INCONSISTENT


async def test_storage_provider_failure_is_not_classified_as_object_missing() -> None:
    session = _FakeAuditSession()
    document = _indexed_document()
    _seed(session, document)
    session.ingestion_jobs[str(uuid.uuid4())] = _ingestion_job(document.id, IngestionStatus.COMPLETED)

    file_storage = _FakeFileStorage(unavailable=True)
    vector_store = _FakeVectorStore(
        collection_sizes={document.collection_name: 768},
        vector_counts={(document.collection_name, document.id): 1},
    )

    result = await _run(session, document.id, file_storage=file_storage, vector_store=vector_store)

    codes = _codes(result)
    assert DocumentLifecycleFindingCode.OBJECT_MISSING not in codes
    assert DocumentLifecycleFindingCode.STORAGE_INSPECTION_UNAVAILABLE in codes
    assert result.storage_state.inspected is False


# --- 5/6/7: qdrant -----------------------------------------------------------------------------


async def test_missing_active_collection_produces_active_collection_missing() -> None:
    session = _FakeAuditSession()
    document = _indexed_document()
    _seed(session, document)
    session.ingestion_jobs[str(uuid.uuid4())] = _ingestion_job(document.id, IngestionStatus.COMPLETED)

    file_storage = _FakeFileStorage(existing_keys={document.storage_key})
    vector_store = _FakeVectorStore(collection_sizes={})  # collection absent

    result = await _run(session, document.id, file_storage=file_storage, vector_store=vector_store)

    assert DocumentLifecycleFindingCode.ACTIVE_COLLECTION_MISSING in _codes(result)
    assert result.overall_status == AuditOverallStatus.INCONSISTENT
    assert result.vector_state.collection_exists is False


async def test_missing_active_vectors_produces_active_vectors_missing() -> None:
    session = _FakeAuditSession()
    document = _indexed_document()
    _seed(session, document)
    session.ingestion_jobs[str(uuid.uuid4())] = _ingestion_job(document.id, IngestionStatus.COMPLETED)

    file_storage = _FakeFileStorage(existing_keys={document.storage_key})
    vector_store = _FakeVectorStore(
        collection_sizes={document.collection_name: 768}, vector_counts={}  # zero vectors
    )

    result = await _run(session, document.id, file_storage=file_storage, vector_store=vector_store)

    assert DocumentLifecycleFindingCode.ACTIVE_VECTORS_MISSING in _codes(result)
    assert result.overall_status == AuditOverallStatus.INCONSISTENT


async def test_qdrant_provider_failure_is_not_classified_as_missing_vectors() -> None:
    session = _FakeAuditSession()
    document = _indexed_document()
    _seed(session, document)
    session.ingestion_jobs[str(uuid.uuid4())] = _ingestion_job(document.id, IngestionStatus.COMPLETED)

    file_storage = _FakeFileStorage(existing_keys={document.storage_key})
    vector_store = _FakeVectorStore(unavailable_collections={document.collection_name})

    result = await _run(session, document.id, file_storage=file_storage, vector_store=vector_store)

    codes = _codes(result)
    assert DocumentLifecycleFindingCode.ACTIVE_VECTORS_MISSING not in codes
    assert DocumentLifecycleFindingCode.ACTIVE_COLLECTION_MISSING not in codes
    assert DocumentLifecycleFindingCode.VECTOR_INSPECTION_UNAVAILABLE in codes
    assert result.vector_state.inspected is False


# --- 8/9: index metadata + ingestion lifecycle interpretation --------------------------------


async def test_incomplete_active_index_metadata_after_completed_ingestion_is_classified() -> None:
    session = _FakeAuditSession()
    document = _document(collection_name=None)  # completed ingestion but no index metadata
    _seed(session, document)
    session.ingestion_jobs[str(uuid.uuid4())] = _ingestion_job(document.id, IngestionStatus.COMPLETED)

    result = await _run(session, document.id)

    assert DocumentLifecycleFindingCode.INDEX_METADATA_INCOMPLETE in _codes(result)
    assert result.overall_status == AuditOverallStatus.INCONSISTENT


async def test_failed_ingestion_without_active_index_is_not_misclassified() -> None:
    session = _FakeAuditSession()
    document = _document(collection_name=None)
    _seed(session, document)
    session.ingestion_jobs[str(uuid.uuid4())] = _ingestion_job(document.id, IngestionStatus.FAILED)

    result = await _run(
        session, document.id, file_storage=_FakeFileStorage(existing_keys={document.storage_key})
    )

    codes = _codes(result)
    assert DocumentLifecycleFindingCode.ACTIVE_VECTORS_MISSING not in codes
    assert DocumentLifecycleFindingCode.INDEX_METADATA_INCOMPLETE not in codes
    assert result.overall_status == AuditOverallStatus.CONSISTENT


# --- 10/11: stale/active ingestion ---------------------------------------------------------------


async def test_stale_ingestion_is_reported_using_existing_stale_policy() -> None:
    session = _FakeAuditSession()
    document = _document(collection_name=None)
    _seed(session, document)
    stale_cutoff = datetime.now(UTC) - timedelta(seconds=_SETTINGS.ingestion_stale_after_seconds + 1)
    session.ingestion_jobs[str(uuid.uuid4())] = _ingestion_job(
        document.id, IngestionStatus.PROCESSING, updated_at=stale_cutoff
    )

    result = await _run(session, document.id)

    assert DocumentLifecycleFindingCode.STALE_INGESTION_JOB in _codes(result)
    stale_finding = next(
        f for f in result.findings if f.code == DocumentLifecycleFindingCode.STALE_INGESTION_JOB
    )
    assert stale_finding.severity == FindingSeverity.WARNING


async def test_active_non_stale_ingestion_is_reported_as_transitional() -> None:
    session = _FakeAuditSession()
    document = _document(collection_name=None)
    _seed(session, document)
    session.ingestion_jobs[str(uuid.uuid4())] = _ingestion_job(
        document.id, IngestionStatus.PROCESSING, updated_at=datetime.now(UTC)
    )

    result = await _run(
        session, document.id, file_storage=_FakeFileStorage(existing_keys={document.storage_key})
    )

    assert DocumentLifecycleFindingCode.INGESTION_IN_PROGRESS in _codes(result)
    finding = next(f for f in result.findings if f.code == DocumentLifecycleFindingCode.INGESTION_IN_PROGRESS)
    assert finding.severity == FindingSeverity.INFO
    assert result.overall_status == AuditOverallStatus.CONSISTENT


# --- 12/13: deletion lifecycle -------------------------------------------------------------------


async def test_partially_failed_deletion_is_reported() -> None:
    session = _FakeAuditSession()
    document = _indexed_document()
    _seed(session, document)
    session.deletion_jobs[str(uuid.uuid4())] = _deletion_job(
        document.id, DocumentDeletionStatus.PARTIALLY_FAILED
    )
    file_storage = _FakeFileStorage(existing_keys={document.storage_key})
    vector_store = _FakeVectorStore(
        collection_sizes={document.collection_name: 768},
        vector_counts={(document.collection_name, document.id): 1},
    )

    result = await _run(session, document.id, file_storage=file_storage, vector_store=vector_store)

    assert DocumentLifecycleFindingCode.DELETION_PARTIALLY_FAILED in _codes(result)
    finding = next(
        f for f in result.findings if f.code == DocumentLifecycleFindingCode.DELETION_PARTIALLY_FAILED
    )
    assert finding.severity == FindingSeverity.WARNING


async def test_completed_deletion_suppresses_external_checks() -> None:
    session = _FakeAuditSession()
    document = _indexed_document()
    _seed(session, document)
    session.deletion_jobs[str(uuid.uuid4())] = _deletion_job(document.id, DocumentDeletionStatus.COMPLETED)
    # No file_storage/vector_store keys registered — if the audit tried to inspect them, it would
    # otherwise report OBJECT_MISSING/ACTIVE_COLLECTION_MISSING/ACTIVE_VECTORS_MISSING.
    file_storage = _FakeFileStorage(existing_keys=set())
    vector_store = _FakeVectorStore(collection_sizes={})

    result = await _run(session, document.id, file_storage=file_storage, vector_store=vector_store)

    codes = _codes(result)
    assert DocumentLifecycleFindingCode.OBJECT_MISSING not in codes
    assert DocumentLifecycleFindingCode.ACTIVE_COLLECTION_MISSING not in codes
    assert DocumentLifecycleFindingCode.ACTIVE_VECTORS_MISSING not in codes
    assert result.storage_state is None
    assert result.vector_state is None
    assert result.overall_status == AuditOverallStatus.CONSISTENT


# --- 14/15: vector cleanup -------------------------------------------------------------------


async def test_failed_cleanup_job_produces_vector_cleanup_incomplete() -> None:
    session = _FakeAuditSession()
    document = _indexed_document()
    _seed(session, document)
    session.ingestion_jobs[str(uuid.uuid4())] = _ingestion_job(document.id, IngestionStatus.COMPLETED)
    session.cleanup_jobs[str(uuid.uuid4())] = _cleanup_job(
        document.id, collection_name="unrelated-old-collection", status=VectorCleanupStatus.FAILED
    )
    file_storage = _FakeFileStorage(existing_keys={document.storage_key})
    vector_store = _FakeVectorStore(
        collection_sizes={document.collection_name: 768},
        vector_counts={(document.collection_name, document.id): 1},
    )

    result = await _run(session, document.id, file_storage=file_storage, vector_store=vector_store)

    assert DocumentLifecycleFindingCode.VECTOR_CLEANUP_INCOMPLETE in _codes(result)


async def test_completed_cleanup_job_produces_no_cleanup_finding() -> None:
    session = _FakeAuditSession()
    document = _indexed_document()
    _seed(session, document)
    session.ingestion_jobs[str(uuid.uuid4())] = _ingestion_job(document.id, IngestionStatus.COMPLETED)
    session.cleanup_jobs[str(uuid.uuid4())] = _cleanup_job(
        document.id, collection_name="old-collection", status=VectorCleanupStatus.COMPLETED
    )
    file_storage = _FakeFileStorage(existing_keys={document.storage_key})
    vector_store = _FakeVectorStore(
        collection_sizes={document.collection_name: 768},
        vector_counts={(document.collection_name, document.id): 1},
    )

    result = await _run(session, document.id, file_storage=file_storage, vector_store=vector_store)

    assert DocumentLifecycleFindingCode.VECTOR_CLEANUP_INCOMPLETE not in _codes(result)
    assert result.overall_status == AuditOverallStatus.CONSISTENT


# --- 16/17/18: re-index build/activation ------------------------------------------------------


async def test_completed_unactivated_reindex_produces_target_built_not_activated() -> None:
    session = _FakeAuditSession()
    document = _indexed_document()
    _seed(session, document)
    session.ingestion_jobs[str(uuid.uuid4())] = _ingestion_job(document.id, IngestionStatus.COMPLETED)
    session.reindex_jobs[str(uuid.uuid4())] = _reindex_job(
        document.id, source_collection_name=document.collection_name, activated_at=None
    )
    file_storage = _FakeFileStorage(existing_keys={document.storage_key})
    vector_store = _FakeVectorStore(
        collection_sizes={document.collection_name: 768},
        vector_counts={(document.collection_name, document.id): 1},
    )

    result = await _run(session, document.id, file_storage=file_storage, vector_store=vector_store)

    assert DocumentLifecycleFindingCode.REINDEX_TARGET_BUILT_NOT_ACTIVATED in _codes(result)
    finding = next(
        f for f in result.findings
        if f.code == DocumentLifecycleFindingCode.REINDEX_TARGET_BUILT_NOT_ACTIVATED
    )
    assert finding.severity == FindingSeverity.INFO
    # Valid build-ahead state — not classified as broken serving index.
    assert result.overall_status == AuditOverallStatus.CONSISTENT


async def test_valid_build_ahead_reindex_state_is_not_classified_as_broken_serving_index() -> None:
    """Document still serves the source collection; a target was already built (not yet
    activated); both source and target vectors exist. The audit must judge the document healthy
    against the collection it actually serves (source), never against the not-yet-active target."""
    session = _FakeAuditSession()
    source_collection = "documents__ollama__old__ev0__cv0__d768"
    target_collection = "documents__ollama__model__ev1__cv1__d768"
    document = _indexed_document(collection_name=source_collection)  # still serves the source
    _seed(session, document)
    session.ingestion_jobs[str(uuid.uuid4())] = _ingestion_job(document.id, IngestionStatus.COMPLETED)
    session.reindex_jobs[str(uuid.uuid4())] = _reindex_job(
        document.id,
        source_collection_name=source_collection,
        target_collection_name=target_collection,
        activated_at=None,
    )
    file_storage = _FakeFileStorage(existing_keys={document.storage_key})
    vector_store = _FakeVectorStore(
        collection_sizes={source_collection: 768, target_collection: 768},
        vector_counts={
            (source_collection, document.id): 1,  # source vectors still exist
            (target_collection, document.id): 1,  # target vectors already exist (build-ahead)
        },
    )

    result = await _run(session, document.id, file_storage=file_storage, vector_store=vector_store)

    codes = _codes(result)
    assert DocumentLifecycleFindingCode.ACTIVE_COLLECTION_MISSING not in codes
    assert DocumentLifecycleFindingCode.ACTIVE_VECTORS_MISSING not in codes
    assert DocumentLifecycleFindingCode.REINDEX_TARGET_BUILT_NOT_ACTIVATED in codes
    assert result.overall_status == AuditOverallStatus.CONSISTENT
    assert result.vector_state.has_vectors is True  # judged against the currently-serving source


async def test_activated_reindex_with_pending_cleanup_produces_reindex_cleanup_pending() -> None:
    session = _FakeAuditSession()
    document = _indexed_document()  # document already serves the target
    _seed(session, document)
    session.ingestion_jobs[str(uuid.uuid4())] = _ingestion_job(document.id, IngestionStatus.COMPLETED)
    session.reindex_jobs[str(uuid.uuid4())] = _reindex_job(
        document.id,
        source_collection_name="documents__ollama__old__ev0__cv0__d768",
        target_collection_name=document.collection_name,
        activated_at=_BASE_TIME,
    )
    session.cleanup_jobs[str(uuid.uuid4())] = _cleanup_job(
        document.id,
        collection_name="documents__ollama__old__ev0__cv0__d768",
        status=VectorCleanupStatus.PENDING,
    )
    file_storage = _FakeFileStorage(existing_keys={document.storage_key})
    vector_store = _FakeVectorStore(
        collection_sizes={document.collection_name: 768},
        vector_counts={(document.collection_name, document.id): 1},
    )

    result = await _run(session, document.id, file_storage=file_storage, vector_store=vector_store)

    codes = _codes(result)
    assert DocumentLifecycleFindingCode.REINDEX_CLEANUP_PENDING in codes
    assert DocumentLifecycleFindingCode.VECTOR_CLEANUP_INCOMPLETE not in codes  # not double-reported
    # Document may still be serving correctly from the target — not marked unusable.
    finding = next(
        f for f in result.findings if f.code == DocumentLifecycleFindingCode.REINDEX_CLEANUP_PENDING
    )
    assert finding.severity == FindingSeverity.WARNING


# --- no mutation ---------------------------------------------------------------------------------


async def test_audit_performs_no_lifecycle_mutations() -> None:
    session = _FakeAuditSession()
    document = _indexed_document()
    original_collection_name = document.collection_name
    _seed(session, document)
    ingestion_job = _ingestion_job(document.id, IngestionStatus.COMPLETED)
    session.ingestion_jobs[ingestion_job.id] = ingestion_job

    await _run(session, document.id)

    assert document.collection_name == original_collection_name
    assert ingestion_job.status == IngestionStatus.COMPLETED


async def test_audit_does_not_commit() -> None:
    """The fake session exposes no `.commit()` at all — the audit must never call it."""
    session = _FakeAuditSession()
    document = _indexed_document()
    _seed(session, document)

    assert not hasattr(session, "commit")
    await _run(session, document.id)  # would raise AttributeError if the audit tried to commit


async def test_audit_does_not_delete_vectors_or_objects() -> None:
    class _NoDeleteFileStorage(_FakeFileStorage):
        async def delete(self, key: str) -> None:  # pragma: no cover - must never be called
            raise AssertionError("audit must never delete objects")

    class _NoDeleteVectorStore(_FakeVectorStore):
        async def delete_by_document_id(self, collection_name: str, document_id: str) -> None:
            raise AssertionError("audit must never delete vectors")

    session = _FakeAuditSession()
    document = _indexed_document()
    _seed(session, document)

    await _run(
        session,
        document.id,
        file_storage=_NoDeleteFileStorage(existing_keys={document.storage_key}),
        vector_store=_NoDeleteVectorStore(
            collection_sizes={document.collection_name: 768},
            vector_counts={(document.collection_name, document.id): 1},
        ),
    )
    # No assertion error raised means the delete paths were never invoked.


# --- sanitization ---------------------------------------------------------------------------


async def test_provider_error_messages_are_bounded_and_sanitized() -> None:
    session = _FakeAuditSession()
    document = _indexed_document()
    _seed(session, document)

    secret_detail = "connection refused at internal-qdrant-host:6333 with credential XYZ"

    class _LeakyVectorStore(_FakeVectorStore):
        async def get_collection_vector_size(self, collection_name: str) -> int | None:
            raise QdrantVectorStoreError(secret_detail)

    result = await _run(
        session,
        document.id,
        file_storage=_FakeFileStorage(existing_keys={document.storage_key}),
        vector_store=_LeakyVectorStore(),
    )

    for finding in result.findings:
        assert secret_detail not in finding.summary
        assert secret_detail not in finding.actual_state
        assert secret_detail not in finding.expected_state
        assert secret_detail not in finding.suggested_action
