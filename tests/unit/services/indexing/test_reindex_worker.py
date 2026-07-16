"""Unit tests for app.services.indexing.reindex_worker against a fake session double.

Covers ReindexWorker.process_next_job()'s claim/target-reconstruction/build-delegation/lifecycle
decision table — orchestration only. A narrow fake `build_reindex_target` delegate is used
throughout (never the real extraction/chunking/embedding pipeline) so these tests focus on what
the worker itself is responsible for. Real PostgreSQL row-locking/concurrency behavior is covered
separately by tests/integration/indexing/test_reindex_worker_postgres.py; one real-build scenario
(genuine extraction/chunking/fake-embedding/real Qdrant) lives in
tests/integration/indexing/test_reindex_worker_build_postgres.py.
"""

import inspect
import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

import app.services.indexing.reindex_worker as reindex_worker_module
from app.core.config import Settings
from app.models.document import Document
from app.models.document_deletion_job import DocumentDeletionJob, DocumentDeletionStatus
from app.models.index_collection import IndexCollection
from app.models.reindex_job import ReindexJob, ReindexJobStatus
from app.rag.embedding_config import EmbeddingIndexConfig
from app.services.indexing.reindex_worker import (
    ReindexWorker,
    ReindexWorkerOutcome,
)

_BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


# --- fake session -----------------------------------------------------------------------------


class _Scalars:
    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def first(self) -> Any | None:
        return self._items[0] if self._items else None


class _ListResult:
    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def scalars(self) -> _Scalars:
        return _Scalars(self._items)

    def scalar_one_or_none(self) -> Any | None:
        return self._items[0] if self._items else None


class _FakeWorkerSession:
    """In-memory AsyncSession double for ReindexWorker, dispatching SELECTs by compiled SQL."""

    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.deletion_jobs: dict[str, DocumentDeletionJob] = {}
        self.reindex_jobs: dict[str, ReindexJob] = {}
        self.index_collections: dict[str, IndexCollection] = {}
        self.commit_count = 0
        self.rollback_count = 0
        self.get_calls: list[tuple[type, str]] = []

    def add(self, instance: object) -> None:
        if isinstance(instance, Document):
            self.documents[instance.id] = instance
        elif isinstance(instance, DocumentDeletionJob):
            self.deletion_jobs[instance.id] = instance
        elif isinstance(instance, ReindexJob):
            self.reindex_jobs[instance.id] = instance
        elif isinstance(instance, IndexCollection):
            self.index_collections[instance.collection_name] = instance

    async def get(self, model: type, instance_id: str) -> object | None:
        self.get_calls.append((model, instance_id))
        if model is Document:
            return self.documents.get(instance_id)
        if model is ReindexJob:
            return self.reindex_jobs.get(instance_id)
        if model is IndexCollection:
            return self.index_collections.get(instance_id)
        return None

    async def execute(self, stmt: Any) -> _ListResult:
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))

        if "reindex_jobs" in compiled:
            jobs = list(self.reindex_jobs.values())
            eq_status = re.search(r"reindex_jobs\.status = '([^']*)'", compiled)
            if eq_status:
                jobs = [job for job in jobs if job.status.value == eq_status.group(1)]
            jobs.sort(key=lambda job: (job.created_at, job.id))
            limit_match = re.search(r"LIMIT (\d+)", compiled)
            if limit_match:
                jobs = jobs[: int(limit_match.group(1))]
            return _ListResult(jobs)

        if "document_deletion_jobs" in compiled:
            jobs = list(self.deletion_jobs.values())
            eq_match = re.search(r"document_deletion_jobs\.document_id = '([^']*)'", compiled)
            if eq_match:
                jobs = [job for job in jobs if job.document_id == eq_match.group(1)]
            jobs.sort(key=lambda job: (job.created_at, job.id), reverse=True)
            limit_match = re.search(r"LIMIT (\d+)", compiled)
            if limit_match:
                jobs = jobs[: int(limit_match.group(1))]
            # A column-only SELECT (e.g. `.status` alone) renders as
            # "SELECT document_deletion_jobs.status \nFROM ..." — checking the SELECT clause
            # prefix specifically (not just presence anywhere) avoids false-matching the
            # `document_deletion_jobs.id DESC` ORDER BY tiebreaker a full-row query also has.
            if compiled.startswith("SELECT document_deletion_jobs.status "):
                return _ListResult([job.status for job in jobs])
            return _ListResult(jobs)

        return _ListResult([])

    async def commit(self) -> None:
        self.commit_count += 1

    async def rollback(self) -> None:
        self.rollback_count += 1


# --- builders -----------------------------------------------------------------------------------


def _document(**overrides: object) -> Document:
    fields: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        original_filename="notes.txt",
        stored_filename=f"{uuid.uuid4().hex}.txt",
        content_type="text/plain",
        file_size=100,
        stored_path="unset",
        collection_name="serving-collection",
    )
    fields.update(overrides)
    return Document(**fields)  # type: ignore[arg-type]


def _index_collection(**overrides: object) -> IndexCollection:
    fields: dict[str, object] = dict(
        collection_name="documents__ollama__target-model__ev9__cv9__d3",
        embedding_provider="ollama",
        embedding_model="target-model",
        embedding_dimension=3,
        embedding_version="v9",
        chunking_version="v9",
    )
    fields.update(overrides)
    return IndexCollection(**fields)  # type: ignore[arg-type]


def _reindex_job(document_id: str, target_collection_name: str, **overrides: object) -> ReindexJob:
    fields: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        document_id=document_id,
        source_collection_name="serving-collection",  # matches _document()'s default collection_name
        target_collection_name=target_collection_name,
        target_chunk_size=500,
        target_chunk_overlap=50,
        status=ReindexJobStatus.PENDING,
        created_at=_BASE_TIME,
    )
    fields.update(overrides)
    return ReindexJob(**fields)  # type: ignore[arg-type]


def _base_settings(**overrides: object) -> Settings:
    fields: dict[str, object] = dict(
        EMBEDDING_PROVIDER="ollama",
        OLLAMA_EMBEDDING_MODEL="live-model",
        EMBEDDING_MODEL=None,
        VECTOR_SIZE=999,
        EMBEDDING_VERSION="v-live",
        CHUNKING_VERSION="v-live",
        QDRANT_COLLECTION_NAME="documents",
        CHUNK_SIZE=111,
        CHUNK_OVERLAP=22,
    )
    fields.update(overrides)
    return Settings(**fields)  # type: ignore[arg-type]


class _RecordingBuildDelegate:
    """A fake build_reindex_target() spy: records every call, optionally raises."""

    def __init__(self, raise_message: str | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.raise_message = raise_message

    async def __call__(
        self,
        document: Document,
        session: Any,
        settings: Settings,
        file_storage: Any,
        target_config: EmbeddingIndexConfig,
        *,
        target_chunk_size: int,
        target_chunk_overlap: int,
    ) -> object:
        self.calls.append(
            dict(
                document=document,
                settings=settings,
                target_config=target_config,
                target_chunk_size=target_chunk_size,
                target_chunk_overlap=target_chunk_overlap,
                commit_count_at_call=session.commit_count,
            )
        )
        if self.raise_message is not None:
            raise RuntimeError(self.raise_message)
        return object()


def _worker(monkeypatch: pytest.MonkeyPatch, delegate: _RecordingBuildDelegate) -> ReindexWorker:
    monkeypatch.setattr(reindex_worker_module, "build_reindex_target", delegate)
    return ReindexWorker(file_storage=object())  # type: ignore[arg-type]


# --- claiming -------------------------------------------------------------------------------


async def test_no_pending_job_returns_no_job(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeWorkerSession()
    worker = _worker(monkeypatch, _RecordingBuildDelegate())

    result = await worker.process_next_job(session, _base_settings())

    assert result.outcome == ReindexWorkerOutcome.NO_JOB
    assert result.job_id is None
    assert result.document_id is None


async def test_worker_claims_exactly_one_pending_job(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeWorkerSession()
    document = _document()
    session.documents[document.id] = document
    collection = _index_collection()
    session.index_collections[collection.collection_name] = collection
    job_a = _reindex_job(document.id, collection.collection_name, created_at=_BASE_TIME)
    job_b = _reindex_job(
        document.id, collection.collection_name, created_at=_BASE_TIME + timedelta(minutes=5)
    )
    session.reindex_jobs[job_a.id] = job_a
    session.reindex_jobs[job_b.id] = job_b
    delegate = _RecordingBuildDelegate()

    result = await _worker(monkeypatch, delegate).process_next_job(session, _base_settings())

    assert len(delegate.calls) == 1
    assert result.job_id in (job_a.id, job_b.id)
    remaining_pending = [j for j in session.reindex_jobs.values() if j.status == ReindexJobStatus.PENDING]
    assert len(remaining_pending) == 1  # only one job was claimed


async def test_claim_order_is_deterministic_oldest_first(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeWorkerSession()
    document = _document()
    session.documents[document.id] = document
    collection = _index_collection()
    session.index_collections[collection.collection_name] = collection
    older = _reindex_job(document.id, collection.collection_name, created_at=_BASE_TIME)
    newer = _reindex_job(
        document.id, collection.collection_name, created_at=_BASE_TIME + timedelta(minutes=5)
    )
    session.reindex_jobs[newer.id] = newer
    session.reindex_jobs[older.id] = older  # inserted out of order deliberately
    delegate = _RecordingBuildDelegate()

    result = await _worker(monkeypatch, delegate).process_next_job(session, _base_settings())

    assert result.job_id == older.id


async def test_claimed_job_becomes_processing_before_build(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeWorkerSession()
    document = _document()
    session.documents[document.id] = document
    collection = _index_collection()
    session.index_collections[collection.collection_name] = collection
    job = _reindex_job(document.id, collection.collection_name)
    session.reindex_jobs[job.id] = job

    class _AssertsProcessingDelegate(_RecordingBuildDelegate):
        async def __call__(self, document, session, *args, **kwargs):  # type: ignore[override]
            assert session.reindex_jobs[job.id].status == ReindexJobStatus.PROCESSING
            return await super().__call__(document, session, *args, **kwargs)

    await _worker(monkeypatch, _AssertsProcessingDelegate()).process_next_job(session, _base_settings())


async def test_claim_commits_before_external_build_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeWorkerSession()
    document = _document()
    session.documents[document.id] = document
    collection = _index_collection()
    session.index_collections[collection.collection_name] = collection
    job = _reindex_job(document.id, collection.collection_name)
    session.reindex_jobs[job.id] = job
    delegate = _RecordingBuildDelegate()

    await _worker(monkeypatch, delegate).process_next_job(session, _base_settings())

    assert delegate.calls[0]["commit_count_at_call"] == 1  # claim's commit already happened


@pytest.mark.parametrize(
    "status", [ReindexJobStatus.PROCESSING, ReindexJobStatus.COMPLETED, ReindexJobStatus.FAILED]
)
async def test_non_pending_jobs_are_never_claimed(
    status: ReindexJobStatus, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _FakeWorkerSession()
    document = _document()
    session.documents[document.id] = document
    collection = _index_collection()
    session.index_collections[collection.collection_name] = collection
    job = _reindex_job(document.id, collection.collection_name, status=status)
    session.reindex_jobs[job.id] = job
    delegate = _RecordingBuildDelegate()

    result = await _worker(monkeypatch, delegate).process_next_job(session, _base_settings())

    assert result.outcome == ReindexWorkerOutcome.NO_JOB
    assert delegate.calls == []
    assert session.reindex_jobs[job.id].status == status  # untouched


# --- target reconstruction -----------------------------------------------------------------------


async def test_worker_reconstructs_target_from_index_collection_not_live_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeWorkerSession()
    document = _document()
    session.documents[document.id] = document
    collection = _index_collection(
        collection_name="documents__ollama__target-model__ev9__cv9__d3",
        embedding_provider="ollama",
        embedding_model="target-model",
        embedding_dimension=3,
        embedding_version="v9",
        chunking_version="v9",
    )
    session.index_collections[collection.collection_name] = collection
    job = _reindex_job(
        document.id, collection.collection_name, target_chunk_size=777, target_chunk_overlap=88
    )
    session.reindex_jobs[job.id] = job
    delegate = _RecordingBuildDelegate()
    live_settings = _base_settings(
        EMBEDDING_PROVIDER="a-different-live-provider",
        OLLAMA_EMBEDDING_MODEL="a-different-live-model",
        VECTOR_SIZE=1,
        EMBEDDING_VERSION="a-different-live-version",
        CHUNKING_VERSION="a-different-live-chunking-version",
    )

    await _worker(monkeypatch, delegate).process_next_job(session, live_settings)

    used_config = delegate.calls[0]["target_config"]
    assert used_config.provider == "ollama"  # test 9
    assert used_config.model == "target-model"  # test 10
    assert used_config.dimension == 3  # test 11
    assert used_config.embedding_version == "v9"  # test 12
    assert used_config.chunking_version == "v9"  # test 13
    assert delegate.calls[0]["target_chunk_size"] == 777  # test 14
    assert delegate.calls[0]["target_chunk_overlap"] == 88  # test 15
    # None of the live settings' values leaked into the reconstructed target (test 16).
    assert used_config.provider != live_settings.embedding_provider
    assert used_config.model != live_settings.ollama_embedding_model
    assert used_config.dimension != live_settings.vector_size
    assert used_config.embedding_version != live_settings.embedding_version
    assert used_config.chunking_version != live_settings.chunking_version


async def test_worker_delegates_exactly_once_to_build_reindex_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeWorkerSession()
    document = _document()
    session.documents[document.id] = document
    collection = _index_collection()
    session.index_collections[collection.collection_name] = collection
    job = _reindex_job(document.id, collection.collection_name)
    session.reindex_jobs[job.id] = job
    delegate = _RecordingBuildDelegate()

    await _worker(monkeypatch, delegate).process_next_job(session, _base_settings())

    assert len(delegate.calls) == 1


# --- success handling -----------------------------------------------------------------------------


async def test_successful_build_marks_job_completed_with_timestamp_and_no_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeWorkerSession()
    document = _document(
        collection_name="serving-collection",
        embedding_provider="ollama",
        embedding_model="serving-model",
        embedding_version="v-serving",
        chunking_version="v-serving",
    )
    session.documents[document.id] = document
    collection = _index_collection()
    session.index_collections[collection.collection_name] = collection
    job = _reindex_job(document.id, collection.collection_name)
    job.error_message = "leftover from a hypothetical prior attempt"
    session.reindex_jobs[job.id] = job

    result = await _worker(monkeypatch, _RecordingBuildDelegate()).process_next_job(
        session, _base_settings()
    )

    assert result.outcome == ReindexWorkerOutcome.COMPLETED  # test 18
    completed = session.reindex_jobs[job.id]
    assert completed.status == ReindexJobStatus.COMPLETED
    assert completed.completed_at is not None  # test 19
    assert completed.error_message is None  # test 20

    # Document serving metadata is completely untouched (test 21).
    assert document.collection_name == "serving-collection"
    assert document.embedding_provider == "ollama"
    assert document.embedding_model == "serving-model"
    assert document.embedding_version == "v-serving"
    assert document.chunking_version == "v-serving"

    assert session.deletion_jobs == {}  # test 22: no cleanup/deletion-adjacent job created


async def test_successful_build_creates_no_cleanup_job(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeWorkerSession()
    document = _document()
    session.documents[document.id] = document
    collection = _index_collection()
    session.index_collections[collection.collection_name] = collection
    job = _reindex_job(document.id, collection.collection_name)
    session.reindex_jobs[job.id] = job

    await _worker(monkeypatch, _RecordingBuildDelegate()).process_next_job(session, _base_settings())

    # Check actual code (constructor/call syntax), not the module docstring's prose explaining
    # what the worker deliberately does *not* do.
    code_source = "".join(
        inspect.getsource(member)
        for _, member in inspect.getmembers(reindex_worker_module, inspect.isfunction)
    ) + inspect.getsource(ReindexWorker)
    assert "VectorCleanupJob(" not in code_source  # test 22
    assert "delete_by_document_id(" not in code_source  # test 23: no old-vector deletion anywhere


# --- failure handling -----------------------------------------------------------------------------


async def test_build_failure_marks_job_failed_with_bounded_message_and_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeWorkerSession()
    document = _document()
    session.documents[document.id] = document
    collection = _index_collection()
    session.index_collections[collection.collection_name] = collection
    job = _reindex_job(document.id, collection.collection_name)
    session.reindex_jobs[job.id] = job
    oversized_message = "x" * 3000
    delegate = _RecordingBuildDelegate(raise_message=oversized_message)

    result = await _worker(monkeypatch, delegate).process_next_job(session, _base_settings())

    assert result.outcome == ReindexWorkerOutcome.FAILED  # test 24
    failed = session.reindex_jobs[job.id]
    assert failed.status == ReindexJobStatus.FAILED
    assert failed.error_message is not None
    assert len(failed.error_message) <= 2048  # test 25
    assert failed.completed_at is not None  # test 26
    assert session.rollback_count == 1


async def test_build_failure_does_not_create_a_retry_job(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeWorkerSession()
    document = _document()
    session.documents[document.id] = document
    collection = _index_collection()
    session.index_collections[collection.collection_name] = collection
    job = _reindex_job(document.id, collection.collection_name)
    session.reindex_jobs[job.id] = job
    delegate = _RecordingBuildDelegate(raise_message="boom")

    await _worker(monkeypatch, delegate).process_next_job(session, _base_settings())

    assert len(session.reindex_jobs) == 1  # test 27: no new row appeared


async def test_missing_document_terminates_the_job_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeWorkerSession()
    collection = _index_collection()
    session.index_collections[collection.collection_name] = collection
    missing_document_id = str(uuid.uuid4())
    job = _reindex_job(missing_document_id, collection.collection_name)
    session.reindex_jobs[job.id] = job
    delegate = _RecordingBuildDelegate()

    result = await _worker(monkeypatch, delegate).process_next_job(session, _base_settings())

    assert result.outcome == ReindexWorkerOutcome.FAILED  # test 28
    assert session.reindex_jobs[job.id].status == ReindexJobStatus.FAILED
    assert delegate.calls == []


# --- deletion defense-in-depth -----------------------------------------------------------------


@pytest.mark.parametrize(
    "status",
    [
        DocumentDeletionStatus.PENDING,
        DocumentDeletionStatus.PROCESSING,
        DocumentDeletionStatus.PARTIALLY_FAILED,
        DocumentDeletionStatus.COMPLETED,
    ],
)
async def test_deletion_lifecycle_states_block_build(
    status: DocumentDeletionStatus, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _FakeWorkerSession()
    document = _document()
    session.documents[document.id] = document
    collection = _index_collection()
    session.index_collections[collection.collection_name] = collection
    job = _reindex_job(document.id, collection.collection_name)
    session.reindex_jobs[job.id] = job
    deletion_job = DocumentDeletionJob(
        id=str(uuid.uuid4()),
        document_id=document.id,
        status=status,
        vector_cleanup_completed=False,
        storage_cleanup_completed=False,
        created_at=_BASE_TIME,
    )
    session.deletion_jobs[deletion_job.id] = deletion_job
    delegate = _RecordingBuildDelegate()

    result = await _worker(monkeypatch, delegate).process_next_job(session, _base_settings())

    assert result.outcome == ReindexWorkerOutcome.SKIPPED_DELETED  # tests 29-32
    assert session.reindex_jobs[job.id].status == ReindexJobStatus.FAILED
    assert delegate.calls == []  # test 33


# --- rollback safety -------------------------------------------------------------------------------


def test_scalar_identifiers_are_captured_before_any_rollback() -> None:
    """Structural proof (test 34/35): job_id/document_id must be captured immediately after the
    claim commit — textually before the rollback() call in the failure path — so the failure
    handler never re-reads attributes off the original (possibly-expired) `job` object.
    """
    source = inspect.getsource(ReindexWorker.process_next_job)
    capture_index = source.index("job_id = job.id")
    rollback_index = source.index("await session.rollback()")
    assert capture_index < rollback_index


async def test_failure_path_reloads_the_job_by_id_after_rollback(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeWorkerSession()
    document = _document()
    session.documents[document.id] = document
    collection = _index_collection()
    session.index_collections[collection.collection_name] = collection
    job = _reindex_job(document.id, collection.collection_name)
    session.reindex_jobs[job.id] = job
    delegate = _RecordingBuildDelegate(raise_message="boom")

    result = await _worker(monkeypatch, delegate).process_next_job(session, _base_settings())

    assert result.job_id == job.id
    assert result.document_id == document.id
    # The job was re-fetched by its scalar id via session.get(), not reused from the stale reference.
    assert (ReindexJob, job.id) in session.get_calls


# --- no activation, at most one job per call ------------------------------------------------------


async def test_worker_never_activates_the_target(monkeypatch: pytest.MonkeyPatch) -> None:
    # No import of either activation-adjacent symbol — the docstring's prose mention (explaining
    # what the worker deliberately does *not* do) is not a code reference, so a plain "not in
    # module source" check would false-positive; checking the import list is precise instead.
    assert "activate_reindexed_document" not in dir(reindex_worker_module)
    assert "mark_document_indexed" not in dir(reindex_worker_module)
    class_source = inspect.getsource(ReindexWorker)
    assert "activate_reindexed_document(" not in class_source
    assert "mark_document_indexed(" not in class_source


async def test_worker_processes_at_most_one_job_per_invocation(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeWorkerSession()
    document = _document()
    session.documents[document.id] = document
    collection = _index_collection()
    session.index_collections[collection.collection_name] = collection
    job_a = _reindex_job(document.id, collection.collection_name, created_at=_BASE_TIME)
    job_b = _reindex_job(
        document.id, collection.collection_name, created_at=_BASE_TIME + timedelta(minutes=5)
    )
    session.reindex_jobs[job_a.id] = job_a
    session.reindex_jobs[job_b.id] = job_b
    delegate = _RecordingBuildDelegate()
    worker = _worker(monkeypatch, delegate)

    await worker.process_next_job(session, _base_settings())

    non_pending = [j for j in session.reindex_jobs.values() if j.status != ReindexJobStatus.PENDING]
    assert len(non_pending) == 1
    assert len(delegate.calls) == 1
