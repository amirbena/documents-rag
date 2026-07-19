"""Unit tests for app.services.indexing.reindex_scheduling_service against a fake session double.

Covers schedule_reindex()'s full decision table and the concurrent-insert race — scheduling
behavior only. Never exercises a real build/activation, Qdrant vectors, or object storage. Real
PostgreSQL row-locking/partial-unique-index behavior is covered separately by
tests/integration/indexing/test_reindex_scheduling_postgres.py.
"""

import inspect
import re
import uuid
from typing import Any

import asyncpg.exceptions
import pytest
from sqlalchemy.exc import IntegrityError

from app.models.document import Document
from app.models.document_deletion_job import DocumentDeletionJob, DocumentDeletionStatus
from app.models.index_collection import IndexCollection
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.models.reindex_job import ReindexJob, ReindexJobStatus
from app.services.indexing.reindex_scheduling_service import (
    MissingActiveReindexJobAfterRaceError,
    ReindexSchedulingOutcome,
    schedule_reindex,
)
from tests.support.indexing.builders import build_document, build_embedding_config
from tests.support.indexing.fakes import FakeVectorStore

_ONE_ACTIVE_REINDEX_JOB_CONSTRAINT = "ix_reindex_jobs_one_active_per_document"
_UNRELATED_CONSTRAINT = "reindex_jobs_document_id_fkey"


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


def _fake_orig_for(constraint_name: str | None) -> Exception:
    """Build an exception shaped like the real driver's, mirroring test_upload_service.py's helper."""
    if constraint_name is None:
        return RuntimeError("simulated unrelated DB failure with no constraint diagnostics")
    orig = asyncpg.exceptions.UniqueViolationError(
        f'duplicate key value violates unique constraint "{constraint_name}"'
    )
    orig.constraint_name = constraint_name
    return orig


class _FakeSchedulingSession:
    """In-memory AsyncSession double for schedule_reindex(), dispatching SELECTs by compiled SQL."""

    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.ingestion_jobs: dict[str, IngestionJob] = {}
        self.deletion_jobs: dict[str, DocumentDeletionJob] = {}
        self.reindex_jobs: dict[str, ReindexJob] = {}
        self.index_collections: dict[str, IndexCollection] = {}
        self._pending_new: list[object] = []
        self.commit_count = 0
        self.rollback_count = 0
        self.force_next_commit_integrity_error: str | None = None
        self.concurrent_winner_job: ReindexJob | None = None
        # When True, ensure_active_collection() never issues its own internal commit — used by
        # the race tests below so a forced commit-time IntegrityError only ever fires on the
        # ReindexJob insert's own commit, never on ensure_active_collection()'s.
        self.pretend_collection_already_tracked = False

    def add(self, instance: object) -> None:
        if isinstance(instance, ReindexJob):
            self._pending_new.append(instance)
        elif isinstance(instance, IndexCollection):
            self.index_collections[instance.collection_name] = instance
        elif isinstance(instance, DocumentDeletionJob):
            self.deletion_jobs[instance.id] = instance
        elif isinstance(instance, IngestionJob):
            self.ingestion_jobs[instance.id] = instance

    async def get(self, model: type, instance_id: str) -> object | None:
        if model is IndexCollection:
            if self.pretend_collection_already_tracked:
                self.index_collections.setdefault(instance_id, object())
            return self.index_collections.get(instance_id)
        if model is Document:
            return self.documents.get(instance_id)
        return None

    async def execute(self, stmt: Any) -> _ListResult:
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))

        if "reindex_jobs" in compiled:
            jobs = [*self.reindex_jobs.values(), *(j for j in self._pending_new if isinstance(j, ReindexJob))]
            eq_match = re.search(r"reindex_jobs\.document_id = '([^']*)'", compiled)
            if eq_match:
                jobs = [job for job in jobs if job.document_id == eq_match.group(1)]
            in_match = re.search(r"reindex_jobs\.status IN \(([^)]*)\)", compiled)
            if in_match:
                statuses = {token.strip().strip("'") for token in in_match.group(1).split(",")}
                jobs = [job for job in jobs if job.status.value in statuses]
            jobs.sort(key=lambda job: (job.created_at, job.id), reverse=True)
            limit_match = re.search(r"LIMIT (\d+)", compiled)
            if limit_match:
                jobs = jobs[: int(limit_match.group(1))]
            return _ListResult(jobs)

        if "ingestion_jobs" in compiled:
            jobs = list(self.ingestion_jobs.values())
            eq_match = re.search(r"ingestion_jobs\.document_id = '([^']*)'", compiled)
            if eq_match:
                jobs = [job for job in jobs if job.document_id == eq_match.group(1)]
            jobs.sort(key=lambda job: (job.created_at, job.id), reverse=True)
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
            return _ListResult(jobs)

        return _ListResult([])

    async def commit(self) -> None:
        if self.force_next_commit_integrity_error is not None:
            constraint = self.force_next_commit_integrity_error
            self.force_next_commit_integrity_error = None
            self._pending_new.clear()
            if self.concurrent_winner_job is not None:
                self.reindex_jobs[self.concurrent_winner_job.id] = self.concurrent_winner_job
            raise IntegrityError("INSERT", {}, _fake_orig_for(constraint))

        for job in self._pending_new:
            self.reindex_jobs[job.id] = job
        self._pending_new.clear()
        self.commit_count += 1

    async def rollback(self) -> None:
        self._pending_new.clear()
        self.rollback_count += 1


def _indexed_document(target_collection: str, **overrides: object) -> Document:
    fields: dict[str, object] = dict(collection_name=target_collection)
    fields.update(overrides)
    return build_document(**fields)


# --- Step 1: document eligibility ---------------------------------------------------------------


async def test_never_indexed_document_is_ineligible() -> None:
    session = _FakeSchedulingSession()
    document = build_document(collection_name=None)
    target = build_embedding_config()

    result = await schedule_reindex(
        session, document, FakeVectorStore(), target, target_chunk_size=500, target_chunk_overlap=50
    )

    assert result.outcome == ReindexSchedulingOutcome.INELIGIBLE_NEVER_INDEXED
    assert result.job is None
    assert session.commit_count == 0


# --- Step 2: current target check ---------------------------------------------------------------


async def test_document_already_on_target_returns_already_current() -> None:
    session = _FakeSchedulingSession()
    target = build_embedding_config()
    document = _indexed_document(target.collection_name)

    result = await schedule_reindex(
        session, document, FakeVectorStore(), target, target_chunk_size=500, target_chunk_overlap=50
    )

    assert result.outcome == ReindexSchedulingOutcome.ALREADY_CURRENT
    assert result.job is None
    assert session.commit_count == 0


# --- Step 3: existing active re-index ------------------------------------------------------------


async def test_existing_pending_reindex_returns_already_active() -> None:
    session = _FakeSchedulingSession()
    target = build_embedding_config()
    document = _indexed_document("documents__ollama__old__ev0__cv0__d768")
    existing = ReindexJob(
        id=str(uuid.uuid4()), document_id=document.id, target_collection_name=target.collection_name,
        target_chunk_size=500, target_chunk_overlap=50, status=ReindexJobStatus.PENDING,
    )
    session.reindex_jobs[existing.id] = existing

    result = await schedule_reindex(
        session, document, FakeVectorStore(), target, target_chunk_size=500, target_chunk_overlap=50
    )

    assert result.outcome == ReindexSchedulingOutcome.ALREADY_ACTIVE
    assert result.job is not None
    assert result.job.id == existing.id
    assert session.commit_count == 0


async def test_existing_processing_reindex_returns_already_active() -> None:
    session = _FakeSchedulingSession()
    target = build_embedding_config()
    document = _indexed_document("documents__ollama__old__ev0__cv0__d768")
    existing = ReindexJob(
        id=str(uuid.uuid4()), document_id=document.id, target_collection_name=target.collection_name,
        target_chunk_size=500, target_chunk_overlap=50, status=ReindexJobStatus.PROCESSING,
    )
    session.reindex_jobs[existing.id] = existing

    result = await schedule_reindex(
        session, document, FakeVectorStore(), target, target_chunk_size=500, target_chunk_overlap=50
    )

    assert result.outcome == ReindexSchedulingOutcome.ALREADY_ACTIVE
    assert result.job is not None
    assert result.job.id == existing.id


# --- Step 4: active ingestion interaction --------------------------------------------------------


async def test_pending_ingestion_blocks_scheduling() -> None:
    session = _FakeSchedulingSession()
    target = build_embedding_config()
    document = _indexed_document("documents__ollama__old__ev0__cv0__d768")
    job = IngestionJob(id=str(uuid.uuid4()), document_id=document.id, status=IngestionStatus.PENDING)
    session.ingestion_jobs[job.id] = job

    result = await schedule_reindex(
        session, document, FakeVectorStore(), target, target_chunk_size=500, target_chunk_overlap=50
    )

    assert result.outcome == ReindexSchedulingOutcome.INGESTION_ACTIVE
    assert result.job is None
    assert session.commit_count == 0


async def test_processing_ingestion_blocks_scheduling() -> None:
    session = _FakeSchedulingSession()
    target = build_embedding_config()
    document = _indexed_document("documents__ollama__old__ev0__cv0__d768")
    job = IngestionJob(id=str(uuid.uuid4()), document_id=document.id, status=IngestionStatus.PROCESSING)
    session.ingestion_jobs[job.id] = job

    result = await schedule_reindex(
        session, document, FakeVectorStore(), target, target_chunk_size=500, target_chunk_overlap=50
    )

    assert result.outcome == ReindexSchedulingOutcome.INGESTION_ACTIVE


async def test_failed_ingestion_does_not_block_with_valid_prior_index() -> None:
    session = _FakeSchedulingSession()
    target = build_embedding_config()
    document = _indexed_document("documents__ollama__old__ev0__cv0__d768")
    job = IngestionJob(id=str(uuid.uuid4()), document_id=document.id, status=IngestionStatus.FAILED)
    session.ingestion_jobs[job.id] = job

    result = await schedule_reindex(
        session, document, FakeVectorStore(), target, target_chunk_size=500, target_chunk_overlap=50
    )

    assert result.outcome == ReindexSchedulingOutcome.CREATED


async def test_completed_ingestion_does_not_block() -> None:
    session = _FakeSchedulingSession()
    target = build_embedding_config()
    document = _indexed_document("documents__ollama__old__ev0__cv0__d768")
    job = IngestionJob(id=str(uuid.uuid4()), document_id=document.id, status=IngestionStatus.COMPLETED)
    session.ingestion_jobs[job.id] = job

    result = await schedule_reindex(
        session, document, FakeVectorStore(), target, target_chunk_size=500, target_chunk_overlap=50
    )

    assert result.outcome == ReindexSchedulingOutcome.CREATED


# --- Step 5: deletion interaction -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("deletion_status", "expected_outcome"),
    [
        (DocumentDeletionStatus.PENDING, ReindexSchedulingOutcome.DELETION_ACTIVE),
        (DocumentDeletionStatus.PROCESSING, ReindexSchedulingOutcome.DELETION_ACTIVE),
        (DocumentDeletionStatus.PARTIALLY_FAILED, ReindexSchedulingOutcome.DELETION_INCOMPLETE),
        (DocumentDeletionStatus.COMPLETED, ReindexSchedulingOutcome.DOCUMENT_DELETED),
    ],
)
async def test_deletion_lifecycle_states_block_scheduling(
    deletion_status: DocumentDeletionStatus, expected_outcome: ReindexSchedulingOutcome
) -> None:
    session = _FakeSchedulingSession()
    target = build_embedding_config()
    document = _indexed_document("documents__ollama__old__ev0__cv0__d768")
    job = DocumentDeletionJob(
        id=str(uuid.uuid4()),
        document_id=document.id,
        status=deletion_status,
        vector_cleanup_completed=False,
        storage_cleanup_completed=False,
    )
    session.deletion_jobs[job.id] = job

    result = await schedule_reindex(
        session, document, FakeVectorStore(), target, target_chunk_size=500, target_chunk_overlap=50
    )

    assert result.outcome == expected_outcome
    assert result.job is None
    assert session.commit_count == 0


# --- Step 6: create the attempt -------------------------------------------------------------------


async def test_eligible_stale_document_creates_one_pending_job() -> None:
    session = _FakeSchedulingSession()
    target = build_embedding_config()
    document = _indexed_document("documents__ollama__old__ev0__cv0__d768")

    result = await schedule_reindex(
        session, document, FakeVectorStore(), target, target_chunk_size=500, target_chunk_overlap=50
    )

    assert result.outcome == ReindexSchedulingOutcome.CREATED
    assert result.job is not None
    assert result.job.status == ReindexJobStatus.PENDING
    assert len(session.reindex_jobs) == 1


async def test_created_job_stores_the_exact_target_collection() -> None:
    session = _FakeSchedulingSession()
    target = build_embedding_config()
    document = _indexed_document("documents__ollama__old__ev0__cv0__d768")

    result = await schedule_reindex(
        session, document, FakeVectorStore(), target, target_chunk_size=500, target_chunk_overlap=50
    )

    assert result.job is not None
    assert result.job.target_collection_name == target.collection_name


async def test_created_job_stores_target_chunk_size_and_overlap() -> None:
    session = _FakeSchedulingSession()
    target = build_embedding_config()
    document = _indexed_document("documents__ollama__old__ev0__cv0__d768")

    result = await schedule_reindex(
        session, document, FakeVectorStore(), target, target_chunk_size=777, target_chunk_overlap=88
    )

    assert result.job is not None
    assert result.job.target_chunk_size == 777
    assert result.job.target_chunk_overlap == 88


async def test_scheduling_ensures_the_target_index_collection_is_persisted() -> None:
    session = _FakeSchedulingSession()
    target = build_embedding_config()
    document = _indexed_document("documents__ollama__old__ev0__cv0__d768")

    await schedule_reindex(
        session, document, FakeVectorStore(), target, target_chunk_size=500, target_chunk_overlap=50
    )

    assert target.collection_name in session.index_collections


async def test_failed_historical_jobs_do_not_block_a_new_attempt() -> None:
    session = _FakeSchedulingSession()
    target = build_embedding_config()
    document = _indexed_document("documents__ollama__old__ev0__cv0__d768")
    session.reindex_jobs[str(uuid.uuid4())] = ReindexJob(
        id=str(uuid.uuid4()), document_id=document.id, target_collection_name="some-other-collection",
        target_chunk_size=500, target_chunk_overlap=50, status=ReindexJobStatus.FAILED,
    )

    result = await schedule_reindex(
        session, document, FakeVectorStore(), target, target_chunk_size=500, target_chunk_overlap=50
    )

    assert result.outcome == ReindexSchedulingOutcome.CREATED


async def test_completed_historical_jobs_do_not_block_a_new_attempt() -> None:
    session = _FakeSchedulingSession()
    target = build_embedding_config()
    document = _indexed_document("documents__ollama__old__ev0__cv0__d768")
    session.reindex_jobs[str(uuid.uuid4())] = ReindexJob(
        id=str(uuid.uuid4()), document_id=document.id, target_collection_name="some-other-collection",
        target_chunk_size=500, target_chunk_overlap=50, status=ReindexJobStatus.COMPLETED,
    )

    result = await schedule_reindex(
        session, document, FakeVectorStore(), target, target_chunk_size=500, target_chunk_overlap=50
    )

    assert result.outcome == ReindexSchedulingOutcome.CREATED


async def test_scheduling_does_not_modify_document_index_metadata() -> None:
    session = _FakeSchedulingSession()
    target = build_embedding_config()
    document = _indexed_document(
        "documents__ollama__old__ev0__cv0__d768",
        embedding_provider="ollama", embedding_model="old-model", embedding_dimension=768,
        embedding_version="v0", chunking_version="v0", indexed_at=None,
    )

    await schedule_reindex(
        session, document, FakeVectorStore(), target, target_chunk_size=500, target_chunk_overlap=50
    )

    assert document.collection_name == "documents__ollama__old__ev0__cv0__d768"
    assert document.embedding_provider == "ollama"
    assert document.embedding_model == "old-model"
    assert document.embedding_dimension == 768
    assert document.embedding_version == "v0"
    assert document.chunking_version == "v0"
    assert document.indexed_at is None


async def test_scheduling_never_touches_object_storage() -> None:
    source = inspect.getsource(schedule_reindex)
    assert "FileStorage" not in source
    assert "file_storage" not in source
    assert "storage" not in inspect.signature(schedule_reindex).parameters


async def test_scheduling_never_writes_qdrant_vectors() -> None:
    import app.services.indexing.reindex_scheduling_service as reindex_scheduling_service_module

    source = inspect.getsource(reindex_scheduling_service_module)
    assert "upsert_vectors" not in source


# --- concurrent scheduling race -------------------------------------------------------------------


async def test_active_job_constraint_violation_reloads_the_winner_as_already_active() -> None:
    session = _FakeSchedulingSession()
    target = build_embedding_config()
    document = _indexed_document("documents__ollama__old__ev0__cv0__d768")
    winner = ReindexJob(
        id=str(uuid.uuid4()), document_id=document.id, target_collection_name=target.collection_name,
        target_chunk_size=500, target_chunk_overlap=50, status=ReindexJobStatus.PENDING,
    )
    session.pretend_collection_already_tracked = True
    session.force_next_commit_integrity_error = _ONE_ACTIVE_REINDEX_JOB_CONSTRAINT
    session.concurrent_winner_job = winner

    result = await schedule_reindex(
        session, document, FakeVectorStore(), target, target_chunk_size=500, target_chunk_overlap=50
    )

    assert result.outcome == ReindexSchedulingOutcome.ALREADY_ACTIVE  # test 2
    assert result.job is not None
    assert result.job.id == winner.id
    assert session.rollback_count == 1  # test 3: rollback happened


async def test_unrelated_integrity_error_is_reraised() -> None:
    session = _FakeSchedulingSession()
    target = build_embedding_config()
    document = _indexed_document("documents__ollama__old__ev0__cv0__d768")
    session.pretend_collection_already_tracked = True
    session.force_next_commit_integrity_error = _UNRELATED_CONSTRAINT

    with pytest.raises(IntegrityError):
        await schedule_reindex(
            session, document, FakeVectorStore(), target, target_chunk_size=500, target_chunk_overlap=50
        )

    assert session.rollback_count == 1


async def test_missing_winner_after_active_job_violation_raises_consistency_error() -> None:
    session = _FakeSchedulingSession()
    target = build_embedding_config()
    document = _indexed_document("documents__ollama__old__ev0__cv0__d768")
    session.pretend_collection_already_tracked = True
    session.force_next_commit_integrity_error = _ONE_ACTIVE_REINDEX_JOB_CONSTRAINT
    # Deliberately no concurrent_winner_job set — nothing to reload.

    with pytest.raises(MissingActiveReindexJobAfterRaceError):
        await schedule_reindex(
            session, document, FakeVectorStore(), target, target_chunk_size=500, target_chunk_overlap=50
        )
