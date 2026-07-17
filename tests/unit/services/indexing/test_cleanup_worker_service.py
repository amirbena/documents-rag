"""Unit tests for process_next_vector_cleanup_job()/retry_cleanup_job()'s active-serving-collection
guard, against a fake session double (Phase 2.8.6, subtask 7).

Covers cleanup orchestration only: claiming, delegation, safety-guard blocking, and rollback
behavior. Does not duplicate the complete vector-deletion unit matrix already covered in Phase
2.8.1 (tests/unit/services/indexing/test_vector_deletion_service.py) or the existing
create/list/retry-given-a-job coverage in test_cleanup_job_service.py. Real PostgreSQL
locking/concurrency behavior is covered separately by
tests/integration/indexing/test_cleanup_worker_postgres.py.
"""

import inspect
import re
import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from app.models.document import Document
from app.models.vector_cleanup_job import VectorCleanupJob, VectorCleanupStatus
from app.services.indexing.cleanup_job_service import (
    _ACTIVE_COLLECTION_GUARD_MESSAGE,
    _MAX_ERROR_MESSAGE_LENGTH,
    VectorCleanupWorkerOutcome,
    process_next_vector_cleanup_job,
    retry_cleanup_job,
)
from tests.support.indexing.builders import build_document
from tests.support.indexing.fakes import FakeVectorStore

_BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


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

    def scalar_one_or_none(self) -> Any | None:
        return self._items[0] if self._items else None


class _FakeCleanupSession:
    """In-memory AsyncSession double for process_next_vector_cleanup_job()/retry_cleanup_job()."""

    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.cleanup_jobs: dict[str, VectorCleanupJob] = {}
        self.commit_count = 0
        self.rollback_count = 0
        self.expired: list[object] = []
        self.fail_next_commit = False

    def add(self, instance: object) -> None:
        if isinstance(instance, VectorCleanupJob):
            self.cleanup_jobs[instance.id] = instance
        elif isinstance(instance, Document):
            self.documents[instance.id] = instance

    async def get(self, model: type, instance_id: str) -> object | None:
        if model is Document:
            return self.documents.get(instance_id)
        if model is VectorCleanupJob:
            return self.cleanup_jobs.get(instance_id)
        return None

    async def execute(self, stmt: Any) -> _ListResult:
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))

        if "vector_cleanup_jobs" in compiled:
            jobs = list(self.cleanup_jobs.values())
            in_match = re.search(r"vector_cleanup_jobs\.status IN \(([^)]*)\)", compiled)
            if in_match:
                statuses = {token.strip().strip("'") for token in in_match.group(1).split(",")}
                jobs = [job for job in jobs if job.status.value in statuses]
            jobs.sort(key=lambda job: (job.created_at, job.id))
            limit_match = re.search(r"LIMIT (\d+)", compiled)
            if limit_match:
                jobs = jobs[: int(limit_match.group(1))]
            return _ListResult(jobs)

        return _ListResult([])

    async def commit(self) -> None:
        if self.fail_next_commit:
            self.fail_next_commit = False
            raise RuntimeError("db unavailable")
        self.commit_count += 1

    async def rollback(self) -> None:
        self.rollback_count += 1

    def expire(self, instance: object) -> None:
        self.expired.append(instance)


def _document(**overrides: object) -> Document:
    return build_document(**overrides)


def _cleanup_job(document_id: str, **overrides: object) -> VectorCleanupJob:
    fields: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        document_id=document_id,
        collection_name="old-collection",
        status=VectorCleanupStatus.PENDING,
        attempts=0,
        last_error=None,
        created_at=_BASE_TIME,
        completed_at=None,
    )
    fields.update(overrides)
    return VectorCleanupJob(**fields)  # type: ignore[arg-type]


# --- claiming / no-op -------------------------------------------------------------------------


async def test_no_eligible_job_returns_no_job() -> None:
    session = _FakeCleanupSession()
    vector_store = FakeVectorStore()

    result = await process_next_vector_cleanup_job(session, vector_store)

    assert result.outcome == VectorCleanupWorkerOutcome.NO_JOB
    assert result.job_id is None


async def test_at_most_one_cleanup_job_is_processed_per_call() -> None:
    session = _FakeCleanupSession()
    document = _document()
    session.documents[document.id] = document
    older_job = _cleanup_job(document.id, collection_name="collection-a", created_at=_BASE_TIME)
    newer_job = _cleanup_job(
        document.id, collection_name="collection-b", created_at=datetime(2026, 1, 2, tzinfo=UTC)
    )
    session.cleanup_jobs[older_job.id] = older_job
    session.cleanup_jobs[newer_job.id] = newer_job
    vector_store = FakeVectorStore()

    result = await process_next_vector_cleanup_job(session, vector_store)

    assert result.outcome == VectorCleanupWorkerOutcome.COMPLETED
    assert result.job_id == older_job.id  # oldest first
    assert len(vector_store.deleted) == 1
    assert newer_job.status == VectorCleanupStatus.PENDING  # untouched


# --- persisted-record authority ------------------------------------------------------------------


async def test_cleanup_uses_the_persisted_collection_name() -> None:
    session = _FakeCleanupSession()
    document = _document(collection_name="some-other-collection")
    session.documents[document.id] = document
    job = _cleanup_job(document.id, collection_name="old-collection")
    session.cleanup_jobs[job.id] = job
    vector_store = FakeVectorStore()

    await process_next_vector_cleanup_job(session, vector_store)

    assert vector_store.deleted == [("old-collection", document.id)]


async def test_cleanup_uses_the_persisted_document_id() -> None:
    session = _FakeCleanupSession()
    document = _document()
    session.documents[document.id] = document
    job = _cleanup_job(document.id)
    session.cleanup_jobs[job.id] = job
    vector_store = FakeVectorStore()

    await process_next_vector_cleanup_job(session, vector_store)

    assert vector_store.deleted[0][1] == document.id


async def test_cleanup_does_not_derive_collection_from_settings() -> None:
    """retry_cleanup_job()/process_next_vector_cleanup_job() take no Settings parameter at all."""
    assert "settings" not in inspect.signature(retry_cleanup_job).parameters
    assert "settings" not in inspect.signature(process_next_vector_cleanup_job).parameters


async def test_cleanup_delegates_exactly_once_to_the_deletion_primitive() -> None:
    session = _FakeCleanupSession()
    document = _document()
    session.documents[document.id] = document
    job = _cleanup_job(document.id)
    session.cleanup_jobs[job.id] = job
    vector_store = FakeVectorStore()

    await process_next_vector_cleanup_job(session, vector_store)

    assert len(vector_store.deleted) == 1


# --- successful cleanup -----------------------------------------------------------------------


async def test_successful_cleanup_marks_the_record_successful() -> None:
    session = _FakeCleanupSession()
    document = _document()
    session.documents[document.id] = document
    job = _cleanup_job(document.id)
    session.cleanup_jobs[job.id] = job
    vector_store = FakeVectorStore()

    result = await process_next_vector_cleanup_job(session, vector_store)

    assert result.outcome == VectorCleanupWorkerOutcome.COMPLETED
    assert job.status == VectorCleanupStatus.COMPLETED
    assert job.completed_at is not None


async def test_successful_cleanup_clears_the_error_message() -> None:
    session = _FakeCleanupSession()
    document = _document()
    session.documents[document.id] = document
    job = _cleanup_job(document.id, status=VectorCleanupStatus.FAILED, last_error="a prior failure")
    session.cleanup_jobs[job.id] = job
    vector_store = FakeVectorStore()

    await process_next_vector_cleanup_job(session, vector_store)

    assert job.last_error is None


async def test_already_missing_vectors_are_treated_as_successful() -> None:
    """FakeVectorStore's delete is unconditionally successful, whether or not vectors existed —
    matching the real Qdrant delete-by-filter's idempotent-no-op-on-empty-match semantics."""
    session = _FakeCleanupSession()
    document = _document()
    session.documents[document.id] = document
    job = _cleanup_job(document.id)
    session.cleanup_jobs[job.id] = job
    vector_store = FakeVectorStore()  # never configured to fail — "already absent" is a success

    result = await process_next_vector_cleanup_job(session, vector_store)

    assert result.outcome == VectorCleanupWorkerOutcome.COMPLETED


# --- failure handling --------------------------------------------------------------------------


async def test_qdrant_failure_persists_the_failed_state() -> None:
    session = _FakeCleanupSession()
    document = _document()
    session.documents[document.id] = document
    job = _cleanup_job(document.id, collection_name="old-collection")
    session.cleanup_jobs[job.id] = job
    vector_store = FakeVectorStore(fail_delete_for={"old-collection"})

    result = await process_next_vector_cleanup_job(session, vector_store)

    assert result.outcome == VectorCleanupWorkerOutcome.FAILED
    assert job.status == VectorCleanupStatus.FAILED
    assert job.attempts == 1


async def test_failure_stores_a_bounded_error_message() -> None:
    session = _FakeCleanupSession()
    document = _document()
    session.documents[document.id] = document
    job = _cleanup_job(document.id, collection_name="old-collection")
    session.cleanup_jobs[job.id] = job

    class _VerboseFailingVectorStore(FakeVectorStore):
        async def delete_by_document_id(self, collection_name: str, document_id: str) -> None:
            raise RuntimeError("x" * (_MAX_ERROR_MESSAGE_LENGTH * 5))

    await process_next_vector_cleanup_job(session, _VerboseFailingVectorStore())

    assert job.last_error is not None
    assert len(job.last_error) <= _MAX_ERROR_MESSAGE_LENGTH


async def test_failure_does_not_create_another_cleanup_job() -> None:
    session = _FakeCleanupSession()
    document = _document()
    session.documents[document.id] = document
    job = _cleanup_job(document.id, collection_name="old-collection")
    session.cleanup_jobs[job.id] = job
    vector_store = FakeVectorStore(fail_delete_for={"old-collection"})

    await process_next_vector_cleanup_job(session, vector_store)

    assert len(session.cleanup_jobs) == 1  # still only the original job


# --- document/re-index metadata never mutated ----------------------------------------------------


async def test_cleanup_does_not_modify_document_serving_metadata() -> None:
    session = _FakeCleanupSession()
    document = _document(
        collection_name="current-collection",
        embedding_provider="ollama",
        embedding_model="current-model",
    )
    session.documents[document.id] = document
    job = _cleanup_job(document.id, collection_name="old-collection")
    session.cleanup_jobs[job.id] = job
    vector_store = FakeVectorStore()

    await process_next_vector_cleanup_job(session, vector_store)

    assert document.collection_name == "current-collection"
    assert document.embedding_provider == "ollama"
    assert document.embedding_model == "current-model"


async def test_cleanup_does_not_modify_reindex_build_or_activation_fields() -> None:
    """Structural: the cleanup module never imports ReindexJob at all."""
    import app.services.indexing.cleanup_job_service as cleanup_module

    assert not hasattr(cleanup_module, "ReindexJob")


# --- active-serving-collection guard -------------------------------------------------------------


async def test_cleanup_never_targets_the_active_serving_collection() -> None:
    session = _FakeCleanupSession()
    document = _document(collection_name="still-serving-collection")
    session.documents[document.id] = document
    job = _cleanup_job(document.id, collection_name="still-serving-collection")
    session.cleanup_jobs[job.id] = job
    vector_store = FakeVectorStore()

    result = await process_next_vector_cleanup_job(session, vector_store)

    assert result.outcome == VectorCleanupWorkerOutcome.FAILED
    assert vector_store.deleted == []  # no delete call was ever made


async def test_active_collection_record_is_blocked_safely() -> None:
    session = _FakeCleanupSession()
    document = _document(collection_name="still-serving-collection")
    session.documents[document.id] = document
    job = _cleanup_job(document.id, collection_name="still-serving-collection")
    session.cleanup_jobs[job.id] = job
    vector_store = FakeVectorStore()

    await process_next_vector_cleanup_job(session, vector_store)

    assert job.status == VectorCleanupStatus.FAILED
    assert job.last_error == _ACTIVE_COLLECTION_GUARD_MESSAGE
    assert document.collection_name == "still-serving-collection"  # document itself untouched


async def test_missing_document_never_blocks_cleanup() -> None:
    """A document already fully deleted must never block its own historical cleanup."""
    session = _FakeCleanupSession()
    job = _cleanup_job(str(uuid.uuid4()), collection_name="old-collection")
    session.cleanup_jobs[job.id] = job
    vector_store = FakeVectorStore()

    result = await process_next_vector_cleanup_job(session, vector_store)

    assert result.outcome == VectorCleanupWorkerOutcome.COMPLETED


# --- collection/deletion boundary ----------------------------------------------------------------


async def test_cleanup_does_not_delete_the_entire_collection() -> None:
    """FakeVectorStore exposes only a per-document delete — no collection-wide delete exists to call."""
    assert not hasattr(FakeVectorStore(), "delete_collection")
    assert not hasattr(FakeVectorStore(), "clear_collection")


async def test_cleanup_does_not_invoke_full_document_deletion() -> None:
    import app.services.indexing.cleanup_job_service as cleanup_module

    assert not hasattr(cleanup_module, "delete_all_tracked_document_vectors")
    assert not hasattr(cleanup_module, "DocumentDeletionWorker")


async def test_cleanup_does_not_access_object_storage() -> None:
    assert "file_storage" not in inspect.signature(process_next_vector_cleanup_job).parameters
    assert "file_storage" not in inspect.signature(retry_cleanup_job).parameters
    import app.services.indexing.cleanup_job_service as cleanup_module

    assert not hasattr(cleanup_module, "FileStorage")


# --- rollback safety ---------------------------------------------------------------------------


async def test_rollback_handling_uses_captured_scalar_identifiers() -> None:
    session = _FakeCleanupSession()
    document = _document()
    session.documents[document.id] = document
    job = _cleanup_job(document.id)
    session.cleanup_jobs[job.id] = job
    vector_store = FakeVectorStore()
    session.fail_next_commit = True  # fails the claim commit, before any Qdrant call

    with pytest.raises(RuntimeError, match="db unavailable"):
        await process_next_vector_cleanup_job(session, vector_store)

    assert session.rollback_count == 0  # claim commit failure has nothing to roll back yet
    assert vector_store.deleted == []  # never reached the delete call


async def test_terminal_commit_failure_rolls_back_and_expires_the_job() -> None:
    session = _FakeCleanupSession()
    document = _document()
    session.documents[document.id] = document
    job = _cleanup_job(document.id)
    session.cleanup_jobs[job.id] = job
    vector_store = FakeVectorStore()

    original_commit = session.commit
    calls = {"n": 0}

    async def _commit_fail_on_second_call() -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("db unavailable")
        await original_commit()

    session.commit = _commit_fail_on_second_call  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="db unavailable"):
        await process_next_vector_cleanup_job(session, vector_store)

    assert session.rollback_count == 1
    assert job in session.expired
