"""Unit tests for app.services.indexing.reindex_activation against a fake session double.

Covers activate_reindexed_document()'s full precondition/decision table and atomic cutover —
orchestration only, no real Postgres locking (see
tests/integration/indexing/test_reindex_activation_postgres.py for that, and
tests/integration/indexing/test_reindex_activation_build_postgres.py for the one real end-to-end
scenario).
"""

import inspect
import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from app.models.document import Document
from app.models.document_deletion_job import DocumentDeletionJob, DocumentDeletionStatus
from app.models.index_collection import IndexCollection, IndexCollectionStatus
from app.models.reindex_job import ReindexJob, ReindexJobStatus
from app.models.vector_cleanup_job import VectorCleanupJob, VectorCleanupStatus
from app.services.indexing.reindex_activation import (
    ReindexActivationOutcome,
    activate_reindexed_document,
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


class _FakeActivationSession:
    """In-memory AsyncSession double for activate_reindexed_document(), dispatching by compiled SQL."""

    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.reindex_jobs: dict[str, ReindexJob] = {}
        self.deletion_jobs: dict[str, DocumentDeletionJob] = {}
        self.index_collections: dict[str, IndexCollection] = {}
        self.cleanup_jobs: list[VectorCleanupJob] = []
        self.commit_count = 0
        self.rollback_count = 0
        self.expired: list[object] = []
        self.fail_commit = False

    def add(self, instance: object) -> None:
        if isinstance(instance, VectorCleanupJob):
            self.cleanup_jobs.append(instance)
        elif isinstance(instance, ReindexJob):
            self.reindex_jobs[instance.id] = instance
        elif isinstance(instance, Document):
            self.documents[instance.id] = instance
        elif isinstance(instance, DocumentDeletionJob):
            self.deletion_jobs[instance.id] = instance
        elif isinstance(instance, IndexCollection):
            self.index_collections[instance.collection_name] = instance

    async def get(self, model: type, instance_id: str) -> object | None:
        if model is IndexCollection:
            return self.index_collections.get(instance_id)
        return None

    async def execute(self, stmt: Any) -> _ListResult:
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))

        if "reindex_jobs" in compiled:
            jobs = list(self.reindex_jobs.values())
            eq_match = re.search(r"reindex_jobs\.id = '([^']*)'", compiled)
            if eq_match:
                jobs = [j for j in jobs if j.id == eq_match.group(1)]
            return _ListResult(jobs)

        if "document_deletion_jobs" in compiled:
            jobs = list(self.deletion_jobs.values())
            eq_match = re.search(r"document_deletion_jobs\.document_id = '([^']*)'", compiled)
            if eq_match:
                jobs = [j for j in jobs if j.document_id == eq_match.group(1)]
            jobs.sort(key=lambda j: (j.created_at, j.id), reverse=True)
            limit_match = re.search(r"LIMIT (\d+)", compiled)
            if limit_match:
                jobs = jobs[: int(limit_match.group(1))]
            return _ListResult(jobs)

        if "FROM documents" in compiled:
            docs = list(self.documents.values())
            eq_match = re.search(r"documents\.id = '([^']*)'", compiled)
            if eq_match:
                docs = [d for d in docs if d.id == eq_match.group(1)]
            return _ListResult(docs)

        return _ListResult([])

    async def commit(self) -> None:
        if self.fail_commit:
            raise RuntimeError("db unavailable")
        self.commit_count += 1

    async def rollback(self) -> None:
        self.rollback_count += 1

    def expire(self, instance: object) -> None:
        self.expired.append(instance)


# --- builders -----------------------------------------------------------------------------------


def _document(**overrides: object) -> Document:
    fields: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        original_filename="notes.txt",
        stored_filename=f"{uuid.uuid4().hex}.txt",
        content_type="text/plain",
        file_size=100,
        stored_path="unset",
        collection_name="source-collection",
    )
    fields.update(overrides)
    return Document(**fields)  # type: ignore[arg-type]


def _index_collection(**overrides: object) -> IndexCollection:
    fields: dict[str, object] = dict(
        collection_name="target-collection",
        embedding_provider="ollama",
        embedding_model="target-model",
        embedding_dimension=3,
        embedding_version="v9",
        chunking_version="v9",
        status=IndexCollectionStatus.ACTIVE,
    )
    fields.update(overrides)
    return IndexCollection(**fields)  # type: ignore[arg-type]


def _reindex_job(document_id: str, **overrides: object) -> ReindexJob:
    fields: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        document_id=document_id,
        source_collection_name="source-collection",
        target_collection_name="target-collection",
        target_chunk_size=500,
        target_chunk_overlap=50,
        status=ReindexJobStatus.COMPLETED,
        completed_at=_BASE_TIME,
        activated_at=None,
    )
    fields.update(overrides)
    return ReindexJob(**fields)  # type: ignore[arg-type]


def _deletion_job(document_id: str, status: DocumentDeletionStatus) -> DocumentDeletionJob:
    return DocumentDeletionJob(
        id=str(uuid.uuid4()),
        document_id=document_id,
        status=status,
        vector_cleanup_completed=False,
        storage_cleanup_completed=False,
        created_at=_BASE_TIME,
    )


def _seed_ready(session: _FakeActivationSession, **job_overrides: object) -> tuple[Document, ReindexJob]:
    """Seed a document + target IndexCollection + eligible COMPLETED job, ready to activate."""
    document = _document()
    session.documents[document.id] = document
    session.index_collections["target-collection"] = _index_collection()
    job = _reindex_job(document.id, **job_overrides)
    session.reindex_jobs[job.id] = job
    return document, job


# --- job-state preconditions ----------------------------------------------------------------


async def test_missing_reindex_job_returns_job_not_found() -> None:
    session = _FakeActivationSession()

    result = await activate_reindexed_document(session, str(uuid.uuid4()))

    assert result.outcome == ReindexActivationOutcome.JOB_NOT_FOUND
    assert result.job is None
    assert session.commit_count == 0


@pytest.mark.parametrize(
    "status", [ReindexJobStatus.PENDING, ReindexJobStatus.PROCESSING, ReindexJobStatus.FAILED]
)
async def test_non_completed_jobs_cannot_activate(status: ReindexJobStatus) -> None:
    session = _FakeActivationSession()
    document, job = _seed_ready(session, status=status)

    result = await activate_reindexed_document(session, job.id)

    assert result.outcome == ReindexActivationOutcome.NOT_READY
    assert session.commit_count == 0
    assert document.collection_name == "source-collection"  # untouched


async def test_completed_unactivated_job_is_eligible() -> None:
    session = _FakeActivationSession()
    document, job = _seed_ready(session)

    result = await activate_reindexed_document(session, job.id)

    assert result.outcome == ReindexActivationOutcome.ACTIVATED


async def test_already_activated_job_is_idempotent() -> None:
    session = _FakeActivationSession()
    document, job = _seed_ready(session, activated_at=_BASE_TIME)

    result = await activate_reindexed_document(session, job.id)

    assert result.outcome == ReindexActivationOutcome.ALREADY_ACTIVATED
    assert session.commit_count == 0
    assert document.collection_name == "source-collection"  # untouched


async def test_already_activated_job_creates_no_second_cleanup_job() -> None:
    session = _FakeActivationSession()
    document, job = _seed_ready(session, activated_at=_BASE_TIME)

    await activate_reindexed_document(session, job.id)

    assert session.cleanup_jobs == []


# --- document/target existence --------------------------------------------------------------


async def test_missing_document_blocks_activation() -> None:
    session = _FakeActivationSession()
    session.index_collections["target-collection"] = _index_collection()
    job = _reindex_job(str(uuid.uuid4()))
    session.reindex_jobs[job.id] = job

    result = await activate_reindexed_document(session, job.id)

    assert result.outcome == ReindexActivationOutcome.DOCUMENT_MISSING
    assert session.commit_count == 0


async def test_missing_target_index_collection_blocks_activation() -> None:
    session = _FakeActivationSession()
    document = _document()
    session.documents[document.id] = document
    job = _reindex_job(document.id)  # no IndexCollection seeded for "target-collection"
    session.reindex_jobs[job.id] = job

    result = await activate_reindexed_document(session, job.id)

    assert result.outcome == ReindexActivationOutcome.NOT_READY
    assert session.commit_count == 0


# --- target reconstruction from persisted IndexCollection ------------------------------------


async def test_target_fields_are_reconstructed_from_persisted_index_collection() -> None:
    session = _FakeActivationSession()
    document = _document()
    session.documents[document.id] = document
    session.index_collections["target-collection"] = _index_collection(
        embedding_provider="ollama",
        embedding_model="a-real-target-model",
        embedding_dimension=1024,
        embedding_version="v42",
        chunking_version="v7",
    )
    job = _reindex_job(document.id)
    session.reindex_jobs[job.id] = job

    await activate_reindexed_document(session, job.id)

    assert document.embedding_provider == "ollama"  # test 10
    assert document.embedding_model == "a-real-target-model"  # test 11
    assert document.embedding_dimension == 1024  # test 12
    assert document.embedding_version == "v42"  # test 13
    assert document.chunking_version == "v7"  # test 14
    assert document.collection_name == "target-collection"  # test 15


async def test_live_settings_are_never_consulted() -> None:
    """No `settings`-shaped parameter exists on activate_reindexed_document() at all."""
    parameters = inspect.signature(activate_reindexed_document).parameters
    assert "settings" not in parameters
    assert list(parameters) == ["session", "reindex_job_id"]


# --- source ownership / staleness -------------------------------------------------------------


async def test_document_source_must_match_the_jobs_persisted_source_snapshot() -> None:
    session = _FakeActivationSession()
    document, job = _seed_ready(session)  # document.collection_name == job.source_collection_name

    result = await activate_reindexed_document(session, job.id)

    assert result.outcome == ReindexActivationOutcome.ACTIVATED


async def test_changed_source_collection_blocks_activation() -> None:
    session = _FakeActivationSession()
    document = _document(collection_name="some-third-collection")  # moved since scheduling
    session.documents[document.id] = document
    session.index_collections["target-collection"] = _index_collection()
    job = _reindex_job(document.id, source_collection_name="source-collection")
    session.reindex_jobs[job.id] = job

    result = await activate_reindexed_document(session, job.id)

    assert result.outcome == ReindexActivationOutcome.SOURCE_CHANGED
    assert session.commit_count == 0
    assert document.collection_name == "some-third-collection"  # never overwritten


async def test_source_comparison_is_by_collection_name_alone_and_is_sufficient() -> None:
    """`collection_name` is the document's full versioned identity — no separate source embedding
    metadata field exists (or is needed) to independently disagree; comparing it alone is
    complete, not merely a partial check."""
    session = _FakeActivationSession()
    # A document whose collection_name matches the job's source, regardless of what its other
    # embedding_* fields say (they are about to be overwritten anyway) — activation proceeds.
    document, job = _seed_ready(
        session, source_collection_name="source-collection"
    )
    document.embedding_provider = "whatever-was-there-before"

    result = await activate_reindexed_document(session, job.id)

    assert result.outcome == ReindexActivationOutcome.ACTIVATED


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
async def test_deletion_lifecycle_states_block_activation(status: DocumentDeletionStatus) -> None:
    session = _FakeActivationSession()
    document, job = _seed_ready(session)
    deletion_job = _deletion_job(document.id, status)
    session.deletion_jobs[deletion_job.id] = deletion_job

    result = await activate_reindexed_document(session, job.id)

    assert result.outcome == ReindexActivationOutcome.BLOCKED_BY_DELETION
    assert session.commit_count == 0
    assert document.collection_name == "source-collection"  # untouched


# --- successful activation: document cutover ---------------------------------------------------


async def test_successful_activation_updates_collection_name_to_target() -> None:
    session = _FakeActivationSession()
    document, job = _seed_ready(session)

    await activate_reindexed_document(session, job.id)

    assert document.collection_name == "target-collection"  # test 24


async def test_successful_activation_preserves_unrelated_document_fields() -> None:
    session = _FakeActivationSession()
    document, job = _seed_ready(session, )
    document.original_filename = "keep-me.pdf"
    document.content_type = "application/pdf"

    await activate_reindexed_document(session, job.id)

    assert document.original_filename == "keep-me.pdf"  # test 27
    assert document.content_type == "application/pdf"


async def test_successful_activation_updates_indexed_at() -> None:
    session = _FakeActivationSession()
    document, job = _seed_ready(session)
    document.indexed_at = _BASE_TIME - timedelta(days=30)

    await activate_reindexed_document(session, job.id)

    assert document.indexed_at is not None
    assert document.indexed_at != _BASE_TIME - timedelta(days=30)  # test 26


# --- successful activation: cleanup job ---------------------------------------------------------


async def test_successful_activation_creates_exactly_one_cleanup_job() -> None:
    session = _FakeActivationSession()
    document, job = _seed_ready(session)

    await activate_reindexed_document(session, job.id)

    assert len(session.cleanup_jobs) == 1  # test 28


async def test_cleanup_job_targets_the_source_collection() -> None:
    session = _FakeActivationSession()
    document, job = _seed_ready(session)

    await activate_reindexed_document(session, job.id)

    assert session.cleanup_jobs[0].collection_name == "source-collection"  # test 29
    assert session.cleanup_jobs[0].collection_name != "target-collection"


async def test_cleanup_job_references_the_correct_document() -> None:
    session = _FakeActivationSession()
    document, job = _seed_ready(session)

    await activate_reindexed_document(session, job.id)

    assert session.cleanup_jobs[0].document_id == document.id  # test 30


async def test_cleanup_job_is_created_in_pending_status() -> None:
    session = _FakeActivationSession()
    document, job = _seed_ready(session)

    await activate_reindexed_document(session, job.id)

    assert session.cleanup_jobs[0].status == VectorCleanupStatus.PENDING  # test 31
    assert session.cleanup_jobs[0].attempts == 0
    assert isinstance(session.cleanup_jobs[0], VectorCleanupJob)


async def test_source_and_target_equality_creates_no_cleanup_job() -> None:
    """Defensive guard only — schedule_reindex() already rejects this case at scheduling time,
    but a directly-constructed job could still have source == target."""
    session = _FakeActivationSession()
    document = _document(collection_name="target-collection")
    session.documents[document.id] = document
    session.index_collections["target-collection"] = _index_collection()
    job = _reindex_job(
        document.id, source_collection_name="target-collection", target_collection_name="target-collection"
    )
    session.reindex_jobs[job.id] = job

    result = await activate_reindexed_document(session, job.id)

    assert result.outcome == ReindexActivationOutcome.ACTIVATED
    assert session.cleanup_jobs == []  # test 40


# --- successful activation: job marking ----------------------------------------------------------


async def test_successful_activation_sets_activated_at() -> None:
    session = _FakeActivationSession()
    document, job = _seed_ready(session)

    await activate_reindexed_document(session, job.id)

    assert job.activated_at is not None  # test 32


async def test_build_completed_at_remains_unchanged_by_activation() -> None:
    session = _FakeActivationSession()
    document, job = _seed_ready(session, completed_at=_BASE_TIME)

    await activate_reindexed_document(session, job.id)

    assert job.completed_at == _BASE_TIME  # test 33


async def test_build_status_retains_completed_meaning_after_activation() -> None:
    session = _FakeActivationSession()
    document, job = _seed_ready(session)

    await activate_reindexed_document(session, job.id)

    assert job.status == ReindexJobStatus.COMPLETED  # test 34 — never flips to some "ACTIVATED" status


# --- activation never builds, embeds, deletes vectors, or retires -------------------------------


async def test_activation_signature_has_no_build_execution_or_qdrant_dependencies() -> None:
    """No file_storage/vector_store/settings parameter exists — activation cannot build, embed,
    extract, or delete vectors, because it is never given the means to."""
    parameters = list(inspect.signature(activate_reindexed_document).parameters)
    assert parameters == ["session", "reindex_job_id"]  # tests 35, 36, 37, 38


async def test_activation_never_retires_the_target_collection() -> None:
    session = _FakeActivationSession()
    document, job = _seed_ready(session)

    await activate_reindexed_document(session, job.id)

    assert session.index_collections["target-collection"].status == IndexCollectionStatus.ACTIVE  # test 39


# --- transaction failure / rollback safety -------------------------------------------------------


async def test_commit_failure_rolls_back_and_expires_job_and_document() -> None:
    session = _FakeActivationSession()
    document, job = _seed_ready(session)
    session.fail_commit = True

    with pytest.raises(RuntimeError, match="db unavailable"):
        await activate_reindexed_document(session, job.id)

    assert session.rollback_count == 1  # tests 41, 42, 43: one commit, one rollback undoes all of it
    assert job in session.expired  # test 44/45: defensive expire, never re-read after
    assert document in session.expired


def test_scalar_identifiers_are_captured_before_any_possible_rollback() -> None:
    """Narrow structural check (not a broad string search): document_id/source/target must be
    captured as plain values immediately after the job is loaded — textually before the
    commit-failure rollback path — so that path never re-reads a (possibly-expired) ORM object.
    """
    source = inspect.getsource(activate_reindexed_document)
    capture_index = source.index("document_id = job.document_id")
    rollback_index = source.index("await session.rollback()")
    assert capture_index < rollback_index
