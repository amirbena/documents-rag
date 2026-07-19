"""Scheduling decisions for re-index build attempts (Phase 2.8.6, subtask 2).

`schedule_reindex()` decides whether a build (`app.services.indexing.reindex_service
.build_reindex_target()`) may be scheduled for a document against an explicit, caller-supplied
target configuration — it never derives a target from global settings itself, and never performs
a build, activation, Qdrant vector write, or object-storage read. Re-index attempts are append-only
`ReindexJob` rows, exactly like `IngestionJob`/`DocumentDeletionJob`: a `FAILED` row is never reset
or reused, and at most one `PENDING`/`PROCESSING` ("active") row may exist per document at a time,
enforced by the partial unique index `ix_reindex_jobs_one_active_per_document`
(migration `a8685da857f3`) — not application logic alone.

No worker, script, or public API consumes this module yet — see `ReindexWorker`/activation/API in
later subtasks.

## Why this module duplicates two small lookups instead of importing them

`get_latest_ingestion_job` (`app.services.documents.query_service`) and `get_latest_deletion_job`
(`app.services.documents.deletion_service`) already exist and do exactly what this module needs.
They are deliberately *not* imported here: `app/services/documents/deletion_service.py` must call
into this module (to block deletion scheduling while a re-index is active — see
`get_active_reindex_job()` below), so this module importing back from either `documents/` module
would create a real import cycle, not just a style violation. `deletion_service.py`'s own
`_latest_active_ingestion_job()` sets this exact precedent already, for the same reason.
"""

import uuid
from dataclasses import dataclass
from enum import StrEnum

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_deletion_job import DocumentDeletionJob, DocumentDeletionStatus
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.models.reindex_job import ReindexJob, ReindexJobStatus
from app.rag.embedding_config import EmbeddingIndexConfig
from app.rag.providers.vector_store import VectorStore
from app.services.indexing.collection_registry import ensure_active_collection

_ACTIVE_REINDEX_STATUSES = (ReindexJobStatus.PENDING, ReindexJobStatus.PROCESSING)
_ACTIVE_INGESTION_STATUSES = (IngestionStatus.PENDING, IngestionStatus.PROCESSING)
_ACTIVE_DELETION_STATUSES = (DocumentDeletionStatus.PENDING, DocumentDeletionStatus.PROCESSING)
_ONE_ACTIVE_REINDEX_JOB_CONSTRAINT_NAME = "ix_reindex_jobs_one_active_per_document"


class MissingActiveReindexJobAfterRaceError(Exception):
    """A one-active-job unique-constraint conflict was raised, but no active job could be reloaded.

    Should be unreachable in practice — a committed conflicting INSERT means a winning row exists —
    but if it is ever observed, this is a genuine data-consistency problem, never a cue to silently
    create a second active job for the same document.
    """

    def __init__(self, document_id: str) -> None:
        self.document_id = document_id
        super().__init__(
            f"Scheduling a re-index job for document {document_id!r} raised the "
            f"{_ONE_ACTIVE_REINDEX_JOB_CONSTRAINT_NAME!r} unique-constraint conflict, but no "
            "active job could be reloaded afterward."
        )


class ReindexSchedulingOutcome(StrEnum):
    """The decided outcome of one schedule_reindex() call."""

    INELIGIBLE_NEVER_INDEXED = "ineligible_never_indexed"
    ALREADY_CURRENT = "already_current"
    ALREADY_ACTIVE = "already_active"
    INGESTION_ACTIVE = "ingestion_active"
    DELETION_ACTIVE = "deletion_active"
    DELETION_INCOMPLETE = "deletion_incomplete"
    DOCUMENT_DELETED = "document_deleted"
    CREATED = "created"


@dataclass(frozen=True)
class ReindexSchedulingResult:
    """Typed outcome of schedule_reindex(): the outcome, the relevant job (if any), and the target.

    `job` is populated for `ALREADY_ACTIVE` (the existing active job) and `CREATED` (the new job);
    `None` for every other outcome. `target_config` is always the caller-supplied target, echoed
    back for a future API layer to report without needing to re-resolve it.
    """

    outcome: ReindexSchedulingOutcome
    document: Document
    job: ReindexJob | None
    target_config: EmbeddingIndexConfig


def _diagnostic_constraint_name(exc: IntegrityError) -> str | None:
    """Return the PostgreSQL diagnostic `constraint_name` for `exc`, if one is available.

    Mirrors `app.services.documents.dedup_service._diagnostic_constraint_name` exactly — see that
    function's docstring for why both `exc.orig` and `exc.orig.__cause__` are checked.
    """
    for candidate in (exc.orig, getattr(exc.orig, "__cause__", None)):
        constraint_name = getattr(candidate, "constraint_name", None)
        if constraint_name is not None:
            return constraint_name
    return None


def is_active_reindex_job_violation(exc: IntegrityError) -> bool:
    """Return True only if `exc` was raised specifically by `ix_reindex_jobs_one_active_per_document`.

    Inspects the underlying PostgreSQL diagnostic constraint name rather than matching on
    human-readable error message text — an unrelated integrity error (a bad foreign key, a NOT
    NULL violation) is never misclassified as this specific race.
    """
    return _diagnostic_constraint_name(exc) == _ONE_ACTIVE_REINDEX_JOB_CONSTRAINT_NAME


async def get_document(session: AsyncSession, document_id: str) -> Document | None:
    """Return the Document with `document_id`, or None if it does not exist.

    A trivial by-primary-key lookup, duplicated here (rather than imported from
    `app.services.documents.query_service.get_document`) purely so callers in `app/api/v1/routes/`
    that need a `Document` object for `schedule_reindex()` never touch `session.get()`/ORM query
    logic directly — see CLAUDE.md's "Route Layer Style" — without this module reaching into
    `app/services/documents/*` (see module docstring's dependency-cycle rationale).
    """
    return await session.get(Document, document_id)


async def get_reindex_job(session: AsyncSession, job_id: str) -> ReindexJob | None:
    """Return the ReindexJob with `job_id`, or None if it does not exist.

    Used by the re-index activation route to confirm a caller-supplied `job_id` actually belongs
    to the document named in the URL, before ever calling `reindex_activation
    .activate_reindexed_document()` — a basic resource-ownership check, not a duplication of
    activation's own precondition validation.
    """
    return await session.get(ReindexJob, job_id)


async def get_latest_reindex_job(session: AsyncSession, document_id: str) -> ReindexJob | None:
    """Return `document_id`'s most recent ReindexJob (created_at DESC, id DESC), or None."""
    stmt = (
        select(ReindexJob)
        .where(ReindexJob.document_id == document_id)
        .order_by(ReindexJob.created_at.desc(), ReindexJob.id.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalars().first()


async def get_active_reindex_job(session: AsyncSession, document_id: str) -> ReindexJob | None:
    """Return `document_id`'s active (PENDING/PROCESSING) ReindexJob, or None if it has none.

    Used both by `schedule_reindex()` itself and by
    `app.services.documents.deletion_service.request_document_deletion()`, which must block
    deletion scheduling while a re-index build is active.
    """
    stmt = (
        select(ReindexJob)
        .where(ReindexJob.document_id == document_id, ReindexJob.status.in_(_ACTIVE_REINDEX_STATUSES))
        .order_by(ReindexJob.created_at.desc(), ReindexJob.id.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalars().first()


async def get_completed_reindex_target_collections(session: AsyncSession, document_id: str) -> list[str]:
    """Return every distinct `target_collection_name` from a COMPLETED ReindexJob for `document_id`.

    Used by `app.services.indexing.vector_deletion_service.delete_all_tracked_document_vectors()`
    (Phase 2.8.6, subtask 3): a COMPLETED job is durable proof that a full target vector set may
    already exist in its target collection, even while `Document.collection_name` still points at
    the serving collection (build-ahead, not yet activated). `PENDING`/`PROCESSING`/`FAILED` jobs
    are deliberately excluded — none of them durably prove a complete vector set exists. Selects
    only the `target_collection_name` column, never full `ReindexJob` rows, since nothing else
    about a completed job is needed for this resolution step.
    """
    stmt = select(ReindexJob.target_collection_name).where(
        ReindexJob.document_id == document_id, ReindexJob.status == ReindexJobStatus.COMPLETED
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def _latest_ingestion_job(session: AsyncSession, document_id: str) -> IngestionJob | None:
    """Return `document_id`'s most recent IngestionJob — duplicated locally, see module docstring."""
    stmt = (
        select(IngestionJob)
        .where(IngestionJob.document_id == document_id)
        .order_by(IngestionJob.created_at.desc(), IngestionJob.id.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalars().first()


async def _latest_deletion_job(session: AsyncSession, document_id: str) -> DocumentDeletionJob | None:
    """Return `document_id`'s most recent DocumentDeletionJob — duplicated locally, see module docstring."""
    stmt = (
        select(DocumentDeletionJob)
        .where(DocumentDeletionJob.document_id == document_id)
        .order_by(DocumentDeletionJob.created_at.desc(), DocumentDeletionJob.id.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalars().first()


def _result(
    outcome: ReindexSchedulingOutcome,
    document: Document,
    target_config: EmbeddingIndexConfig,
    job: ReindexJob | None = None,
) -> ReindexSchedulingResult:
    return ReindexSchedulingResult(outcome=outcome, document=document, job=job, target_config=target_config)


async def schedule_reindex(
    session: AsyncSession,
    document: Document,
    vector_store: VectorStore,
    target_config: EmbeddingIndexConfig,
    *,
    target_chunk_size: int,
    target_chunk_overlap: int,
) -> ReindexSchedulingResult:
    """Decide whether a re-index build may be scheduled for `document` against `target_config`.

    Deterministic lifecycle order — the first matching condition decides the outcome:

    1. `document.collection_name IS NULL` (never successfully indexed) -> `INELIGIBLE_NEVER_INDEXED`;
       the caller should direct the operator toward the existing ingestion retry lifecycle instead —
       re-indexing is never a second initial-ingestion recovery mechanism.
    2. `document.collection_name == target_config.collection_name` -> `ALREADY_CURRENT`; no job.
    3. An active (PENDING/PROCESSING) ReindexJob already exists -> `ALREADY_ACTIVE`; the existing
       job is returned, never duplicated.
    4. The latest IngestionJob is PENDING/PROCESSING -> `INGESTION_ACTIVE`; a failed or completed
       ingestion attempt never blocks on its own, since `document.collection_name` already being
       non-null (step 1) proves a valid prior index exists.
    5. The latest DocumentDeletionJob is PENDING/PROCESSING -> `DELETION_ACTIVE`;
       PARTIALLY_FAILED -> `DELETION_INCOMPLETE`; COMPLETED -> `DOCUMENT_DELETED`.
    6. Otherwise: `ensure_active_collection()` persists the target `IndexCollection` row (the
       `ReindexJob.target_collection_name` foreign key requires it to already exist), then one
       `PENDING` `ReindexJob` is inserted with the full pinned target-chunk snapshot and committed
       -> `CREATED`.

    Two sessions may both pass step 3's active-job lookup before either commits — the partial
    unique index is the actual concurrency guarantee. The losing insert's `IntegrityError` is
    classified via the PostgreSQL diagnostic `constraint_name` (never message-text matching) to
    confirm it is specifically `ix_reindex_jobs_one_active_per_document` before being treated as
    this race; any other integrity error is re-raised unchanged. On a confirmed race, the loser
    rolls back and reloads the winning active job, returning `ALREADY_ACTIVE` — or raises
    `MissingActiveReindexJobAfterRaceError` in the (should-be-unreachable) case where no winner can
    be reloaded.
    """
    # Captured once, up front: a rollback later in this function (the race-recovery path) expires
    # every object in the session, including `document` — any attribute access on it afterward
    # would trigger an unsafe synchronous lazy-load under an AsyncSession. Every subsequent lookup
    # in this function uses these plain values, never `document.*` again. `source_collection_name`
    # is the document's serving collection *at scheduling time* — pinned onto the job so
    # `reindex_activation.activate_reindexed_document()` (subtask 5) can later detect whether the
    # document has since moved to a third collection before ever overwriting it.
    document_id = document.id

    if document.collection_name is None:
        return _result(ReindexSchedulingOutcome.INELIGIBLE_NEVER_INDEXED, document, target_config)

    source_collection_name = document.collection_name

    if document.collection_name == target_config.collection_name:
        return _result(ReindexSchedulingOutcome.ALREADY_CURRENT, document, target_config)

    active_reindex = await get_active_reindex_job(session, document_id)
    if active_reindex is not None:
        return _result(ReindexSchedulingOutcome.ALREADY_ACTIVE, document, target_config, active_reindex)

    latest_ingestion = await _latest_ingestion_job(session, document_id)
    if latest_ingestion is not None and latest_ingestion.status in _ACTIVE_INGESTION_STATUSES:
        return _result(ReindexSchedulingOutcome.INGESTION_ACTIVE, document, target_config)

    latest_deletion = await _latest_deletion_job(session, document_id)
    if latest_deletion is not None:
        if latest_deletion.status in _ACTIVE_DELETION_STATUSES:
            return _result(ReindexSchedulingOutcome.DELETION_ACTIVE, document, target_config)
        if latest_deletion.status == DocumentDeletionStatus.PARTIALLY_FAILED:
            return _result(ReindexSchedulingOutcome.DELETION_INCOMPLETE, document, target_config)
        if latest_deletion.status == DocumentDeletionStatus.COMPLETED:
            return _result(ReindexSchedulingOutcome.DOCUMENT_DELETED, document, target_config)

    await ensure_active_collection(vector_store, session, target_config)

    job = ReindexJob(
        id=str(uuid.uuid4()),
        document_id=document_id,
        source_collection_name=source_collection_name,
        target_collection_name=target_config.collection_name,
        target_chunk_size=target_chunk_size,
        target_chunk_overlap=target_chunk_overlap,
        status=ReindexJobStatus.PENDING,
    )
    session.add(job)
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        if not is_active_reindex_job_violation(exc):
            raise

        winner = await get_active_reindex_job(session, document_id)
        if winner is None:
            raise MissingActiveReindexJobAfterRaceError(document_id) from exc
        return _result(ReindexSchedulingOutcome.ALREADY_ACTIVE, document, target_config, winner)

    return _result(ReindexSchedulingOutcome.CREATED, document, target_config, job)


__all__ = [
    "MissingActiveReindexJobAfterRaceError",
    "ReindexSchedulingOutcome",
    "ReindexSchedulingResult",
    "get_active_reindex_job",
    "get_completed_reindex_target_collections",
    "get_document",
    "get_latest_reindex_job",
    "get_reindex_job",
    "is_active_reindex_job_violation",
    "schedule_reindex",
]
