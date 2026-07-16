"""Atomic activation of a successfully built re-index target (Phase 2.8.6, subtask 5).

`activate_reindexed_document(session, reindex_job_id)` is the explicit, durable cutover: it never
builds or rebuilds anything (no call to `reindex_service.build_reindex_target()` anywhere in this
module), it only switches which collection a document's metadata says it serves from, and — in the
exact same PostgreSQL transaction — persists the cleanup obligation for the collection it is
leaving behind. `ReindexJob.status == COMPLETED` means only "the pinned target build succeeded";
`ReindexJob.activated_at` (added by this subtask) is the separate, durable marker that cutover
actually happened. A job may sit `COMPLETED` with `activated_at IS NULL` indefinitely — build
success and activation eligibility are different facts, and this module never conflates them.

## Why this does not reuse `reindex_service.activate_reindexed_document()`

Subtask 1 already added a same-named primitive in `reindex_service.py` that also flips document
metadata and creates a cleanup job atomically — but it takes a full `EmbeddingIndexConfig`, which
requires a `collection_prefix` that is not itself a persisted column anywhere (only the fully
joined `collection_name` is). The build worker (subtask 4) can safely supply that prefix from its
own live `Settings`, because it already needs `Settings` to resolve a real embedding provider
anyway. Activation must never consult live settings at all (a stale `QDRANT_COLLECTION_NAME` must
never silently corrupt a cutover), so this module updates `Document`'s serving-metadata columns
*directly* from the target `IndexCollection` row's own columns — never through a reconstructed
`EmbeddingIndexConfig` — which is both simpler and structurally immune to any "reconstructed name
disagrees with the persisted one" failure mode, since nothing is ever recomputed.

## Staleness: source ownership must not have changed since scheduling

`ReindexJob.source_collection_name` (added by this subtask, populated by
`reindex_scheduling_service.schedule_reindex()` at scheduling time) is the document's serving
collection *as it was when this job was created*. Activation compares it against the document's
*current* `collection_name`: if they differ, some other valid operation has already moved the
document elsewhere since this job was scheduled — most plausibly, a different `ReindexJob` for the
same document activated first — and this job's build must never overwrite that newer reality.
Example: Job 1 builds B from source A; before Job 1 activates, Job 2 (scheduled and built
independently) activates first, moving the document to C. Job 1 must not later overwrite C with B.
This is reported as `SOURCE_CHANGED`, never silently applied. Comparing `collection_name` alone is
sufficient — `Document.collection_name` is the versioned identity that fully determines the rest
of a document's active embedding/chunking configuration (see `IndexCollection`), so there is no
separate source-embedding-metadata field that could disagree independently.

## Locking, not a new unique constraint, is what makes double-activation impossible

The claimed `ReindexJob` row is loaded with `SELECT ... FOR UPDATE` before any check runs. Two
concurrent activation attempts for the *same* job serialize on that single row's lock — the second
caller only proceeds once the first's transaction has committed, at which point it re-reads
`activated_at` already set and returns `ALREADY_ACTIVATED` — no unique constraint is needed for
this race, because it is one row being contended, not many rows racing an insert (contrast with
`ix_reindex_jobs_one_active_per_document`, which exists precisely because *that* race is a
multi-row insert race).

## Transaction shape

Everything in this module happens in one transaction, ending in exactly one commit: lock the job,
lock the document, re-check every precondition, mutate the document's serving-metadata columns,
add the `VectorCleanupJob` row for the vacated source collection, set `activated_at` — commit once.
No Qdrant call, no extraction, no embedding, and no physical vector deletion happens anywhere in
this module. A failure anywhere before that commit rolls back with nothing applied; scalar
identifiers are captured immediately after the job is loaded, before any possible rollback, so the
failure/early-return paths never re-touch a (possibly-expired) ORM object's attributes — the same
`MissingGreenlet` defect class already found and fixed in subtask 2.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_deletion_job import DocumentDeletionJob, DocumentDeletionStatus
from app.models.index_collection import IndexCollection
from app.models.reindex_job import ReindexJob, ReindexJobStatus
from app.models.vector_cleanup_job import VectorCleanupJob, VectorCleanupStatus

_BLOCKING_DELETION_STATUSES = (
    DocumentDeletionStatus.PENDING,
    DocumentDeletionStatus.PROCESSING,
    DocumentDeletionStatus.PARTIALLY_FAILED,
    DocumentDeletionStatus.COMPLETED,
)


class ReindexActivationOutcome(StrEnum):
    """The decided outcome of one activate_reindexed_document() call."""

    JOB_NOT_FOUND = "job_not_found"
    ALREADY_ACTIVATED = "already_activated"
    NOT_READY = "not_ready"
    DOCUMENT_MISSING = "document_missing"
    BLOCKED_BY_DELETION = "blocked_by_deletion"
    SOURCE_CHANGED = "source_changed"
    ACTIVATED = "activated"


@dataclass(frozen=True)
class ReindexActivationResult:
    """Typed outcome of activate_reindexed_document(): the outcome plus the relevant rows.

    `job` is `None` only for `JOB_NOT_FOUND`. `document` is `None` for `JOB_NOT_FOUND`,
    `ALREADY_ACTIVATED` (idempotent — no document lookup was needed), `NOT_READY`, and
    `DOCUMENT_MISSING` itself; populated for every outcome from `BLOCKED_BY_DELETION` onward, since
    the document was already successfully loaded and locked by that point.
    """

    outcome: ReindexActivationOutcome
    job: ReindexJob | None
    document: Document | None


async def _latest_deletion_job(session: AsyncSession, document_id: str) -> DocumentDeletionJob | None:
    """Return `document_id`'s most recent DocumentDeletionJob — duplicated locally, mirroring
    `reindex_scheduling_service._latest_deletion_job()`'s exact precedent and reasoning."""
    stmt = (
        select(DocumentDeletionJob)
        .where(DocumentDeletionJob.document_id == document_id)
        .order_by(DocumentDeletionJob.created_at.desc(), DocumentDeletionJob.id.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalars().first()


def _result(
    outcome: ReindexActivationOutcome,
    job: ReindexJob | None = None,
    document: Document | None = None,
) -> ReindexActivationResult:
    return ReindexActivationResult(outcome=outcome, job=job, document=document)


async def activate_reindexed_document(session: AsyncSession, reindex_job_id: str) -> ReindexActivationResult:
    """Activate one successfully built ReindexJob: atomic document cutover + deferred cleanup job.

    Deterministic precondition order — the first failing check decides the outcome, nothing after
    it is evaluated:

    1. The job must exist -> `JOB_NOT_FOUND` otherwise.
    2. The job must not already be activated (`activated_at IS NULL`) -> `ALREADY_ACTIVATED`
       otherwise, idempotently, with no further mutation.
    3. The job must be `COMPLETED` -> `NOT_READY` otherwise (`PENDING`/`PROCESSING`/`FAILED`).
    4. The referenced document must exist -> `DOCUMENT_MISSING` otherwise.
    5. The document's deletion lifecycle must not be active/incomplete/completed ->
       `BLOCKED_BY_DELETION` otherwise.
    6. The document's *current* `collection_name` must still equal the job's pinned
       `source_collection_name` -> `SOURCE_CHANGED` otherwise (some other operation already moved
       the document elsewhere since this job was scheduled).
    7. The target `IndexCollection` row must exist -> `NOT_READY` otherwise.

    On success: `Document.collection_name`/`embedding_*`/`chunking_version` are set from the target
    `IndexCollection`'s own columns, `Document.indexed_at` is set to the activation timestamp, a
    `VectorCleanupJob` is persisted for the vacated source collection (skipped only if source and
    target happen to be identical — never reachable via `schedule_reindex()`, which already rejects
    that case at scheduling time, but guarded here defensively), `ReindexJob.activated_at` is set
    to the same timestamp, and all of it commits together, once.
    """
    job_stmt = select(ReindexJob).where(ReindexJob.id == reindex_job_id).with_for_update()
    job_result = await session.execute(job_stmt)
    job = job_result.scalar_one_or_none()
    if job is None:
        return _result(ReindexActivationOutcome.JOB_NOT_FOUND)

    # Captured once, immediately after the job is loaded — never re-read from `job`/`document`
    # after this point, since a later rollback would expire them. See module docstring.
    document_id = job.document_id
    source_collection_name = job.source_collection_name
    target_collection_name = job.target_collection_name

    if job.activated_at is not None:
        return _result(ReindexActivationOutcome.ALREADY_ACTIVATED, job=job)

    if job.status != ReindexJobStatus.COMPLETED:
        return _result(ReindexActivationOutcome.NOT_READY, job=job)

    document_stmt = select(Document).where(Document.id == document_id).with_for_update()
    document_result = await session.execute(document_stmt)
    document = document_result.scalar_one_or_none()
    if document is None:
        return _result(ReindexActivationOutcome.DOCUMENT_MISSING, job=job)

    latest_deletion = await _latest_deletion_job(session, document_id)
    if latest_deletion is not None and latest_deletion.status in _BLOCKING_DELETION_STATUSES:
        return _result(ReindexActivationOutcome.BLOCKED_BY_DELETION, job=job, document=document)

    if document.collection_name != source_collection_name:
        return _result(ReindexActivationOutcome.SOURCE_CHANGED, job=job, document=document)

    target_collection = await session.get(IndexCollection, target_collection_name)
    if target_collection is None:
        return _result(ReindexActivationOutcome.NOT_READY, job=job, document=document)

    activation_timestamp = datetime.now(UTC)

    document.collection_name = target_collection.collection_name
    document.embedding_provider = target_collection.embedding_provider
    document.embedding_model = target_collection.embedding_model
    document.embedding_dimension = target_collection.embedding_dimension
    document.embedding_version = target_collection.embedding_version
    document.chunking_version = target_collection.chunking_version
    document.indexed_at = activation_timestamp

    if source_collection_name and source_collection_name != target_collection_name:
        session.add(
            VectorCleanupJob(
                id=str(uuid.uuid4()),
                document_id=document_id,
                collection_name=source_collection_name,
                status=VectorCleanupStatus.PENDING,
                attempts=0,
                last_error=None,
            )
        )

    job.activated_at = activation_timestamp
    try:
        await session.commit()
    except Exception:
        # Mirrors reindex_service.build_reindex_target()'s exact commit-failure convention: roll
        # back, expire the mutated objects (so no stale in-memory attribute survives the failed
        # commit), and re-raise — never fabricate a result for a failure this module cannot itself
        # resolve. Neither `job` nor `document` is read again after this point.
        await session.rollback()
        session.expire(job)
        session.expire(document)
        raise

    return _result(ReindexActivationOutcome.ACTIVATED, job=job, document=document)


__all__ = [
    "ReindexActivationOutcome",
    "ReindexActivationResult",
    "activate_reindexed_document",
]
