"""VectorCleanupJob persistence: the durable, retryable record of a legacy collection's vectors
still needing deletion after a document was re-indexed into a new collection (see
`app.services.indexing.reindex_service`).

A cleanup failure is tracked here independently of whether the document itself is still
considered stale, so it stays discoverable and retryable even after the document is already
current under the active configuration, and multiple historical collections pending cleanup for
the same document are never conflated into a single record.

## Historical cleanup execution (Phase 2.8.6, subtask 7)

`process_next_vector_cleanup_job()` is the minimal worker wiring this module was missing:
`retry_cleanup_job()` already executed the actual delete-and-mark-terminal logic (Phase 2.8.1), but
nothing claimed "the next eligible job" — `get_pending_cleanup_jobs()` lists candidates without
locking, and nothing ever called `retry_cleanup_job()` in a loop. This module still contains no new
cleanup state machine: `VectorCleanupJob` has no `PROCESSING` status, and none is added here
("preserve the existing contract rather than introducing a new state machine solely for
symmetry" — this subtask's own instruction). The claim step
(`_claim_next_pending_cleanup_job()`) mirrors `reindex_worker.py`'s `_claim_next_pending_reindex_job`
exactly — `SELECT ... FOR UPDATE SKIP LOCKED`, oldest first, one row — except its transaction
commits immediately after the claim (releasing the lock) with no status mutation, since there is no
intermediate status to set. `retry_cleanup_job()` itself is not restructured for this: it is
`process_next_vector_cleanup_job()`'s one delegated call for "do the actual delete and mark
terminal," exactly the "reuse existing cleanup primitive, add minimal runner/service wiring"
instruction this subtask is built around.

## Active-serving-collection safety guard

`retry_cleanup_job()` now defensively refuses to delete from a collection that is still the
document's *current* `collection_name` — a stale or invalid `VectorCleanupJob` record must never
be allowed to delete vectors a document is actively serving from. The check only blocks when the
document exists and its `collection_name` equals the job's own `collection_name` exactly; a missing
document (already fully deleted) never blocks cleanup — see "Prefer safe, idempotent handling
rather than recreating lifecycle state" in this subtask's spec. A blocked attempt is recorded using
the *existing* `FAILED` status and a stable, fixed internal message — never a new persisted status
invented solely to distinguish this case.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.vector_cleanup_job import VectorCleanupJob, VectorCleanupStatus
from app.rag.providers.vector_store import VectorStore

# A fixed, bounded internal message — never the raw exception text truncated arbitrarily, and
# never derived from anything document/collection-specific that could grow unbounded.
_ACTIVE_COLLECTION_GUARD_MESSAGE = (
    "Refusing cleanup: this collection is the document's current active serving collection."
)

# Defensive length cap on a stored Qdrant/driver exception's stringified text — `last_error` is an
# internal operational field (never returned by a public API today), but an unbounded provider
# exception string should still never be allowed to grow a row without limit.
_MAX_ERROR_MESSAGE_LENGTH = 2000


def _bounded_error_message(raw: str) -> str:
    """Truncate `raw` to `_MAX_ERROR_MESSAGE_LENGTH`, appending an ellipsis marker if cut."""
    if len(raw) <= _MAX_ERROR_MESSAGE_LENGTH:
        return raw
    return raw[: _MAX_ERROR_MESSAGE_LENGTH - 1] + "…"


async def create_cleanup_job(
    session: AsyncSession, document_id: str, collection_name: str, error: str | None = None
) -> VectorCleanupJob:
    """Persist a new legacy-vector cleanup for `collection_name`, and commit it.

    Called after a re-index whose new collection/Document metadata already committed
    successfully, but whose immediately-previous-collection vector deletion failed (or was
    never attempted) — the re-index itself is not a failure, so this is tracked separately and
    retryably. Pass `error` (the first attempt's exception, stringified) to record the job as
    already FAILED with one attempt logged; omit it to record a fresh PENDING job.
    """
    job = VectorCleanupJob(
        id=str(uuid.uuid4()),
        document_id=document_id,
        collection_name=collection_name,
        status=VectorCleanupStatus.FAILED if error is not None else VectorCleanupStatus.PENDING,
        attempts=1 if error is not None else 0,
        last_error=error,
    )
    session.add(job)
    await session.commit()
    return job


async def get_pending_cleanup_jobs(
    session: AsyncSession, document_id: str | None = None
) -> list[VectorCleanupJob]:
    """Return every PENDING or FAILED VectorCleanupJob, optionally scoped to one document.

    Multiple rows for the same document are returned independently — a second failed cleanup
    for a different historical collection never overwrites or hides the first.
    """
    stmt = select(VectorCleanupJob).where(
        VectorCleanupJob.status.in_([VectorCleanupStatus.PENDING, VectorCleanupStatus.FAILED])
    )
    result = await session.execute(stmt)
    jobs = list(result.scalars().all())
    if document_id is not None:
        jobs = [job for job in jobs if job.document_id == document_id]
    return jobs


async def _commit_or_rollback(session: AsyncSession, job: VectorCleanupJob) -> None:
    """Commit `session`; on failure, roll back and expire `job` before re-raising.

    Mirrors `reindex_activation.activate_reindexed_document()`'s exact commit-failure convention:
    a broken commit must never leave a half-applied in-memory mutation readable, and `job` must
    never be read again by the caller after this raises (see module docstring's "Do not access
    expired ORM attributes after rollback").
    """
    try:
        await session.commit()
    except Exception:
        await session.rollback()
        session.expire(job)
        raise


async def retry_cleanup_job(
    session: AsyncSession, vector_store: VectorStore, job: VectorCleanupJob
) -> bool:
    """Retry deleting `job`'s collection's vectors for its document; update and commit status.

    Retried regardless of whether the document itself is still considered stale — cleanup
    success/failure is tracked independently of `is_document_stale()`. Idempotent: retrying a
    cleanup whose vectors were already removed (e.g. a partially-succeeded prior attempt) is a
    harmless no-op delete-by-filter call. Returns True (and marks the job COMPLETED) on success,
    False (and marks it FAILED, incrementing `attempts`/recording `last_error`) on failure.

    Defensively refuses to delete when `job.collection_name` equals the document's *current*
    `collection_name` (see module docstring's "Active-serving-collection safety guard") — this is
    always checked first, before any Qdrant call. A missing document (already fully deleted) never
    blocks cleanup; only an exact match against a still-existing document's active collection does.
    """
    document = await session.get(Document, job.document_id)
    if document is not None and document.collection_name == job.collection_name:
        job.status = VectorCleanupStatus.FAILED
        job.attempts += 1
        job.last_error = _ACTIVE_COLLECTION_GUARD_MESSAGE
        await _commit_or_rollback(session, job)
        return False

    try:
        await vector_store.delete_by_document_id(job.collection_name, job.document_id)
    except Exception as exc:
        job.status = VectorCleanupStatus.FAILED
        job.attempts += 1
        job.last_error = _bounded_error_message(str(exc))
        await _commit_or_rollback(session, job)
        return False

    job.status = VectorCleanupStatus.COMPLETED
    job.attempts += 1
    job.last_error = None
    job.completed_at = datetime.now(UTC)
    await _commit_or_rollback(session, job)
    return True


class VectorCleanupWorkerOutcome(StrEnum):
    """The decided outcome of one process_next_vector_cleanup_job() call."""

    NO_JOB = "no_job"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class VectorCleanupWorkerResult:
    """Typed outcome of process_next_vector_cleanup_job(): the outcome plus the claimed job's identity.

    All three identity fields are `None` only for `NO_JOB` — no job was claimed, so there is
    nothing to identify.
    """

    outcome: VectorCleanupWorkerOutcome
    job_id: str | None
    document_id: str | None
    collection_name: str | None


async def _claim_next_pending_cleanup_job(session: AsyncSession) -> VectorCleanupJob | None:
    """Select-for-update the oldest eligible (PENDING/FAILED) cleanup job, skipping locked rows.

    Mirrors `reindex_worker.py`'s `_claim_next_pending_reindex_job` exactly, except this claim's
    own transaction commits with no status mutation — `VectorCleanupJob` has no `PROCESSING`
    status (see module docstring).
    """
    stmt = (
        select(VectorCleanupJob)
        .where(VectorCleanupJob.status.in_([VectorCleanupStatus.PENDING, VectorCleanupStatus.FAILED]))
        .order_by(VectorCleanupJob.created_at, VectorCleanupJob.id)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def process_next_vector_cleanup_job(
    session: AsyncSession, vector_store: VectorStore
) -> VectorCleanupWorkerResult:
    """Claim and process at most one eligible VectorCleanupJob — historical-vector cleanup only.

    Claims one row (`_claim_next_pending_cleanup_job()`), commits immediately to release the row
    lock before any external I/O (never holding a lock during the Qdrant call), then delegates the
    actual delete-and-mark-terminal work entirely to the existing `retry_cleanup_job()` — this
    function never duplicates that logic, never rebuilds vectors, never touches Object Storage, and
    never invokes full document deletion. Returns `NO_JOB` if nothing was eligible.
    """
    job = await _claim_next_pending_cleanup_job(session)
    if job is None:
        return VectorCleanupWorkerResult(
            outcome=VectorCleanupWorkerOutcome.NO_JOB,
            job_id=None,
            document_id=None,
            collection_name=None,
        )

    # Captured immediately after the claim — the upcoming commit expires every object in the
    # session, including `job`; every subsequent step uses these plain values, never `job` itself.
    job_id = job.id
    document_id = job.document_id
    collection_name = job.collection_name
    await session.commit()

    fresh_job = await session.get(VectorCleanupJob, job_id)
    assert fresh_job is not None
    succeeded = await retry_cleanup_job(session, vector_store, fresh_job)

    return VectorCleanupWorkerResult(
        outcome=VectorCleanupWorkerOutcome.COMPLETED if succeeded else VectorCleanupWorkerOutcome.FAILED,
        job_id=job_id,
        document_id=document_id,
        collection_name=collection_name,
    )


__all__ = [
    "VectorCleanupWorkerOutcome",
    "VectorCleanupWorkerResult",
    "create_cleanup_job",
    "get_pending_cleanup_jobs",
    "process_next_vector_cleanup_job",
    "retry_cleanup_job",
]
