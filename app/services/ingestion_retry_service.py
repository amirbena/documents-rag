"""Retry a failed/stale ingestion attempt, and recover PROCESSING jobs abandoned by a dead worker.

Two independent, transactional operations, both scoped to Postgres only — neither ever touches
`FileStorage` or a vector store directly:

- `retry_ingestion()`: the service behind `POST /api/v1/documents/{id}/ingestion/retry`. Never
  mutates an already-FAILED row (it stays FAILED forever, preserving history); the one exception
  is a stale-PROCESSING row, which retry itself flips to FAILED in the same commit as creating the
  replacement (see "latest job PROCESSING and stale" in `retry_ingestion`'s docstring for why this
  is required, not optional). Retrying always means creating a brand-new PENDING row for the same
  document_id, for the existing `IngestionWorker` to claim and process exactly like a first
  attempt. This is safe and requires no vector cleanup: see "Vector idempotency" below.
- `recover_stale_ingestion_jobs()`: an internal maintenance operation (invoked by
  `scripts/recover_stale_ingestion_jobs.py`, not by any HTTP route) that finds PROCESSING jobs
  whose row hasn't been touched in `stale_after_seconds`, marks each one FAILED (preserving it,
  never deleting/resetting it), and creates one fresh PENDING replacement job per recovered row.

## One active job per document

At most one `IngestionJob` per `document_id` may be PENDING or PROCESSING at a time — enforced by
a real Postgres partial unique index (`ix_ingestion_jobs_one_active_per_document`, migration
`b7e2f6a1c9d4`), not merely application logic. `retry_ingestion()` additionally takes a blocking
`SELECT ... FOR UPDATE` lock on the document's existing job rows before deciding whether to
insert, and falls back to catching the unique index's `IntegrityError` (re-reading and returning
the now-existing active job instead of raising) for the residual race the lock alone cannot close
(inserting a brand-new row is never covered by a lock taken on rows that already existed at query
time — see the module's test suite for the concurrent-retry proof). Two concurrent retries for
the same document therefore always converge on exactly one new active job, never two.

## Vector idempotency is free — no cleanup step is needed here

`IngestionWorker._default_process_document()` performs exactly one embedding call followed by
exactly one `vector_store.upsert_vectors()` call; if extraction/chunking/embedding raises, that
exception happens strictly before `upsert_vectors()` is ever reached, so a FAILED (or
stale-recovered) job never wrote any vectors. Chunk IDs
(`f"{document.id}-{chunk_index}"`, see `app/services/document_chunker.py`) and their derived
Qdrant point IDs (`uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)`, see
`app/services/ingestion_worker.to_vector_point()`) are fully deterministic for a given document,
so a retry's successful upsert naturally overwrites the same point IDs a first successful attempt
would have used — Qdrant's own upsert-by-ID semantics make this idempotent with no extra
mechanism. (An orphaned-point edge case exists only if chunking parameters change *between* two
genuinely-successful indexing runs of the same document with different chunk counts — structurally
unreachable within this module's scope, since retry only ever fires for a job that never reached
`upsert_vectors()`.)

## Stale detection is an approximation, not a liveness proof

`IngestionJob` has no dedicated heartbeat column — `updated_at` (bumped by `onupdate=func.now()`
on every status transition) is the only available signal. A PROCESSING job whose `updated_at` is
older than `stale_after_seconds` is *probably* abandoned (crashed/killed worker), but a genuinely
slow-but-alive worker looks identical. `INGESTION_STALE_AFTER_SECONDS` should be set well above
the platform's expected worst-case single-document processing time.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.services.documents.deletion_service import get_latest_deletion_job

# Fixed prefix for a stale-recovery FAILED job's error_message — machine-identifiable in raw
# Postgres data/logs, but never exposed differently by the public API: `sanitize_ingestion_error()`
# (app/services/document_query_service.py) already collapses every error_message, including this
# one, to one fixed generic string before it reaches an API response.
STALE_RECOVERY_ERROR_PREFIX = "STALE_PROCESSING_RECOVERED"


class RetryOutcome(StrEnum):
    """The decided outcome of one retry_ingestion() call — drives the route's HTTP status."""

    DOCUMENT_NOT_FOUND = "document_not_found"
    CREATED = "created"
    ALREADY_ACTIVE = "already_active"
    ALREADY_COMPLETED = "already_completed"
    DELETION_ACTIVE = "deletion_active"


@dataclass(frozen=True)
class IngestionRetryResult:
    """Typed outcome of retry_ingestion(): the outcome plus the relevant job, for the route to map."""

    outcome: RetryOutcome
    job: IngestionJob | None


def _is_stale_processing(job: IngestionJob, *, stale_after_seconds: int, now: datetime) -> bool:
    """A PROCESSING job is stale if its row hasn't been updated within the stale threshold."""
    updated_at = job.updated_at if job.updated_at.tzinfo is not None else job.updated_at.replace(tzinfo=UTC)
    return (now - updated_at).total_seconds() > stale_after_seconds


def _stale_recovery_message(stale_after_seconds: int) -> str:
    """Build the one fixed, machine-identifiable error_message used for every stale recovery."""
    return (
        f"{STALE_RECOVERY_ERROR_PREFIX}: PROCESSING job not updated for over "
        f"{stale_after_seconds}s, treated as abandoned by a dead/crashed worker."
    )


async def _create_pending_job(session: AsyncSession, document_id: str) -> IngestionJob:
    """Insert (but do not commit) a brand-new PENDING IngestionJob row for `document_id`."""
    job = IngestionJob(id=str(uuid.uuid4()), document_id=document_id, status=IngestionStatus.PENDING)
    session.add(job)
    return job


async def _latest_active_job(session: AsyncSession, document_id: str) -> IngestionJob | None:
    """Re-read `document_id`'s latest PENDING/PROCESSING job — used after an IntegrityError race."""
    stmt = (
        select(IngestionJob)
        .where(
            IngestionJob.document_id == document_id,
            IngestionJob.status.in_([IngestionStatus.PENDING, IngestionStatus.PROCESSING]),
        )
        .order_by(IngestionJob.created_at.desc(), IngestionJob.id.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalars().first()


async def retry_ingestion(
    session: AsyncSession,
    document_id: str,
    *,
    stale_after_seconds: int,
    now: datetime | None = None,
) -> IngestionRetryResult:
    """Create a new PENDING attempt for `document_id` if its latest job is FAILED/stale/absent.

    Decision table (latest IngestionJob for the document):
    - no Document row at all -> DOCUMENT_NOT_FOUND (route maps to 404).
    - no IngestionJob row at all -> treated like FAILED: a new PENDING job is created. In
      practice unreachable via the normal upload flow (which always creates one job with the
      document), same defensive stance as document_query_service's UPLOADED status.
    - latest job PENDING, or PROCESSING and not stale -> ALREADY_ACTIVE, no new job created; the
      existing active job is returned (route maps to 200, not 202 — nothing new was scheduled).
    - latest job PROCESSING and stale (per `stale_after_seconds`) -> CREATED. The stale row itself
      *is* transitioned to FAILED as part of this same commit — not left dangling in PROCESSING —
      because the partial unique index (`ix_ingestion_jobs_one_active_per_document`) only allows
      one PENDING/PROCESSING row per document, so a still-PROCESSING row would otherwise make the
      new PENDING insert fail outright. This uses the exact same fixed
      `STALE_RECOVERY_ERROR_PREFIX` marker `recover_stale_ingestion_jobs()` uses, so a
      client-triggered "reactive" stale recovery (via retry) and the background/scheduled
      "proactive" one are indistinguishable in stored data — both are the one conceptual
      transition, just triggered from two different call sites. `recover_stale_ingestion_jobs()`
      remains the only path that recovers a stale job nobody has explicitly retried yet.
    - latest job FAILED, or absent -> CREATED: a new PENDING job is inserted, and the prior FAILED
      row (if any) is left completely unmodified.
    - latest job COMPLETED -> ALREADY_COMPLETED (route maps to 409 — re-index is a separate,
      already-existing endpoint, not this one).
    - any DocumentDeletionJob exists for the document at all (PENDING/PROCESSING/
      PARTIALLY_FAILED/COMPLETED — i.e. the document's lifecycle is DELETING/DELETION_FAILED/
      DELETED, Phase 2.8.4) -> DELETION_ACTIVE (route maps to 409). Checked before any other
      decision below, so a deletion in progress or already completed always blocks ingestion
      retry — a document is never implicitly resurrected by retrying its ingestion.

    Takes a blocking `SELECT ... FOR UPDATE` on the document's existing job rows first, so two
    concurrent retries for an already-active document serialize instead of racing; a residual
    race when the latest job is FAILED/absent (inserting a new row is never covered by a lock on
    rows that already existed) is closed by catching the partial unique index's IntegrityError and
    returning the now-existing active job instead of a duplicate. See module docstring.
    """
    now = now or datetime.now(UTC)

    document = await session.get(Document, document_id)
    if document is None:
        return IngestionRetryResult(outcome=RetryOutcome.DOCUMENT_NOT_FOUND, job=None)

    if await get_latest_deletion_job(session, document_id) is not None:
        return IngestionRetryResult(outcome=RetryOutcome.DELETION_ACTIVE, job=None)

    stmt = (
        select(IngestionJob)
        .where(IngestionJob.document_id == document_id)
        .order_by(IngestionJob.created_at.desc(), IngestionJob.id.desc())
        .with_for_update()
    )
    result = await session.execute(stmt)
    jobs = list(result.scalars().all())
    latest = jobs[0] if jobs else None

    if latest is not None and latest.status == IngestionStatus.COMPLETED:
        return IngestionRetryResult(outcome=RetryOutcome.ALREADY_COMPLETED, job=latest)

    if latest is not None and latest.status == IngestionStatus.PENDING:
        return IngestionRetryResult(outcome=RetryOutcome.ALREADY_ACTIVE, job=latest)

    if latest is not None and latest.status == IngestionStatus.PROCESSING:
        if not _is_stale_processing(latest, stale_after_seconds=stale_after_seconds, now=now):
            return IngestionRetryResult(outcome=RetryOutcome.ALREADY_ACTIVE, job=latest)
        # Stale PROCESSING: must be flipped to FAILED in this same commit — see docstring — so
        # the new PENDING insert below doesn't collide with it under the partial unique index.
        latest.status = IngestionStatus.FAILED
        latest.error_message = _stale_recovery_message(stale_after_seconds)

    new_job = await _create_pending_job(session, document_id)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        existing = await _latest_active_job(session, document_id)
        return IngestionRetryResult(outcome=RetryOutcome.ALREADY_ACTIVE, job=existing)

    return IngestionRetryResult(outcome=RetryOutcome.CREATED, job=new_job)


@dataclass(frozen=True)
class RecoveredJob:
    """One stale PROCESSING job recovered: the row marked FAILED, plus its fresh replacement."""

    stale_job_id: str
    replacement_job_id: str


@dataclass(frozen=True)
class RecoveryResult:
    """Typed outcome of one recover_stale_ingestion_jobs() batch."""

    recovered: tuple[RecoveredJob, ...]

    @property
    def count(self) -> int:
        """Number of stale jobs recovered in this batch."""
        return len(self.recovered)


async def recover_stale_ingestion_jobs(
    session: AsyncSession,
    *,
    batch_size: int,
    stale_after_seconds: int,
    now: datetime | None = None,
) -> RecoveryResult:
    """Mark up to `batch_size` stale PROCESSING jobs FAILED and create a fresh PENDING replacement.

    Locks candidate rows with `SELECT ... FOR UPDATE SKIP LOCKED` (mirroring
    `IngestionWorker._claim_next_pending_job()`'s exact pattern), ordered deterministically
    (`updated_at ASC, id ASC` — oldest-stale-first), so two concurrent recovery runs never both
    recover the same row. Each recovered row is preserved unchanged except for its `status`
    (-> FAILED) and `error_message` (-> a fixed, machine-identifiable `STALE_RECOVERY_ERROR_PREFIX`
    message) — never deleted, never reset back to PENDING. Idempotent: a job already FAILED here
    is PROCESSING no longer, so a later call never re-selects it.
    """
    now = now or datetime.now(UTC)
    cutoff = now.timestamp() - stale_after_seconds

    stmt = (
        select(IngestionJob)
        .where(IngestionJob.status == IngestionStatus.PROCESSING)
        .order_by(IngestionJob.updated_at.asc(), IngestionJob.id.asc())
        .limit(batch_size)
        .with_for_update(skip_locked=True)
    )
    result = await session.execute(stmt)
    candidates = list(result.scalars().all())

    recovered: list[RecoveredJob] = []
    for job in candidates:
        updated_at = (
            job.updated_at if job.updated_at.tzinfo is not None else job.updated_at.replace(tzinfo=UTC)
        )
        if updated_at.timestamp() > cutoff:
            continue  # not stale yet — SKIP LOCKED already excludes concurrently-claimed rows

        job.status = IngestionStatus.FAILED
        job.error_message = _stale_recovery_message(stale_after_seconds)
        replacement = await _create_pending_job(session, job.document_id)
        recovered.append(RecoveredJob(stale_job_id=job.id, replacement_job_id=replacement.id))

    if recovered:
        await session.commit()

    return RecoveryResult(recovered=tuple(recovered))


__all__ = [
    "STALE_RECOVERY_ERROR_PREFIX",
    "IngestionRetryResult",
    "RecoveredJob",
    "RecoveryResult",
    "RetryOutcome",
    "recover_stale_ingestion_jobs",
    "retry_ingestion",
]
