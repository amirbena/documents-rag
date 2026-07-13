"""Recover PROCESSING ingestion jobs abandoned by a dead worker — an operational maintenance
operation invoked by `scripts/recover_stale_ingestion_jobs.py`, never by any HTTP route.

Finds PROCESSING jobs whose row hasn't been touched in `stale_after_seconds`, marks each one
FAILED (preserving it, never deleting/resetting it), and creates one fresh PENDING replacement
job per recovered row — exactly the same conceptual transition `app.services.ingestion
.retry_service.retry_ingestion()` performs reactively when a client retries a document whose
latest job happens to be stale-PROCESSING (see that module's docstring); both share the fixed
`STALE_RECOVERY_ERROR_PREFIX` marker and job-creation helper in `app.services.ingestion.status` so
the two call sites are indistinguishable in stored data.

## Stale detection is an approximation, not a liveness proof

`IngestionJob` has no dedicated heartbeat column — `updated_at` (bumped by `onupdate=func.now()`
on every status transition) is the only available signal. A PROCESSING job whose `updated_at` is
older than `stale_after_seconds` is *probably* abandoned (crashed/killed worker), but a genuinely
slow-but-alive worker looks identical. `INGESTION_STALE_AFTER_SECONDS` should be set well above
the platform's expected worst-case single-document processing time.
"""

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.services.ingestion.status import create_pending_job, stale_recovery_message


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
        job.error_message = stale_recovery_message(stale_after_seconds)
        replacement = await create_pending_job(session, job.document_id)
        recovered.append(RecoveredJob(stale_job_id=job.id, replacement_job_id=replacement.id))

    if recovered:
        await session.commit()

    return RecoveryResult(recovered=tuple(recovered))


__all__ = [
    "RecoveredJob",
    "RecoveryResult",
    "recover_stale_ingestion_jobs",
]
