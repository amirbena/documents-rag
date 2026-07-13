"""Small constants and helpers shared by `app.services.ingestion.retry_service` and
`app.services.ingestion.stale_recovery_service` — both flip a `PROCESSING` `IngestionJob` to
`FAILED` using the exact same fixed marker and create the same shape of replacement `PENDING` job,
so real drift between the two call sites (a "reactive" retry-triggered recovery and a
"proactive" scheduled one) would otherwise be a genuine risk.
"""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.ingestion_job import IngestionJob, IngestionStatus

# Fixed prefix for a stale-recovery FAILED job's error_message — machine-identifiable in raw
# Postgres data/logs, but never exposed differently by the public API: `sanitize_ingestion_error()`
# (app/services/documents/query_service.py) already collapses every error_message, including this
# one, to one fixed generic string before it reaches an API response.
STALE_RECOVERY_ERROR_PREFIX = "STALE_PROCESSING_RECOVERED"


def stale_recovery_message(stale_after_seconds: int) -> str:
    """Build the one fixed, machine-identifiable error_message used for every stale recovery."""
    return (
        f"{STALE_RECOVERY_ERROR_PREFIX}: PROCESSING job not updated for over "
        f"{stale_after_seconds}s, treated as abandoned by a dead/crashed worker."
    )


async def create_pending_job(session: AsyncSession, document_id: str) -> IngestionJob:
    """Insert (but do not commit) a brand-new PENDING IngestionJob row for `document_id`."""
    job = IngestionJob(id=str(uuid.uuid4()), document_id=document_id, status=IngestionStatus.PENDING)
    session.add(job)
    return job


__all__ = [
    "STALE_RECOVERY_ERROR_PREFIX",
    "create_pending_job",
    "stale_recovery_message",
]
