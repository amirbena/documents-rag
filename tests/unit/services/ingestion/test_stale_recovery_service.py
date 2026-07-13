"""Unit tests for app.services.ingestion.stale_recovery_service against a fake session double.

Covers stale-recovery batching, ordering, and idempotency. Real Postgres `SKIP LOCKED`
concurrency behavior is covered separately by
tests/integration/ingestion/test_recovery_concurrency.py, per CLAUDE.md's Database Testing Style.
"""

from datetime import timedelta

from app.models.ingestion_job import IngestionStatus
from app.services.ingestion.stale_recovery_service import recover_stale_ingestion_jobs
from app.services.ingestion.status import STALE_RECOVERY_ERROR_PREFIX
from tests.support.ingestion.builders import NOW, build_document, build_ingestion_job
from tests.support.ingestion.fake_session import FakeIngestionRetrySession

STALE_AFTER = 900


def _seed(session: FakeIngestionRetrySession, document, *jobs) -> None:
    session.documents[document.id] = document
    for job in jobs:
        session.jobs[job.id] = job


async def test_recover_stale_jobs_marks_failed_and_creates_replacement() -> None:
    """A stale PROCESSING job is marked FAILED (with a fixed prefix) and gets a PENDING replacement."""
    session = FakeIngestionRetrySession()
    document = build_document()
    stale_job = build_ingestion_job(
        document.id,
        IngestionStatus.PROCESSING,
        updated_at=NOW - timedelta(seconds=STALE_AFTER + 1),
    )
    _seed(session, document, stale_job)

    result = await recover_stale_ingestion_jobs(
        session, batch_size=50, stale_after_seconds=STALE_AFTER, now=NOW
    )

    assert result.count == 1
    assert result.recovered[0].stale_job_id == stale_job.id
    recovered_stale = session.jobs[stale_job.id]
    assert recovered_stale.status == IngestionStatus.FAILED
    assert recovered_stale.error_message is not None
    assert recovered_stale.error_message.startswith(STALE_RECOVERY_ERROR_PREFIX)

    replacement = session.jobs[result.recovered[0].replacement_job_id]
    assert replacement.status == IngestionStatus.PENDING
    assert replacement.document_id == document.id


async def test_recover_stale_jobs_excludes_non_stale_processing_jobs() -> None:
    """A PROCESSING job updated recently is not recovered."""
    session = FakeIngestionRetrySession()
    document = build_document()
    fresh_job = build_ingestion_job(
        document.id, IngestionStatus.PROCESSING, updated_at=NOW - timedelta(seconds=5)
    )
    _seed(session, document, fresh_job)

    result = await recover_stale_ingestion_jobs(
        session, batch_size=50, stale_after_seconds=STALE_AFTER, now=NOW
    )

    assert result.count == 0
    assert session.jobs[fresh_job.id].status == IngestionStatus.PROCESSING


async def test_recover_stale_jobs_excludes_non_processing_statuses() -> None:
    """PENDING/FAILED/COMPLETED jobs are never candidates for stale recovery, however old."""
    session = FakeIngestionRetrySession()
    document = build_document()
    ancient_pending = build_ingestion_job(
        document.id, IngestionStatus.PENDING, updated_at=NOW - timedelta(days=10)
    )
    _seed(session, document, ancient_pending)

    result = await recover_stale_ingestion_jobs(
        session, batch_size=50, stale_after_seconds=STALE_AFTER, now=NOW
    )

    assert result.count == 0


async def test_recover_stale_jobs_respects_batch_size() -> None:
    """No more than `batch_size` stale jobs are recovered in one call."""
    session = FakeIngestionRetrySession()
    for _ in range(5):
        document = build_document()
        stale_job = build_ingestion_job(
            document.id,
            IngestionStatus.PROCESSING,
            updated_at=NOW - timedelta(seconds=STALE_AFTER + 100),
        )
        _seed(session, document, stale_job)

    result = await recover_stale_ingestion_jobs(
        session, batch_size=2, stale_after_seconds=STALE_AFTER, now=NOW
    )

    assert result.count == 2


async def test_recover_stale_jobs_is_idempotent_on_repeated_calls() -> None:
    """A job already recovered (now FAILED) is never re-recovered by a later call."""
    session = FakeIngestionRetrySession()
    document = build_document()
    stale_job = build_ingestion_job(
        document.id,
        IngestionStatus.PROCESSING,
        updated_at=NOW - timedelta(seconds=STALE_AFTER + 1),
    )
    _seed(session, document, stale_job)

    first = await recover_stale_ingestion_jobs(
        session, batch_size=50, stale_after_seconds=STALE_AFTER, now=NOW
    )
    second = await recover_stale_ingestion_jobs(
        session, batch_size=50, stale_after_seconds=STALE_AFTER, now=NOW
    )

    assert first.count == 1
    assert second.count == 0
