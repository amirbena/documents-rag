"""Unit tests for app.services.ingestion_retry_service against a fake session double — no real DB.

Covers every retry-policy branch, the stale-PROCESSING boundary, the concurrent-insert race
(simulated via a fake commit()-time IntegrityError), and stale-recovery batching/idempotency. Real
Postgres row-locking behavior (the actual concurrency guarantee) is covered separately by
tests/integration/test_ingestion_retry_postgres.py, per CLAUDE.md's Database Testing Style.
"""

import uuid
from datetime import UTC, datetime, timedelta

from app.models.document import Document
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.services.ingestion_retry_service import (
    STALE_RECOVERY_ERROR_PREFIX,
    RetryOutcome,
    recover_stale_ingestion_jobs,
    retry_ingestion,
)
from tests.support.fake_ingestion_retry_session import FakeIngestionRetrySession

STALE_AFTER = 900
NOW = datetime(2026, 7, 13, 12, 0, 0, tzinfo=UTC)


def _document(document_id: str | None = None) -> Document:
    return Document(
        id=document_id or str(uuid.uuid4()),
        original_filename="a.pdf",
        stored_filename="a.pdf",
        content_type="application/pdf",
        file_size=10,
        stored_path="a.pdf",
        storage_provider="local",
        storage_key="a.pdf",
        created_at=NOW - timedelta(days=1),
    )


def _job(
    document_id: str,
    status: IngestionStatus,
    *,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> IngestionJob:
    created_at = created_at or (NOW - timedelta(minutes=30))
    return IngestionJob(
        id=str(uuid.uuid4()),
        document_id=document_id,
        status=status,
        created_at=created_at,
        updated_at=updated_at or created_at,
    )


def _seed(session: FakeIngestionRetrySession, document: Document, *jobs: IngestionJob) -> None:
    session.documents[document.id] = document
    for job in jobs:
        session.jobs[job.id] = job


async def test_retry_missing_document_returns_document_not_found() -> None:
    """A document_id with no Document row at all -> DOCUMENT_NOT_FOUND (route maps to 404)."""
    session = FakeIngestionRetrySession()

    result = await retry_ingestion(session, "missing", stale_after_seconds=STALE_AFTER, now=NOW)

    assert result.outcome == RetryOutcome.DOCUMENT_NOT_FOUND
    assert result.job is None


async def test_retry_with_no_job_at_all_creates_a_pending_job() -> None:
    """A document with zero IngestionJob rows is treated like FAILED — a new job is created."""
    session = FakeIngestionRetrySession()
    document = _document()
    _seed(session, document)

    result = await retry_ingestion(session, document.id, stale_after_seconds=STALE_AFTER, now=NOW)

    assert result.outcome == RetryOutcome.CREATED
    assert result.job is not None
    assert result.job.status == IngestionStatus.PENDING
    assert session.commit_count == 1


async def test_retry_failed_job_creates_new_pending_job_without_resetting_old_one() -> None:
    """A FAILED latest job -> CREATED; the old FAILED row's status is never modified."""
    session = FakeIngestionRetrySession()
    document = _document()
    failed_job = _job(document.id, IngestionStatus.FAILED)
    _seed(session, document, failed_job)

    result = await retry_ingestion(session, document.id, stale_after_seconds=STALE_AFTER, now=NOW)

    assert result.outcome == RetryOutcome.CREATED
    assert result.job is not None
    assert result.job.id != failed_job.id
    assert result.job.status == IngestionStatus.PENDING
    assert session.jobs[failed_job.id].status == IngestionStatus.FAILED


async def test_retry_pending_job_is_already_active_no_new_job_created() -> None:
    """A PENDING latest job -> ALREADY_ACTIVE; no new job is created."""
    session = FakeIngestionRetrySession()
    document = _document()
    pending_job = _job(document.id, IngestionStatus.PENDING)
    _seed(session, document, pending_job)

    result = await retry_ingestion(session, document.id, stale_after_seconds=STALE_AFTER, now=NOW)

    assert result.outcome == RetryOutcome.ALREADY_ACTIVE
    assert result.job is not None
    assert result.job.id == pending_job.id
    assert session.commit_count == 0
    assert len(session.jobs) == 1


async def test_retry_fresh_processing_job_is_already_active() -> None:
    """A PROCESSING latest job updated well within the stale threshold -> ALREADY_ACTIVE."""
    session = FakeIngestionRetrySession()
    document = _document()
    processing_job = _job(
        document.id, IngestionStatus.PROCESSING, updated_at=NOW - timedelta(seconds=10)
    )
    _seed(session, document, processing_job)

    result = await retry_ingestion(session, document.id, stale_after_seconds=STALE_AFTER, now=NOW)

    assert result.outcome == RetryOutcome.ALREADY_ACTIVE
    assert result.job is not None
    assert result.job.id == processing_job.id
    assert len(session.jobs) == 1


async def test_retry_stale_processing_job_creates_new_pending_job() -> None:
    """A PROCESSING job past the stale threshold -> CREATED, and the stale row is flipped FAILED.

    The stale row must transition to FAILED in the same commit (not stay PROCESSING) because the
    partial unique index only allows one active job per document — a still-PROCESSING row would
    otherwise make the new PENDING insert impossible. See ingestion_retry_service's docstring.
    """
    session = FakeIngestionRetrySession()
    document = _document()
    stale_job = _job(
        document.id,
        IngestionStatus.PROCESSING,
        updated_at=NOW - timedelta(seconds=STALE_AFTER + 1),
    )
    _seed(session, document, stale_job)

    result = await retry_ingestion(session, document.id, stale_after_seconds=STALE_AFTER, now=NOW)

    assert result.outcome == RetryOutcome.CREATED
    assert result.job is not None
    assert result.job.id != stale_job.id
    assert session.jobs[stale_job.id].status == IngestionStatus.FAILED
    assert session.jobs[stale_job.id].error_message is not None
    assert session.jobs[stale_job.id].error_message.startswith(STALE_RECOVERY_ERROR_PREFIX)


async def test_retry_stale_boundary_just_under_threshold_is_not_stale() -> None:
    """A PROCESSING job updated exactly at (not past) the threshold is not treated as stale."""
    session = FakeIngestionRetrySession()
    document = _document()
    job = _job(
        document.id,
        IngestionStatus.PROCESSING,
        updated_at=NOW - timedelta(seconds=STALE_AFTER - 1),
    )
    _seed(session, document, job)

    result = await retry_ingestion(session, document.id, stale_after_seconds=STALE_AFTER, now=NOW)

    assert result.outcome == RetryOutcome.ALREADY_ACTIVE


async def test_retry_completed_job_returns_already_completed() -> None:
    """A COMPLETED latest job -> ALREADY_COMPLETED (route maps to 409, use re-index instead)."""
    session = FakeIngestionRetrySession()
    document = _document()
    completed_job = _job(document.id, IngestionStatus.COMPLETED)
    _seed(session, document, completed_job)

    result = await retry_ingestion(session, document.id, stale_after_seconds=STALE_AFTER, now=NOW)

    assert result.outcome == RetryOutcome.ALREADY_COMPLETED
    assert result.job is not None
    assert result.job.id == completed_job.id


async def test_retry_picks_the_most_recent_job_as_latest() -> None:
    """Multiple historical jobs exist — the most recently created one governs the decision."""
    session = FakeIngestionRetrySession()
    document = _document()
    old_failed = _job(document.id, IngestionStatus.FAILED, created_at=NOW - timedelta(days=2))
    latest_completed = _job(document.id, IngestionStatus.COMPLETED, created_at=NOW - timedelta(hours=1))
    _seed(session, document, old_failed, latest_completed)

    result = await retry_ingestion(session, document.id, stale_after_seconds=STALE_AFTER, now=NOW)

    assert result.outcome == RetryOutcome.ALREADY_COMPLETED
    assert result.job is not None
    assert result.job.id == latest_completed.id


async def test_retry_concurrent_insert_race_returns_existing_active_job() -> None:
    """A commit()-time unique-index violation is caught and returns the now-existing active job.

    Simulates two concurrent retry requests both reaching the "create a new job" branch for the
    same document: this session's commit() is forced to raise IntegrityError once (as the real
    partial unique index would for a genuine second concurrent INSERT), and retry_ingestion must
    translate that into ALREADY_ACTIVE with the winning job, never propagate a raw 500.
    """
    session = FakeIngestionRetrySession()
    document = _document()
    failed_job = _job(document.id, IngestionStatus.FAILED)
    _seed(session, document, failed_job)

    # Simulate a concurrent request winning the race: its row only becomes visible at the moment
    # this session's own commit is attempted, not at this session's earlier SELECT ... FOR UPDATE.
    winning_job = _job(document.id, IngestionStatus.PENDING, created_at=NOW)
    session.concurrent_winner_job = winning_job
    session.force_next_commit_integrity_error = True

    result = await retry_ingestion(session, document.id, stale_after_seconds=STALE_AFTER, now=NOW)

    assert result.outcome == RetryOutcome.ALREADY_ACTIVE
    assert result.job is not None
    assert result.job.id == winning_job.id
    assert session.rollback_count == 1


async def test_recover_stale_jobs_marks_failed_and_creates_replacement() -> None:
    """A stale PROCESSING job is marked FAILED (with a fixed prefix) and gets a PENDING replacement."""
    session = FakeIngestionRetrySession()
    document = _document()
    stale_job = _job(
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
    document = _document()
    fresh_job = _job(document.id, IngestionStatus.PROCESSING, updated_at=NOW - timedelta(seconds=5))
    _seed(session, document, fresh_job)

    result = await recover_stale_ingestion_jobs(
        session, batch_size=50, stale_after_seconds=STALE_AFTER, now=NOW
    )

    assert result.count == 0
    assert session.jobs[fresh_job.id].status == IngestionStatus.PROCESSING


async def test_recover_stale_jobs_excludes_non_processing_statuses() -> None:
    """PENDING/FAILED/COMPLETED jobs are never candidates for stale recovery, however old."""
    session = FakeIngestionRetrySession()
    document = _document()
    ancient_pending = _job(document.id, IngestionStatus.PENDING, updated_at=NOW - timedelta(days=10))
    _seed(session, document, ancient_pending)

    result = await recover_stale_ingestion_jobs(
        session, batch_size=50, stale_after_seconds=STALE_AFTER, now=NOW
    )

    assert result.count == 0


async def test_recover_stale_jobs_respects_batch_size() -> None:
    """No more than `batch_size` stale jobs are recovered in one call."""
    session = FakeIngestionRetrySession()
    for _ in range(5):
        document = _document()
        stale_job = _job(
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
    document = _document()
    stale_job = _job(
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
