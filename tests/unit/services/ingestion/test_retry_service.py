"""Unit tests for app.services.ingestion.retry_service against a fake session double — no real DB.

Covers every retry-policy branch, the stale-PROCESSING boundary, and the concurrent-insert race
(simulated via a fake commit()-time IntegrityError). Real Postgres row-locking behavior (the
actual concurrency guarantee) is covered separately by
tests/integration/ingestion/test_retry_postgres.py, per CLAUDE.md's Database Testing Style.
"""

from datetime import timedelta

from app.models.document_deletion_job import DocumentDeletionJob, DocumentDeletionStatus
from app.models.ingestion_job import IngestionStatus
from app.services.ingestion.retry_service import RetryOutcome, retry_ingestion
from app.services.ingestion.status import STALE_RECOVERY_ERROR_PREFIX
from tests.support.ingestion.builders import NOW, build_document, build_ingestion_job
from tests.support.ingestion.fake_session import FakeIngestionRetrySession

STALE_AFTER = 900


def _seed(session: FakeIngestionRetrySession, document, *jobs) -> None:
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
    document = build_document()
    _seed(session, document)

    result = await retry_ingestion(session, document.id, stale_after_seconds=STALE_AFTER, now=NOW)

    assert result.outcome == RetryOutcome.CREATED
    assert result.job is not None
    assert result.job.status == IngestionStatus.PENDING
    assert session.commit_count == 1


async def test_retry_failed_job_creates_new_pending_job_without_resetting_old_one() -> None:
    """A FAILED latest job -> CREATED; the old FAILED row's status is never modified."""
    session = FakeIngestionRetrySession()
    document = build_document()
    failed_job = build_ingestion_job(document.id, IngestionStatus.FAILED)
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
    document = build_document()
    pending_job = build_ingestion_job(document.id, IngestionStatus.PENDING)
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
    document = build_document()
    processing_job = build_ingestion_job(
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
    otherwise make the new PENDING insert impossible. See retry_service's docstring.
    """
    session = FakeIngestionRetrySession()
    document = build_document()
    stale_job = build_ingestion_job(
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
    document = build_document()
    job = build_ingestion_job(
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
    document = build_document()
    completed_job = build_ingestion_job(document.id, IngestionStatus.COMPLETED)
    _seed(session, document, completed_job)

    result = await retry_ingestion(session, document.id, stale_after_seconds=STALE_AFTER, now=NOW)

    assert result.outcome == RetryOutcome.ALREADY_COMPLETED
    assert result.job is not None
    assert result.job.id == completed_job.id


async def test_retry_picks_the_most_recent_job_as_latest() -> None:
    """Multiple historical jobs exist — the most recently created one governs the decision."""
    session = FakeIngestionRetrySession()
    document = build_document()
    old_failed = build_ingestion_job(document.id, IngestionStatus.FAILED, created_at=NOW - timedelta(days=2))
    latest_completed = build_ingestion_job(
        document.id, IngestionStatus.COMPLETED, created_at=NOW - timedelta(hours=1)
    )
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
    document = build_document()
    failed_job = build_ingestion_job(document.id, IngestionStatus.FAILED)
    _seed(session, document, failed_job)

    # Simulate a concurrent request winning the race: its row only becomes visible at the moment
    # this session's own commit is attempted, not at this session's earlier SELECT ... FOR UPDATE.
    winning_job = build_ingestion_job(document.id, IngestionStatus.PENDING, created_at=NOW)
    session.concurrent_winner_job = winning_job
    session.force_next_commit_integrity_error = True

    result = await retry_ingestion(session, document.id, stale_after_seconds=STALE_AFTER, now=NOW)

    assert result.outcome == RetryOutcome.ALREADY_ACTIVE
    assert result.job is not None
    assert result.job.id == winning_job.id
    assert session.rollback_count == 1


async def test_retry_blocked_when_deletion_job_pending() -> None:
    """A Phase 2.8.4 deletion job in progress must block ingestion retry with DELETION_ACTIVE."""
    session = FakeIngestionRetrySession()
    document = build_document()
    failed_job = build_ingestion_job(document.id, IngestionStatus.FAILED)
    _seed(session, document, failed_job)
    session.deletion_jobs["d1"] = DocumentDeletionJob(
        id="d1",
        document_id=document.id,
        status=DocumentDeletionStatus.PENDING,
        vector_cleanup_completed=False,
        storage_cleanup_completed=False,
        created_at=NOW,
        updated_at=NOW,
    )

    result = await retry_ingestion(session, document.id, stale_after_seconds=STALE_AFTER, now=NOW)

    assert result.outcome == RetryOutcome.DELETION_ACTIVE
    assert session.commit_count == 0


async def test_retry_blocked_when_document_already_deleted() -> None:
    """A document whose deletion already COMPLETED must never be implicitly resurrected via retry."""
    session = FakeIngestionRetrySession()
    document = build_document()
    _seed(session, document)
    session.deletion_jobs["d2"] = DocumentDeletionJob(
        id="d2",
        document_id=document.id,
        status=DocumentDeletionStatus.COMPLETED,
        vector_cleanup_completed=True,
        storage_cleanup_completed=True,
        created_at=NOW,
        updated_at=NOW,
    )

    result = await retry_ingestion(session, document.id, stale_after_seconds=STALE_AFTER, now=NOW)

    assert result.outcome == RetryOutcome.DELETION_ACTIVE
