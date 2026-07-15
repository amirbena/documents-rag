"""Unit tests for app.services.documents.deletion_service against a fake session double.

Covers request_document_deletion()'s full decision table, the concurrent-insert race, and public
error sanitization — request-scoped behavior only. Never exercises Qdrant, storage, or worker
claim/execution logic — see test_deletion_worker.py for that. Real Postgres row-locking behavior
is covered separately by tests/integration/documents/deletion/test_postgres.py.
"""

import uuid
from datetime import timedelta

from app.models.document_deletion_job import DocumentDeletionStatus
from app.models.ingestion_job import IngestionStatus
from app.models.reindex_job import ReindexJobStatus
from app.services.documents.deletion_service import (
    DeletionErrorCode,
    DeletionRequestOutcome,
    request_document_deletion,
    sanitize_deletion_error,
)
from tests.support.documents.deletion.builders import (
    NOW,
    build_deletion_job,
    build_document,
    build_ingestion_job,
    build_reindex_job,
)
from tests.support.documents.deletion.fake_session import FakeDocumentDeletionSession

# --- request_document_deletion() decision table -----------------------------------------------


async def test_request_deletion_missing_document_returns_not_found() -> None:
    session = FakeDocumentDeletionSession()

    result = await request_document_deletion(session, "missing")

    assert result.outcome == DeletionRequestOutcome.DOCUMENT_NOT_FOUND
    assert result.job is None


async def test_request_deletion_with_no_prior_attempt_creates_pending_job() -> None:
    session = FakeDocumentDeletionSession()
    document = build_document()
    session.documents[document.id] = document

    result = await request_document_deletion(session, document.id)

    assert result.outcome == DeletionRequestOutcome.CREATED
    assert result.job is not None
    assert result.job.status == DocumentDeletionStatus.PENDING
    assert session.commit_count == 1


async def test_request_deletion_active_ingestion_blocks_with_409() -> None:
    session = FakeDocumentDeletionSession()
    document = build_document()
    session.documents[document.id] = document
    session.ingestion_jobs[str(uuid.uuid4())] = build_ingestion_job(document.id, IngestionStatus.PENDING)

    result = await request_document_deletion(session, document.id)

    assert result.outcome == DeletionRequestOutcome.INGESTION_ACTIVE
    assert result.job is None
    assert session.commit_count == 0


async def test_request_deletion_processing_ingestion_also_blocks() -> None:
    session = FakeDocumentDeletionSession()
    document = build_document()
    session.documents[document.id] = document
    session.ingestion_jobs[str(uuid.uuid4())] = build_ingestion_job(
        document.id, IngestionStatus.PROCESSING
    )

    result = await request_document_deletion(session, document.id)

    assert result.outcome == DeletionRequestOutcome.INGESTION_ACTIVE


async def test_request_deletion_completed_ingestion_allows_deletion() -> None:
    session = FakeDocumentDeletionSession()
    document = build_document()
    session.documents[document.id] = document
    session.ingestion_jobs[str(uuid.uuid4())] = build_ingestion_job(document.id, IngestionStatus.COMPLETED)

    result = await request_document_deletion(session, document.id)

    assert result.outcome == DeletionRequestOutcome.CREATED


async def test_request_deletion_failed_ingestion_allows_deletion() -> None:
    session = FakeDocumentDeletionSession()
    document = build_document()
    session.documents[document.id] = document
    session.ingestion_jobs[str(uuid.uuid4())] = build_ingestion_job(document.id, IngestionStatus.FAILED)

    result = await request_document_deletion(session, document.id)

    assert result.outcome == DeletionRequestOutcome.CREATED


async def test_request_deletion_pending_job_is_already_active() -> None:
    session = FakeDocumentDeletionSession()
    document = build_document()
    session.documents[document.id] = document
    existing = build_deletion_job(document.id, DocumentDeletionStatus.PENDING)
    session.deletion_jobs[existing.id] = existing

    result = await request_document_deletion(session, document.id)

    assert result.outcome == DeletionRequestOutcome.ALREADY_ACTIVE
    assert result.job is not None
    assert result.job.id == existing.id
    assert session.commit_count == 0
    assert len(session.deletion_jobs) == 1


async def test_request_deletion_processing_job_is_already_active() -> None:
    session = FakeDocumentDeletionSession()
    document = build_document()
    session.documents[document.id] = document
    existing = build_deletion_job(document.id, DocumentDeletionStatus.PROCESSING)
    session.deletion_jobs[existing.id] = existing

    result = await request_document_deletion(session, document.id)

    assert result.outcome == DeletionRequestOutcome.ALREADY_ACTIVE
    assert result.job is not None
    assert result.job.id == existing.id


async def test_request_deletion_already_completed_is_idempotent() -> None:
    session = FakeDocumentDeletionSession()
    document = build_document()
    session.documents[document.id] = document
    completed = build_deletion_job(document.id, DocumentDeletionStatus.COMPLETED)
    session.deletion_jobs[completed.id] = completed

    result = await request_document_deletion(session, document.id)

    assert result.outcome == DeletionRequestOutcome.ALREADY_DELETED
    assert result.job is not None
    assert result.job.id == completed.id
    assert session.commit_count == 0
    assert len(session.deletion_jobs) == 1


async def test_request_deletion_partially_failed_creates_new_attempt_without_losing_history() -> None:
    session = FakeDocumentDeletionSession()
    document = build_document()
    session.documents[document.id] = document
    failed = build_deletion_job(document.id, DocumentDeletionStatus.PARTIALLY_FAILED)
    failed.error_code = DeletionErrorCode.DOCUMENT_STORAGE_CLEANUP_FAILED
    session.deletion_jobs[failed.id] = failed

    result = await request_document_deletion(session, document.id)

    assert result.outcome == DeletionRequestOutcome.CREATED
    assert result.job is not None
    assert result.job.id != failed.id
    assert result.job.status == DocumentDeletionStatus.PENDING
    # Prior failed attempt is preserved, untouched.
    assert session.deletion_jobs[failed.id].status == DocumentDeletionStatus.PARTIALLY_FAILED
    assert session.deletion_jobs[failed.id].error_code == DeletionErrorCode.DOCUMENT_STORAGE_CLEANUP_FAILED


async def test_request_deletion_picks_most_recent_job_as_latest() -> None:
    session = FakeDocumentDeletionSession()
    document = build_document()
    session.documents[document.id] = document
    old = build_deletion_job(
        document.id, DocumentDeletionStatus.PARTIALLY_FAILED, created_at=NOW - timedelta(days=2)
    )
    latest = build_deletion_job(
        document.id, DocumentDeletionStatus.COMPLETED, created_at=NOW - timedelta(hours=1)
    )
    session.deletion_jobs[old.id] = old
    session.deletion_jobs[latest.id] = latest

    result = await request_document_deletion(session, document.id)

    assert result.outcome == DeletionRequestOutcome.ALREADY_DELETED
    assert result.job is not None
    assert result.job.id == latest.id


async def test_request_deletion_concurrent_insert_race_returns_existing_active_job() -> None:
    """A commit()-time unique-index violation is caught, returning the winning concurrent job."""
    session = FakeDocumentDeletionSession()
    document = build_document()
    session.documents[document.id] = document

    winning_job = build_deletion_job(document.id, DocumentDeletionStatus.PENDING, created_at=NOW)
    session.concurrent_winner_job = winning_job
    session.force_next_commit_integrity_error = True

    result = await request_document_deletion(session, document.id)

    assert result.outcome == DeletionRequestOutcome.ALREADY_ACTIVE
    assert result.job is not None
    assert result.job.id == winning_job.id
    assert session.rollback_count == 1


# --- active-reindex interlock (Phase 2.8.6) -----------------------------------------------------


async def test_request_deletion_active_reindex_pending_blocks_with_409() -> None:
    session = FakeDocumentDeletionSession()
    document = build_document()
    session.documents[document.id] = document
    active = build_reindex_job(document.id, ReindexJobStatus.PENDING)
    session.reindex_jobs[active.id] = active

    result = await request_document_deletion(session, document.id)

    assert result.outcome == DeletionRequestOutcome.REINDEX_ACTIVE
    assert result.job is None
    assert session.commit_count == 0


async def test_request_deletion_active_reindex_processing_also_blocks() -> None:
    session = FakeDocumentDeletionSession()
    document = build_document()
    session.documents[document.id] = document
    active = build_reindex_job(document.id, ReindexJobStatus.PROCESSING)
    session.reindex_jobs[active.id] = active

    result = await request_document_deletion(session, document.id)

    assert result.outcome == DeletionRequestOutcome.REINDEX_ACTIVE
    assert result.job is None
    assert session.commit_count == 0
    assert len(session.deletion_jobs) == 0


async def test_request_deletion_failed_reindex_does_not_block() -> None:
    session = FakeDocumentDeletionSession()
    document = build_document()
    session.documents[document.id] = document
    session.reindex_jobs[str(uuid.uuid4())] = build_reindex_job(document.id, ReindexJobStatus.FAILED)

    result = await request_document_deletion(session, document.id)

    assert result.outcome == DeletionRequestOutcome.CREATED


async def test_request_deletion_completed_reindex_does_not_block() -> None:
    session = FakeDocumentDeletionSession()
    document = build_document()
    session.documents[document.id] = document
    session.reindex_jobs[str(uuid.uuid4())] = build_reindex_job(document.id, ReindexJobStatus.COMPLETED)

    result = await request_document_deletion(session, document.id)

    assert result.outcome == DeletionRequestOutcome.CREATED


# --- sanitize_deletion_error() ------------------------------------------------------------------


def test_sanitize_deletion_error_returns_none_for_no_error() -> None:
    assert sanitize_deletion_error(None) is None


def test_sanitize_deletion_error_never_echoes_raw_message() -> None:
    message = sanitize_deletion_error(DeletionErrorCode.DOCUMENT_VECTOR_CLEANUP_FAILED.value)
    assert message is not None
    assert "qdrant" not in message.lower()
    assert "connection" not in message.lower()


def test_sanitize_deletion_error_falls_back_for_unknown_code() -> None:
    assert sanitize_deletion_error("some_unrecognized_code") is not None
