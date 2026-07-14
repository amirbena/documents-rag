"""Unit tests for app/services/documents/dedup_service.py — the upload deduplication decision
model, against a fake in-memory session. No storage, no Postgres, no wiring into the real upload
flow yet (see the module docstring — this is an internal, not-yet-connected decision model).
"""

import inspect

from app.models.document_deletion_job import DocumentDeletionStatus
from app.models.ingestion_job import IngestionStatus
from app.services.documents import dedup_service
from app.services.documents.dedup_service import (
    DeletionActiveError,
    DeletionIncompleteError,
    DeletionInvariantViolationError,
    UploadOutcome,
    compute_content_hash,
    decide_upload,
)
from tests.support.documents.read.builders import build_deletion_job, build_document, build_ingestion_job
from tests.support.documents.read.fake_session import FakeDocumentQuerySession

_SOME_HASH = "a" * 64


# --- compute_content_hash -------------------------------------------------------------------


def test_identical_bytes_produce_identical_hashes() -> None:
    content = b"identical content for hashing"
    assert compute_content_hash(content) == compute_content_hash(content)


def test_different_bytes_produce_different_hashes() -> None:
    assert compute_content_hash(b"content A") != compute_content_hash(b"content B")


def test_hash_is_lowercase_and_64_characters() -> None:
    digest = compute_content_hash(b"some uploaded bytes")
    assert len(digest) == 64
    assert digest == digest.lower()
    assert all(c in "0123456789abcdef" for c in digest)


# --- decide_upload: no match -------------------------------------------------------------------


async def test_no_matching_document_results_in_created_decision() -> None:
    session = FakeDocumentQuerySession()

    decision = await decide_upload(session, _SOME_HASH)

    assert decision.outcome == UploadOutcome.CREATED
    assert decision.document is None
    assert decision.ingestion_job is None


# --- decide_upload: ingestion-state decisions (no blocking deletion) ---------------------------


async def test_pending_ingestion_results_in_reused_active() -> None:
    session = FakeDocumentQuerySession()
    doc = build_document(content_hash=_SOME_HASH)
    session.add(doc)
    session.add(build_ingestion_job(doc.id, IngestionStatus.PENDING))

    decision = await decide_upload(session, _SOME_HASH)

    assert decision.outcome == UploadOutcome.REUSED_ACTIVE
    assert decision.document is not None
    assert decision.document.id == doc.id


async def test_processing_ingestion_results_in_reused_active() -> None:
    session = FakeDocumentQuerySession()
    doc = build_document(content_hash=_SOME_HASH)
    session.add(doc)
    session.add(build_ingestion_job(doc.id, IngestionStatus.PROCESSING))

    decision = await decide_upload(session, _SOME_HASH)

    assert decision.outcome == UploadOutcome.REUSED_ACTIVE


async def test_completed_ingestion_results_in_reused_indexed() -> None:
    session = FakeDocumentQuerySession()
    doc = build_document(content_hash=_SOME_HASH)
    session.add(doc)
    session.add(build_ingestion_job(doc.id, IngestionStatus.COMPLETED))

    decision = await decide_upload(session, _SOME_HASH)

    assert decision.outcome == UploadOutcome.REUSED_INDEXED


async def test_failed_ingestion_results_in_reused_failed() -> None:
    session = FakeDocumentQuerySession()
    doc = build_document(content_hash=_SOME_HASH)
    session.add(doc)
    session.add(build_ingestion_job(doc.id, IngestionStatus.FAILED))

    decision = await decide_upload(session, _SOME_HASH)

    assert decision.outcome == UploadOutcome.REUSED_FAILED


# --- decide_upload: deletion precedence over ingestion -----------------------------------------


async def test_pending_deletion_takes_precedence_over_indexed_ingestion() -> None:
    session = FakeDocumentQuerySession()
    doc = build_document(content_hash=_SOME_HASH)
    session.add(doc)
    session.add(build_ingestion_job(doc.id, IngestionStatus.COMPLETED))
    session.add(build_deletion_job(doc.id, DocumentDeletionStatus.PENDING))

    try:
        await decide_upload(session, _SOME_HASH)
        raise AssertionError("expected DeletionActiveError")
    except DeletionActiveError as exc:
        assert exc.document_id == doc.id


async def test_processing_deletion_takes_precedence_over_indexed_ingestion() -> None:
    session = FakeDocumentQuerySession()
    doc = build_document(content_hash=_SOME_HASH)
    session.add(doc)
    session.add(build_ingestion_job(doc.id, IngestionStatus.COMPLETED))
    session.add(build_deletion_job(doc.id, DocumentDeletionStatus.PROCESSING))

    try:
        await decide_upload(session, _SOME_HASH)
        raise AssertionError("expected DeletionActiveError")
    except DeletionActiveError as exc:
        assert exc.document_id == doc.id


async def test_partially_failed_deletion_produces_incomplete_deletion_error() -> None:
    session = FakeDocumentQuerySession()
    doc = build_document(content_hash=_SOME_HASH)
    session.add(doc)
    session.add(build_ingestion_job(doc.id, IngestionStatus.COMPLETED))
    session.add(build_deletion_job(doc.id, DocumentDeletionStatus.PARTIALLY_FAILED))

    try:
        await decide_upload(session, _SOME_HASH)
        raise AssertionError("expected DeletionIncompleteError")
    except DeletionIncompleteError as exc:
        assert exc.document_id == doc.id


async def test_completed_deletion_with_non_null_hash_is_invariant_violation() -> None:
    """A matched document (non-null content_hash, by construction) with a COMPLETED deletion job
    means the hash was never released — a data-invariant break, not a normal conflict.
    """
    session = FakeDocumentQuerySession()
    doc = build_document(content_hash=_SOME_HASH)
    session.add(doc)
    session.add(build_ingestion_job(doc.id, IngestionStatus.COMPLETED))
    session.add(build_deletion_job(doc.id, DocumentDeletionStatus.COMPLETED))

    try:
        await decide_upload(session, _SOME_HASH)
        raise AssertionError("expected DeletionInvariantViolationError")
    except DeletionInvariantViolationError as exc:
        assert exc.document_id == doc.id


# --- reuse never mutates the original document --------------------------------------------------


async def test_same_bytes_different_filename_returns_existing_document_unchanged() -> None:
    """decide_upload() takes only a content_hash — it cannot rename or otherwise mutate the
    original document regardless of what filename a new upload attempt would have used.
    """
    session = FakeDocumentQuerySession()
    doc = build_document(content_hash=_SOME_HASH, original_filename="original-name.pdf")
    session.add(doc)
    session.add(build_ingestion_job(doc.id, IngestionStatus.COMPLETED))

    decision = await decide_upload(session, _SOME_HASH)

    assert decision.document is not None
    assert decision.document.id == doc.id
    assert decision.document.original_filename == "original-name.pdf"
    assert "original_filename" not in inspect.signature(decide_upload).parameters


# --- reuse paths write nothing ------------------------------------------------------------------


def test_decide_upload_has_no_storage_dependency() -> None:
    """decide_upload() must never be able to call FileStorage — it isn't given one at all."""
    signature = inspect.signature(decide_upload)
    assert "storage" not in signature.parameters
    assert "FileStorage" not in inspect.getsource(dedup_service)
    assert ".save(" not in inspect.getsource(dedup_service)


async def test_reuse_outcomes_create_no_new_ingestion_job() -> None:
    session = FakeDocumentQuerySession()
    doc = build_document(content_hash=_SOME_HASH)
    session.add(doc)
    session.add(build_ingestion_job(doc.id, IngestionStatus.FAILED))
    jobs_before = dict(session.jobs)

    decision = await decide_upload(session, _SOME_HASH)

    assert decision.outcome == UploadOutcome.REUSED_FAILED
    assert session.jobs == jobs_before
    assert len(session.documents) == 1
