"""Unit tests for app/services/documents/query_service.py against a fake in-memory session.

Covers lifecycle-status derivation, deterministic ordering/pagination, N+1 avoidance, response
field mapping (including the storage_key/bucket/etag exclusion), and failure sanitization. No
Postgres, no HTTP layer — see tests/test_document_read_routes.py for the HTTP-boundary coverage
and tests/integration/documents/read/test_postgres.py for real-Postgres coverage.
"""

from app.models.document_deletion_job import DocumentDeletionStatus
from app.models.ingestion_job import IngestionStatus
from app.schemas.documents import DocumentLifecycleStatus
from app.services.documents.query_service import (
    DEFAULT_LIST_LIMIT,
    build_document_list_response,
    derive_lifecycle_status,
    get_document_detail_result,
    get_document_failure_result,
    get_document_ingestion_result,
    get_latest_failed_ingestion_job,
    get_latest_ingestion_job,
    get_latest_jobs_for_documents,
    sanitize_ingestion_error,
)
from tests.support.documents.read.builders import (
    BASE_TIME,
    build_deletion_job,
    build_document,
    build_ingestion_job,
)
from tests.support.documents.read.fake_session import FakeDocumentQuerySession

# --- Lifecycle status derivation ---------------------------------------------------------------


def test_lifecycle_status_uploaded_when_no_job_exists() -> None:
    """No IngestionJob at all derives to UPLOADED (defensive; unreachable via the normal flow)."""
    document = build_document()
    assert derive_lifecycle_status(document, None) == DocumentLifecycleStatus.UPLOADED


def test_lifecycle_status_pending() -> None:
    document = build_document()
    job = build_ingestion_job(document.id, IngestionStatus.PENDING)
    assert derive_lifecycle_status(document, job) == DocumentLifecycleStatus.PENDING


def test_lifecycle_status_processing() -> None:
    document = build_document()
    job = build_ingestion_job(document.id, IngestionStatus.PROCESSING)
    assert derive_lifecycle_status(document, job) == DocumentLifecycleStatus.PROCESSING


def test_lifecycle_status_failed() -> None:
    document = build_document()
    job = build_ingestion_job(document.id, IngestionStatus.FAILED)
    assert derive_lifecycle_status(document, job) == DocumentLifecycleStatus.FAILED


def test_lifecycle_status_indexed_when_completed_and_indexed_at_set() -> None:
    document = build_document(indexed_at=BASE_TIME)
    job = build_ingestion_job(document.id, IngestionStatus.COMPLETED)
    assert derive_lifecycle_status(document, job) == DocumentLifecycleStatus.INDEXED


def test_lifecycle_status_indexed_even_if_indexed_at_somehow_missing() -> None:
    """A COMPLETED job is authoritative even in the documented, theoretically-unreachable edge case."""
    document = build_document(indexed_at=None)
    job = build_ingestion_job(document.id, IngestionStatus.COMPLETED)
    assert derive_lifecycle_status(document, job) == DocumentLifecycleStatus.INDEXED


# --- list_documents / build_document_list_response ---------------------------------------------


async def test_list_documents_empty() -> None:
    session = FakeDocumentQuerySession()
    response = await build_document_list_response(session, limit=DEFAULT_LIST_LIMIT, offset=0)
    assert response.items == []
    assert response.total == 0
    assert response.limit == DEFAULT_LIST_LIMIT
    assert response.offset == 0


async def test_list_documents_deterministic_ordering_newest_first() -> None:
    session = FakeDocumentQuerySession()
    docs = [build_document(i) for i in range(3)]
    for doc in docs:
        session.add(doc)

    response = await build_document_list_response(session, limit=10, offset=0)

    assert [item.id for item in response.items] == [docs[2].id, docs[1].id, docs[0].id]
    assert response.total == 3


async def test_list_documents_pagination_limit_and_offset() -> None:
    session = FakeDocumentQuerySession()
    docs = [build_document(i) for i in range(5)]
    for doc in docs:
        session.add(doc)

    page = await build_document_list_response(session, limit=2, offset=2)

    assert len(page.items) == 2
    # Newest-first order: docs[4], docs[3], docs[2], docs[1], docs[0] -> offset 2 -> docs[2], docs[1]
    assert [item.id for item in page.items] == [docs[2].id, docs[1].id]
    assert page.total == 5
    assert page.limit == 2
    assert page.offset == 2


async def test_list_documents_includes_latest_job_status_per_document() -> None:
    session = FakeDocumentQuerySession()
    doc = build_document(0)
    session.add(doc)
    session.add(build_ingestion_job(doc.id, IngestionStatus.PENDING, minutes=0))
    session.add(build_ingestion_job(doc.id, IngestionStatus.PROCESSING, minutes=1))

    response = await build_document_list_response(session, limit=10, offset=0)

    assert len(response.items) == 1
    assert response.items[0].status == DocumentLifecycleStatus.PROCESSING


async def test_list_documents_avoids_n_plus_1_queries() -> None:
    """Listing N documents must issue a fixed number of queries, not one job-lookup per row."""
    session = FakeDocumentQuerySession()
    for i in range(10):
        doc = build_document(i)
        session.add(doc)
        session.add(build_ingestion_job(doc.id, IngestionStatus.PENDING))

    await build_document_list_response(session, limit=10, offset=0)

    # Exactly 4 queries: COUNT(*), the page SELECT, one batched latest-ingestion-jobs SELECT, and
    # one batched latest-deletion-jobs SELECT (Phase 2.8.4) — still fixed regardless of page size.
    assert session.execute_count == 4


# --- get_latest_ingestion_job / get_latest_failed_ingestion_job ---------------------------------


async def test_get_latest_ingestion_job_picks_the_most_recent() -> None:
    session = FakeDocumentQuerySession()
    doc = build_document()
    session.add(doc)
    older = build_ingestion_job(doc.id, IngestionStatus.FAILED, minutes=0)
    newer = build_ingestion_job(doc.id, IngestionStatus.COMPLETED, minutes=5)
    session.add(older)
    session.add(newer)

    latest = await get_latest_ingestion_job(session, doc.id)
    assert latest is not None
    assert latest.id == newer.id


async def test_get_latest_ingestion_job_returns_none_when_no_jobs() -> None:
    session = FakeDocumentQuerySession()
    assert await get_latest_ingestion_job(session, "missing-doc") is None


async def test_get_latest_failed_ingestion_job_ignores_non_failed_jobs() -> None:
    session = FakeDocumentQuerySession()
    doc = build_document()
    session.add(doc)
    session.add(build_ingestion_job(doc.id, IngestionStatus.FAILED, minutes=0))
    session.add(build_ingestion_job(doc.id, IngestionStatus.COMPLETED, minutes=5))

    failed = await get_latest_failed_ingestion_job(session, doc.id)
    assert failed is not None
    assert failed.status == IngestionStatus.FAILED


async def test_get_latest_failed_ingestion_job_none_when_never_failed() -> None:
    session = FakeDocumentQuerySession()
    doc = build_document()
    session.add(doc)
    session.add(build_ingestion_job(doc.id, IngestionStatus.COMPLETED))

    assert await get_latest_failed_ingestion_job(session, doc.id) is None


async def test_get_latest_jobs_for_documents_batches_correctly() -> None:
    session = FakeDocumentQuerySession()
    doc_a, doc_b = build_document(0), build_document(1)
    session.add(doc_a)
    session.add(doc_b)
    session.add(build_ingestion_job(doc_a.id, IngestionStatus.FAILED, minutes=0))
    latest_a = build_ingestion_job(doc_a.id, IngestionStatus.COMPLETED, minutes=3)
    session.add(latest_a)
    latest_b = build_ingestion_job(doc_b.id, IngestionStatus.PENDING, minutes=1)
    session.add(latest_b)

    latest = await get_latest_jobs_for_documents(session, [doc_a.id, doc_b.id])

    assert latest[doc_a.id].id == latest_a.id
    assert latest[doc_b.id].id == latest_b.id


async def test_get_latest_jobs_for_documents_empty_list_short_circuits() -> None:
    session = FakeDocumentQuerySession()
    result = await get_latest_jobs_for_documents(session, [])
    assert result == {}
    assert session.execute_count == 0


# --- get_document_detail_result ------------------------------------------------------------------


async def test_get_document_detail_result_missing_document_is_404() -> None:
    session = FakeDocumentQuerySession()
    result = await get_document_detail_result(session, "does-not-exist")
    assert result.status_code == 404
    assert result.response is None


async def test_get_document_detail_result_maps_real_fields_only() -> None:
    session = FakeDocumentQuerySession()
    doc = build_document(
        0,
        storage_provider="minio",
        storage_bucket="documents",
        storage_key="documents/x/y.pdf",
        storage_etag="abc123",
        collection_name="documents_v1",
        embedding_version="v1",
        chunking_version="c1",
        indexed_at=BASE_TIME,
    )
    session.add(doc)
    job = build_ingestion_job(doc.id, IngestionStatus.COMPLETED)
    session.add(job)

    result = await get_document_detail_result(session, doc.id)

    assert result.status_code == 200
    body = result.response
    assert body is not None
    assert body.id == doc.id
    assert body.original_filename == doc.original_filename
    assert body.size_bytes == doc.file_size
    assert body.storage_provider == "minio"
    assert body.status == DocumentLifecycleStatus.INDEXED
    assert body.collection_name == "documents_v1"
    assert body.latest_ingestion_job_id == job.id
    assert body.latest_ingestion_status == IngestionStatus.COMPLETED

    # Never leak internal storage identity/credentials-shaped fields.
    dumped = body.model_dump()
    for forbidden_field in ("storage_key", "storage_bucket", "storage_etag"):
        assert forbidden_field not in dumped


# --- get_document_ingestion_result ---------------------------------------------------------------


async def test_ingestion_result_missing_document_is_404() -> None:
    session = FakeDocumentQuerySession()
    result = await get_document_ingestion_result(session, "does-not-exist")
    assert result.status_code == 404


async def test_ingestion_result_no_job_is_200_with_nulls() -> None:
    session = FakeDocumentQuerySession()
    doc = build_document()
    session.add(doc)

    result = await get_document_ingestion_result(session, doc.id)

    assert result.status_code == 200
    assert result.response is not None
    assert result.response.job_id is None
    assert result.response.status is None
    assert result.response.created_at is None
    assert result.response.updated_at is None


async def test_ingestion_result_reflects_latest_job_of_each_status() -> None:
    for status in IngestionStatus:
        session = FakeDocumentQuerySession()
        doc = build_document()
        session.add(doc)
        job = build_ingestion_job(doc.id, status)
        session.add(job)

        result = await get_document_ingestion_result(session, doc.id)

        assert result.status_code == 200
        assert result.response is not None
        assert result.response.job_id == job.id
        assert result.response.status == status
        assert result.response.created_at == job.created_at
        assert result.response.updated_at == job.updated_at


async def test_ingestion_result_picks_latest_job_among_several() -> None:
    session = FakeDocumentQuerySession()
    doc = build_document()
    session.add(doc)
    session.add(build_ingestion_job(doc.id, IngestionStatus.FAILED, minutes=0))
    session.add(build_ingestion_job(doc.id, IngestionStatus.PENDING, minutes=1))
    newest = build_ingestion_job(doc.id, IngestionStatus.PROCESSING, minutes=2)
    session.add(newest)

    result = await get_document_ingestion_result(session, doc.id)

    assert result.response is not None
    assert result.response.job_id == newest.id
    assert result.response.status == IngestionStatus.PROCESSING


# --- get_document_failure_result / sanitize_ingestion_error --------------------------------------


async def test_failure_result_missing_document_is_404() -> None:
    session = FakeDocumentQuerySession()
    result = await get_document_failure_result(session, "does-not-exist")
    assert result.status_code == 404


async def test_failure_result_no_failed_job_is_404() -> None:
    session = FakeDocumentQuerySession()
    doc = build_document()
    session.add(doc)
    session.add(build_ingestion_job(doc.id, IngestionStatus.COMPLETED))

    result = await get_document_failure_result(session, doc.id)
    assert result.status_code == 404
    assert result.response is None


async def test_failure_result_returns_latest_failed_job() -> None:
    session = FakeDocumentQuerySession()
    doc = build_document()
    session.add(doc)
    session.add(
        build_ingestion_job(
            doc.id,
            IngestionStatus.FAILED,
            minutes=0,
            error_message="qdrant unreachable at http://internal-qdrant:6333/collections",
        )
    )
    newest_failure = build_ingestion_job(
        doc.id, IngestionStatus.FAILED, minutes=5, error_message="File does not look like a valid PDF"
    )
    session.add(newest_failure)

    result = await get_document_failure_result(session, doc.id)

    assert result.status_code == 200
    assert result.response is not None
    assert result.response.job_id == newest_failure.id
    assert result.response.failed_at == newest_failure.updated_at


def test_sanitize_ingestion_error_never_returns_raw_message() -> None:
    """The raw error_message (which may embed a host/connection detail) is never echoed back."""
    raw = "Qdrant unreachable at /collections: ConnectError('internal-qdrant:6333')"
    safe = sanitize_ingestion_error(raw)
    assert safe != raw
    assert "6333" not in safe
    assert "internal-qdrant" not in safe
    # Deterministic and fixed regardless of the raw message's content.
    assert sanitize_ingestion_error("anything else entirely") == safe


# --- Deletion precedence (Phase 2.8.4) ----------------------------------------------------------


def test_derive_lifecycle_status_deleting_takes_precedence_over_indexed() -> None:
    doc = build_document()
    completed_ingestion = build_ingestion_job(doc.id, IngestionStatus.COMPLETED)
    deletion = build_deletion_job(doc.id, DocumentDeletionStatus.PENDING)

    assert derive_lifecycle_status(doc, completed_ingestion, deletion) == DocumentLifecycleStatus.DELETING


def test_derive_lifecycle_status_processing_deletion_is_deleting() -> None:
    doc = build_document()
    deletion = build_deletion_job(doc.id, DocumentDeletionStatus.PROCESSING)

    assert derive_lifecycle_status(doc, None, deletion) == DocumentLifecycleStatus.DELETING


def test_derive_lifecycle_status_partially_failed_deletion_is_deletion_failed() -> None:
    doc = build_document()
    deletion = build_deletion_job(doc.id, DocumentDeletionStatus.PARTIALLY_FAILED)

    assert derive_lifecycle_status(doc, None, deletion) == DocumentLifecycleStatus.DELETION_FAILED


def test_derive_lifecycle_status_completed_deletion_is_deleted() -> None:
    doc = build_document()
    completed_ingestion = build_ingestion_job(doc.id, IngestionStatus.COMPLETED)
    deletion = build_deletion_job(doc.id, DocumentDeletionStatus.COMPLETED)

    assert derive_lifecycle_status(doc, completed_ingestion, deletion) == DocumentLifecycleStatus.DELETED


def test_derive_lifecycle_status_no_deletion_job_falls_back_to_ingestion() -> None:
    doc = build_document()
    pending_ingestion = build_ingestion_job(doc.id, IngestionStatus.PENDING)

    assert derive_lifecycle_status(doc, pending_ingestion, None) == DocumentLifecycleStatus.PENDING
