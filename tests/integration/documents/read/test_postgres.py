"""Integration tests for app/services/documents/query_service.py against a real, ephemeral
Postgres container — covers ordering/pagination/latest-job selection behavior that a fake
in-memory session cannot faithfully prove (real row storage, real query execution).
"""

import uuid
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.services.documents.query_service import (
    build_document_list_response,
    get_document,
    get_latest_failed_ingestion_job,
    get_latest_ingestion_job,
)

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
async def _clean_tables(migrated_schema: None, integration_db_session: AsyncSession) -> AsyncIterator[None]:
    """Truncate documents/ingestion_jobs before each test for isolation between tests."""
    await integration_db_session.execute(
        text("TRUNCATE TABLE ingestion_jobs, documents RESTART IDENTITY CASCADE")
    )
    await integration_db_session.commit()
    yield


def _document(**overrides: object) -> Document:
    defaults: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        original_filename="report.pdf",
        stored_filename=f"{uuid.uuid4().hex}.pdf",
        content_type="application/pdf",
        file_size=123,
        stored_path=f"documents/{uuid.uuid4()}/report.pdf",
        storage_provider="local",
        storage_key=f"documents/{uuid.uuid4()}/report.pdf",
    )
    defaults.update(overrides)
    return Document(**defaults)  # type: ignore[arg-type]


def _job(document_id: str, status: IngestionStatus, **overrides: object) -> IngestionJob:
    defaults: dict[str, object] = dict(id=str(uuid.uuid4()), document_id=document_id, status=status)
    defaults.update(overrides)
    return IngestionJob(**defaults)  # type: ignore[arg-type]


async def test_list_documents_orders_newest_first_and_paginates(
    integration_db_session: AsyncSession,
) -> None:
    """Multiple real documents come back newest-created-first, with correct paging metadata."""
    docs = [_document(original_filename=f"doc-{i}.pdf") for i in range(3)]
    for doc in docs:
        integration_db_session.add(doc)
        await integration_db_session.commit()

    response = await build_document_list_response(integration_db_session, limit=2, offset=0)

    assert response.total == 3
    assert len(response.items) == 2
    # Newest-first: the last-inserted document (highest created_at) comes first.
    assert response.items[0].id == docs[-1].id


async def test_latest_job_selection_picks_the_true_latest_among_several(
    integration_db_session: AsyncSession,
) -> None:
    """A document with multiple jobs resolves to the actually-latest one, not just the first row."""
    doc = _document()
    integration_db_session.add(doc)
    await integration_db_session.commit()

    job1 = _job(doc.id, IngestionStatus.FAILED)
    integration_db_session.add(job1)
    await integration_db_session.commit()

    job2 = _job(doc.id, IngestionStatus.COMPLETED)
    integration_db_session.add(job2)
    await integration_db_session.commit()

    latest = await get_latest_ingestion_job(integration_db_session, doc.id)
    assert latest is not None
    assert latest.id == job2.id


async def test_document_with_no_job_is_a_real_reachable_row(integration_db_session: AsyncSession) -> None:
    """A Document row with zero IngestionJob rows is valid data the service must handle gracefully."""
    doc = _document()
    integration_db_session.add(doc)
    await integration_db_session.commit()

    latest = await get_latest_ingestion_job(integration_db_session, doc.id)
    assert latest is None

    fetched = await get_document(integration_db_session, doc.id)
    assert fetched is not None
    assert fetched.id == doc.id


async def test_latest_failed_job_selection(integration_db_session: AsyncSession) -> None:
    """The latest FAILED job is found even when a later, non-failed job exists for the same document."""
    doc = _document()
    integration_db_session.add(doc)
    await integration_db_session.commit()

    failed_job = _job(doc.id, IngestionStatus.FAILED, error_message="boom")
    integration_db_session.add(failed_job)
    await integration_db_session.commit()

    pending_job = _job(doc.id, IngestionStatus.PENDING)
    integration_db_session.add(pending_job)
    await integration_db_session.commit()

    latest_failed = await get_latest_failed_ingestion_job(integration_db_session, doc.id)
    assert latest_failed is not None
    assert latest_failed.id == failed_job.id


async def test_unrelated_documents_never_leak_into_each_others_results(
    integration_db_session: AsyncSession,
) -> None:
    """Document A's jobs must never appear when listing/inspecting Document B."""
    doc_a = _document(original_filename="a.pdf")
    doc_b = _document(original_filename="b.pdf")
    integration_db_session.add(doc_a)
    integration_db_session.add(doc_b)
    await integration_db_session.commit()

    integration_db_session.add(_job(doc_a.id, IngestionStatus.FAILED))
    await integration_db_session.commit()

    latest_for_b = await get_latest_ingestion_job(integration_db_session, doc_b.id)
    assert latest_for_b is None

    response = await build_document_list_response(integration_db_session, limit=10, offset=0)
    by_id = {item.id: item for item in response.items}
    assert by_id[doc_a.id].status.value == "failed"
    assert by_id[doc_b.id].status.value == "uploaded"
