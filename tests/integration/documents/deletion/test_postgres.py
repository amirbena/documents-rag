"""Postgres integration tests for full document deletion — real Testcontainers Postgres, real locks.

Proves properties a fake session double cannot faithfully represent: the partial unique index
actually rejecting a second concurrent active deletion job, append-only history, migration
correctness, and lifecycle-status derivation against real rows. Genuinely concurrent scheduling/
claim stress tests live separately in test_concurrency.py — see that module's docstring for why.
Real Qdrant/MinIO cross-system cleanup is covered separately by test_qdrant.py / test_storage.py.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.models.document import Document
from app.models.document_deletion_job import DocumentDeletionJob, DocumentDeletionStatus
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.services.documents.deletion_service import (
    DeletionRequestOutcome,
    get_latest_deletion_job,
    request_document_deletion,
)
from app.services.documents.query_service import derive_lifecycle_status, get_latest_ingestion_job


@pytest.fixture(autouse=True)
async def _clean_tables(migrated_schema: None, postgres_url: str) -> AsyncIterator[None]:
    """Truncate deletion/ingestion/document tables before and after each test in this module."""
    engine = create_async_engine(postgres_url, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def _truncate() -> None:
        async with session_factory() as session:
            await session.execute(
                text(
                    "TRUNCATE TABLE document_deletion_jobs, ingestion_jobs, documents "
                    "RESTART IDENTITY CASCADE"
                )
            )
            await session.commit()

    await _truncate()
    try:
        yield
    finally:
        await _truncate()
        await engine.dispose()


async def _seed_document(session: AsyncSession) -> Document:
    document = Document(
        id=str(uuid.uuid4()),
        original_filename="report.pdf",
        stored_filename=f"{uuid.uuid4().hex}.pdf",
        content_type="application/pdf",
        file_size=10,
        stored_path="storage/documents/report.pdf",
        storage_provider="local",
        storage_key=f"storage/documents/{uuid.uuid4().hex}.pdf",
    )
    session.add(document)
    await session.commit()
    return document


async def test_migration_creates_document_deletion_jobs_table_and_partial_index(
    migrated_schema: None, postgres_url: str
) -> None:
    """alembic upgrade head must create document_deletion_jobs plus its one-active-per-doc index."""
    engine: AsyncEngine = create_async_engine(postgres_url, future=True)
    try:
        async with engine.connect() as conn:
            table_names = await conn.run_sync(lambda sync_conn: set(inspect(sync_conn).get_table_names()))
            index_names = await conn.run_sync(
                lambda sync_conn: {
                    idx["name"] for idx in inspect(sync_conn).get_indexes("document_deletion_jobs")
                }
            )
    finally:
        await engine.dispose()

    assert "document_deletion_jobs" in table_names
    assert "ix_document_deletion_jobs_one_active_per_document" in index_names


async def test_migration_downgrade_and_reupgrade_removes_only_phase_2_8_4_objects(
    migrated_schema: None, postgres_url: str
) -> None:
    """Downgrading to the prior revision removes document_deletion_jobs but leaves ingestion_jobs' index."""
    from tests.integration.conftest import run_alembic_downgrade, run_alembic_upgrade

    await asyncio.to_thread(run_alembic_downgrade, "b7e2f6a1c9d4")

    engine: AsyncEngine = create_async_engine(postgres_url, future=True)
    try:
        async with engine.connect() as conn:
            table_names_before = await conn.run_sync(
                lambda sync_conn: set(inspect(sync_conn).get_table_names())
            )
            ingestion_index_names = await conn.run_sync(
                lambda sync_conn: {idx["name"] for idx in inspect(sync_conn).get_indexes("ingestion_jobs")}
            )
        assert "document_deletion_jobs" not in table_names_before
        # The Phase 2.8.3 ingestion index must be untouched by this migration's downgrade.
        assert "ix_ingestion_jobs_one_active_per_document" in ingestion_index_names

        await asyncio.to_thread(run_alembic_upgrade, "head")

        async with engine.connect() as conn:
            table_names_after = await conn.run_sync(
                lambda sync_conn: set(inspect(sync_conn).get_table_names())
            )
        assert "document_deletion_jobs" in table_names_after
    finally:
        await engine.dispose()


async def test_partial_unique_index_rejects_two_active_deletion_jobs_for_one_document(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    """A second PENDING/PROCESSING deletion row for the same document violates the unique index."""
    document = await _seed_document(integration_db_session)
    integration_db_session.add(
        DocumentDeletionJob(
            id=str(uuid.uuid4()), document_id=document.id, status=DocumentDeletionStatus.PENDING
        )
    )
    await integration_db_session.commit()

    integration_db_session.add(
        DocumentDeletionJob(
            id=str(uuid.uuid4()), document_id=document.id, status=DocumentDeletionStatus.PROCESSING
        )
    )
    raised = False
    try:
        await integration_db_session.commit()
    except IntegrityError:
        raised = True
        await integration_db_session.rollback()

    assert raised


async def test_partial_unique_index_allows_historical_plus_one_active_row(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    """A COMPLETED row plus one active row for the same document is allowed (append-only history)."""
    document = await _seed_document(integration_db_session)
    integration_db_session.add(
        DocumentDeletionJob(
            id=str(uuid.uuid4()), document_id=document.id, status=DocumentDeletionStatus.PARTIALLY_FAILED
        )
    )
    integration_db_session.add(
        DocumentDeletionJob(
            id=str(uuid.uuid4()), document_id=document.id, status=DocumentDeletionStatus.PENDING
        )
    )
    await integration_db_session.commit()  # must not raise

    count = await integration_db_session.execute(
        text("SELECT count(*) FROM document_deletion_jobs WHERE document_id = :id"), {"id": document.id}
    )
    assert count.scalar_one() == 2


async def test_partial_unique_index_allows_two_different_documents_each_active(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    """Two different documents may each have their own active deletion job simultaneously."""
    doc_a = await _seed_document(integration_db_session)
    doc_b = await _seed_document(integration_db_session)
    integration_db_session.add(
        DocumentDeletionJob(id=str(uuid.uuid4()), document_id=doc_a.id, status=DocumentDeletionStatus.PENDING)
    )
    integration_db_session.add(
        DocumentDeletionJob(id=str(uuid.uuid4()), document_id=doc_b.id, status=DocumentDeletionStatus.PENDING)
    )
    await integration_db_session.commit()  # must not raise


async def test_deletion_history_is_append_only_across_retry(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    """A PARTIALLY_FAILED row must remain queryable, unchanged, after a retry creates a new row."""
    document = await _seed_document(integration_db_session)
    failed = DocumentDeletionJob(
        id=str(uuid.uuid4()),
        document_id=document.id,
        status=DocumentDeletionStatus.PARTIALLY_FAILED,
        vector_cleanup_completed=True,
        storage_cleanup_completed=False,
        error_code="document_storage_cleanup_failed",
        error_message="boom",
    )
    integration_db_session.add(failed)
    await integration_db_session.commit()

    result = await request_document_deletion(integration_db_session, document.id)
    assert result.outcome == DeletionRequestOutcome.CREATED

    rows = await integration_db_session.execute(
        text(
            "SELECT id, status, error_message FROM document_deletion_jobs "
            "WHERE document_id = :id ORDER BY created_at"
        ),
        {"id": document.id},
    )
    stored = rows.all()
    assert len(stored) == 2
    assert stored[0].id == failed.id
    assert stored[0].status == "partially_failed"
    assert stored[0].error_message == "boom"
    assert stored[1].status == "pending"


async def test_lifecycle_derivation_after_completed_deletion(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    """A COMPLETED deletion job must derive DELETED, overriding a COMPLETED ingestion job."""
    document = await _seed_document(integration_db_session)
    integration_db_session.add(
        IngestionJob(id=str(uuid.uuid4()), document_id=document.id, status=IngestionStatus.COMPLETED)
    )
    integration_db_session.add(
        DocumentDeletionJob(
            id=str(uuid.uuid4()),
            document_id=document.id,
            status=DocumentDeletionStatus.COMPLETED,
            vector_cleanup_completed=True,
            storage_cleanup_completed=True,
            completed_at=datetime.now(UTC),
        )
    )
    await integration_db_session.commit()

    latest_ingestion = await get_latest_ingestion_job(integration_db_session, document.id)
    latest_deletion = await get_latest_deletion_job(integration_db_session, document.id)
    status = derive_lifecycle_status(document, latest_ingestion, latest_deletion)

    assert status.value == "deleted"


async def test_lifecycle_derivation_after_partial_failure(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(integration_db_session)
    integration_db_session.add(
        DocumentDeletionJob(
            id=str(uuid.uuid4()),
            document_id=document.id,
            status=DocumentDeletionStatus.PARTIALLY_FAILED,
            vector_cleanup_completed=True,
            storage_cleanup_completed=False,
        )
    )
    await integration_db_session.commit()

    latest_deletion = await get_latest_deletion_job(integration_db_session, document.id)
    status = derive_lifecycle_status(document, None, latest_deletion)

    assert status.value == "deletion_failed"


async def test_ingestion_active_blocks_deletion_scheduling(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(integration_db_session)
    integration_db_session.add(
        IngestionJob(id=str(uuid.uuid4()), document_id=document.id, status=IngestionStatus.PROCESSING)
    )
    await integration_db_session.commit()

    result = await request_document_deletion(integration_db_session, document.id)

    assert result.outcome == DeletionRequestOutcome.INGESTION_ACTIVE

    active_deletion_rows = await integration_db_session.execute(
        text("SELECT count(*) FROM document_deletion_jobs WHERE document_id = :id"), {"id": document.id}
    )
    assert active_deletion_rows.scalar_one() == 0
