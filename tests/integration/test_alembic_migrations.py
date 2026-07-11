"""Integration tests for Alembic migrations against a real, ephemeral Postgres container.

Verifies `alembic upgrade head` actually creates the expected schema, that IngestionStatus is
stored as its lowercase string values, and that the documents -> ingestion_jobs foreign key is
enforced — none of which SQLite could represent correctly (see CLAUDE.md's database testing
rules), so this runs against a real Postgres via Testcontainers instead.
"""

import asyncio
import uuid

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from tests.integration.conftest import run_alembic_downgrade, run_alembic_upgrade


async def test_upgrade_head_creates_expected_tables(migrated_schema: None, postgres_url: str) -> None:
    """alembic upgrade head should create the documents and ingestion_jobs tables."""
    engine: AsyncEngine = create_async_engine(postgres_url, future=True)
    try:
        async with engine.connect() as conn:
            table_names = await conn.run_sync(lambda sync_conn: inspect(sync_conn).get_table_names())
    finally:
        await engine.dispose()

    assert "documents" in table_names
    assert "ingestion_jobs" in table_names
    assert "index_collections" in table_names


async def test_upgrade_head_creates_expected_columns(migrated_schema: None, postgres_url: str) -> None:
    """The migrated schema should have the columns the ORM models declare."""
    engine: AsyncEngine = create_async_engine(postgres_url, future=True)
    try:
        async with engine.connect() as conn:
            document_columns = await conn.run_sync(
                lambda sync_conn: {col["name"] for col in inspect(sync_conn).get_columns("documents")}
            )
            job_columns = await conn.run_sync(
                lambda sync_conn: {
                    col["name"] for col in inspect(sync_conn).get_columns("ingestion_jobs")
                }
            )
    finally:
        await engine.dispose()

    assert document_columns == {
        "id",
        "original_filename",
        "stored_filename",
        "content_type",
        "file_size",
        "stored_path",
        "created_at",
        "embedding_provider",
        "embedding_model",
        "embedding_dimension",
        "embedding_version",
        "chunking_version",
        "collection_name",
        "indexed_at",
    }
    assert job_columns == {
        "id",
        "document_id",
        "status",
        "error_message",
        "created_at",
        "updated_at",
    }


async def test_ingestion_status_stored_as_lowercase_string(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    """IngestionStatus should be persisted as its lowercase string value, not the enum name."""
    from app.models.document import Document
    from app.models.ingestion_job import IngestionJob, IngestionStatus

    document = Document(
        id=str(uuid.uuid4()),
        original_filename="report.pdf",
        stored_filename=f"{uuid.uuid4().hex}.pdf",
        content_type="application/pdf",
        file_size=10,
        stored_path="storage/documents/report.pdf",
    )
    job = IngestionJob(id=str(uuid.uuid4()), document_id=document.id, status=IngestionStatus.COMPLETED)
    integration_db_session.add(document)
    integration_db_session.add(job)
    await integration_db_session.commit()

    result = await integration_db_session.execute(
        text("SELECT status FROM ingestion_jobs WHERE id = :id"), {"id": job.id}
    )
    stored_value = result.scalar_one()

    assert stored_value == "completed"
    assert stored_value not in {"COMPLETED", "IngestionStatus.COMPLETED"}


async def test_foreign_key_rejects_ingestion_job_for_missing_document(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    """Inserting an IngestionJob for a document_id that doesn't exist should violate the FK."""
    from app.models.ingestion_job import IngestionJob, IngestionStatus

    orphan_job = IngestionJob(
        id=str(uuid.uuid4()),
        document_id=str(uuid.uuid4()),
        status=IngestionStatus.PENDING,
    )
    integration_db_session.add(orphan_job)

    with pytest.raises(IntegrityError):
        await integration_db_session.commit()

    await integration_db_session.rollback()


async def test_downgrade_and_reupgrade_is_stable(migrated_schema: None, postgres_url: str) -> None:
    """downgrade to base then upgrade head again should leave the same schema behind.

    Alembic's env.py drives migrations via asyncio.run(...), which cannot be called from
    inside this test's already-running event loop — so the sync Alembic calls run on a
    separate thread via asyncio.to_thread instead.
    """
    await asyncio.to_thread(run_alembic_downgrade, "base")

    engine: AsyncEngine = create_async_engine(postgres_url, future=True)
    try:
        async with engine.connect() as conn:
            table_names_after_downgrade = await conn.run_sync(
                lambda sync_conn: set(inspect(sync_conn).get_table_names())
            )
    finally:
        await engine.dispose()

    assert "documents" not in table_names_after_downgrade
    assert "ingestion_jobs" not in table_names_after_downgrade

    await asyncio.to_thread(run_alembic_upgrade, "head")

    engine = create_async_engine(postgres_url, future=True)
    try:
        async with engine.connect() as conn:
            table_names_after_reupgrade = await conn.run_sync(
                lambda sync_conn: set(inspect(sync_conn).get_table_names())
            )
    finally:
        await engine.dispose()

    assert "documents" in table_names_after_reupgrade
    assert "ingestion_jobs" in table_names_after_reupgrade


async def test_upgrade_from_previous_revision_adds_indexing_metadata(
    migrated_schema: None, postgres_url: str
) -> None:
    """Upgrading from the prior revision alone should add index_collections + the new columns.

    Runs downgrade to the prior revision (acf1b01d5a02) first, confirming the new table/columns
    are genuinely absent, then upgrades to head and confirms they appear — proving `head` is
    reachable incrementally from the previous revision, not just from a fresh database.
    """
    await asyncio.to_thread(run_alembic_downgrade, "acf1b01d5a02")

    engine: AsyncEngine = create_async_engine(postgres_url, future=True)
    try:
        async with engine.connect() as conn:
            table_names = await conn.run_sync(lambda sync_conn: set(inspect(sync_conn).get_table_names()))
            document_columns = await conn.run_sync(
                lambda sync_conn: {col["name"] for col in inspect(sync_conn).get_columns("documents")}
            )
    finally:
        await engine.dispose()

    assert "index_collections" not in table_names
    assert "indexed_at" not in document_columns

    await asyncio.to_thread(run_alembic_upgrade, "head")

    engine = create_async_engine(postgres_url, future=True)
    try:
        async with engine.connect() as conn:
            table_names = await conn.run_sync(lambda sync_conn: set(inspect(sync_conn).get_table_names()))
            document_columns = await conn.run_sync(
                lambda sync_conn: {col["name"] for col in inspect(sync_conn).get_columns("documents")}
            )
    finally:
        await engine.dispose()

    assert "index_collections" in table_names
    assert "indexed_at" in document_columns
