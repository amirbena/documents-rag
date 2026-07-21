"""Integration tests for Alembic migrations against a real, ephemeral Postgres container.

Verifies `alembic upgrade head` actually creates the expected schema, that IngestionStatus is
stored as its lowercase string values, and that the documents -> ingestion_jobs foreign key is
enforced — none of which SQLite could represent correctly (see CLAUDE.md's database testing
rules), so this runs against a real Postgres via Testcontainers instead.
"""

import asyncio
import uuid
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from tests.integration.conftest import run_alembic_downgrade, run_alembic_upgrade

if TYPE_CHECKING:
    from app.models.document import Document


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
    assert "vector_cleanup_jobs" in table_names


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
        "storage_provider",
        "storage_bucket",
        "storage_key",
        "storage_etag",
        "content_hash",
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


def _make_document(*, document_id: str | None = None, content_hash: str | None = None) -> "Document":
    """Build a minimal valid Document row for content_hash persistence tests."""
    from app.models.document import Document

    return Document(
        id=document_id or str(uuid.uuid4()),
        original_filename="report.pdf",
        stored_filename=f"{uuid.uuid4().hex}.pdf",
        content_type="application/pdf",
        file_size=10,
        stored_path="storage/documents/report.pdf",
        content_hash=content_hash,
    )


async def test_content_hash_unique_index_exists(migrated_schema: None, postgres_url: str) -> None:
    """The named unique index uq_documents_content_hash must exist after upgrading to head."""
    engine: AsyncEngine = create_async_engine(postgres_url, future=True)
    try:
        async with engine.connect() as conn:
            indexes = await conn.run_sync(lambda sync_conn: inspect(sync_conn).get_indexes("documents"))
    finally:
        await engine.dispose()

    by_name = {index["name"]: index for index in indexes}
    assert "uq_documents_content_hash" in by_name
    assert by_name["uq_documents_content_hash"]["unique"] is True
    assert by_name["uq_documents_content_hash"]["column_names"] == ["content_hash"]


async def test_multiple_documents_with_null_content_hash_are_allowed(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    """PostgreSQL must allow any number of documents with content_hash = NULL."""
    for _ in range(3):
        integration_db_session.add(_make_document(content_hash=None))

    await integration_db_session.commit()  # must not raise


async def test_two_documents_with_different_content_hashes_are_allowed(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    """Two distinct non-null hashes must be able to coexist."""
    integration_db_session.add(_make_document(content_hash="a" * 64))
    integration_db_session.add(_make_document(content_hash="b" * 64))

    await integration_db_session.commit()  # must not raise


async def test_duplicate_content_hash_violates_unique_index(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    """Inserting two documents with the same non-null content_hash must raise IntegrityError."""
    shared_hash = "c" * 64
    integration_db_session.add(_make_document(content_hash=shared_hash))
    await integration_db_session.commit()

    integration_db_session.add(_make_document(content_hash=shared_hash))
    with pytest.raises(IntegrityError):
        await integration_db_session.commit()

    await integration_db_session.rollback()


async def test_content_hash_persists_and_loads_unchanged(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    """A 64-character lowercase hex hash must round-trip through Postgres unchanged."""
    sha256_hex = "d" * 64
    document = _make_document(content_hash=sha256_hex)
    integration_db_session.add(document)
    await integration_db_session.commit()

    result = await integration_db_session.execute(
        text("SELECT content_hash FROM documents WHERE id = :id"), {"id": document.id}
    )
    stored_value = result.scalar_one()

    assert stored_value == sha256_hex
    assert len(stored_value) == 64
    assert stored_value == stored_value.lower()
