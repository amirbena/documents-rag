"""Postgres integration tests for atomic reindex activation (Phase 2.8.6, subtask 5) — real
Testcontainers Postgres, real row locks, real foreign keys, real transactional rollback.

Proves properties a fake session double cannot faithfully represent: the new migration's columns
and foreign key, real commit atomicity (document cutover + cleanup job land together or not at
all), real `SELECT ... FOR UPDATE` serialization preventing double activation under genuine
concurrency, and real rollback actually undoing an in-progress mutation. Full decision-table
coverage against a fake session lives in tests/unit/services/indexing/test_reindex_activation.py —
this module only covers what a fake cannot: real constraints and real races. Build execution,
Qdrant, and cleanup-job execution are all out of scope for this subtask.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.models.document import Document
from app.models.document_deletion_job import DocumentDeletionJob, DocumentDeletionStatus
from app.models.index_collection import IndexCollection, IndexCollectionStatus
from app.models.reindex_job import ReindexJob, ReindexJobStatus
from app.models.vector_cleanup_job import VectorCleanupStatus
from app.services.indexing.reindex_activation import ReindexActivationOutcome, activate_reindexed_document


@pytest.fixture(autouse=True)
async def _clean_tables(migrated_schema: None, postgres_url: str) -> AsyncIterator[None]:
    engine = create_async_engine(postgres_url, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def _truncate() -> None:
        async with session_factory() as session:
            await session.execute(
                text(
                    "TRUNCATE TABLE reindex_jobs, vector_cleanup_jobs, document_deletion_jobs, "
                    "index_collections, ingestion_jobs, documents RESTART IDENTITY CASCADE"
                )
            )
            await session.commit()

    await _truncate()
    try:
        yield
    finally:
        await _truncate()
        await engine.dispose()


def _index_collection(collection_name: str, **overrides: object) -> IndexCollection:
    fields: dict[str, object] = dict(
        collection_name=collection_name,
        embedding_provider="ollama",
        embedding_model="target-model",
        embedding_dimension=768,
        embedding_version="v1",
        chunking_version="v1",
        status=IndexCollectionStatus.ACTIVE,
    )
    fields.update(overrides)
    return IndexCollection(**fields)  # type: ignore[arg-type]


async def _seed_index_collection(
    session: AsyncSession, collection_name: str, **overrides: object
) -> IndexCollection:
    existing = await session.get(IndexCollection, collection_name)
    if existing is not None:
        return existing
    record = _index_collection(collection_name, **overrides)
    session.add(record)
    await session.commit()
    return record


async def _seed_document(
    session: AsyncSession, collection_name: str = "source-collection", **overrides: object
) -> Document:
    await _seed_index_collection(session, collection_name)

    fields: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        original_filename="report.pdf",
        stored_filename=f"{uuid.uuid4().hex}.pdf",
        content_type="application/pdf",
        file_size=10,
        stored_path="storage/documents/report.pdf",
        storage_provider="local",
        storage_key=f"storage/documents/{uuid.uuid4().hex}.pdf",
        collection_name=collection_name,
    )
    fields.update(overrides)
    document = Document(**fields)  # type: ignore[arg-type]
    session.add(document)
    await session.commit()
    return document


async def _seed_ready_job(
    session: AsyncSession,
    document: Document,
    target_collection_name: str = "target-collection",
    **overrides: object,
) -> ReindexJob:
    await _seed_index_collection(session, target_collection_name)
    fields: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        document_id=document.id,
        source_collection_name=document.collection_name,
        target_collection_name=target_collection_name,
        target_chunk_size=500,
        target_chunk_overlap=50,
        status=ReindexJobStatus.COMPLETED,
    )
    fields.update(overrides)
    job = ReindexJob(**fields)  # type: ignore[arg-type]
    session.add(job)
    await session.commit()
    return job


# --- migration ------------------------------------------------------------------------------


async def test_migration_adds_source_and_activation_columns_with_expected_nullability(
    migrated_schema: None, postgres_url: str
) -> None:
    engine: AsyncEngine = create_async_engine(postgres_url, future=True)
    try:
        async with engine.connect() as conn:
            columns = await conn.run_sync(
                lambda sync_conn: {
                    col["name"]: col["nullable"] for col in inspect(sync_conn).get_columns("reindex_jobs")
                }
            )
    finally:
        await engine.dispose()

    assert columns["source_collection_name"] is False
    assert columns["activated_at"] is True


async def test_migration_downgrade_and_reupgrade_succeeds(migrated_schema: None, postgres_url: str) -> None:
    from tests.integration.conftest import run_alembic_downgrade, run_alembic_upgrade

    await asyncio.to_thread(run_alembic_downgrade, "a8685da857f3")

    engine: AsyncEngine = create_async_engine(postgres_url, future=True)
    try:
        async with engine.connect() as conn:
            columns_before = await conn.run_sync(
                lambda sync_conn: {col["name"] for col in inspect(sync_conn).get_columns("reindex_jobs")}
            )
        assert "source_collection_name" not in columns_before
        assert "activated_at" not in columns_before

        await asyncio.to_thread(run_alembic_upgrade, "head")

        async with engine.connect() as conn:
            columns_after = await conn.run_sync(
                lambda sync_conn: {col["name"] for col in inspect(sync_conn).get_columns("reindex_jobs")}
            )
        assert {"source_collection_name", "activated_at"} <= columns_after
    finally:
        await engine.dispose()


async def test_source_collection_name_foreign_key_is_enforced(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(integration_db_session)
    await _seed_index_collection(integration_db_session, "target-collection")

    integration_db_session.add(
        ReindexJob(
            id=str(uuid.uuid4()),
            document_id=document.id,
            source_collection_name="never-persisted-source-collection",
            target_collection_name="target-collection",
            target_chunk_size=500,
            target_chunk_overlap=50,
            status=ReindexJobStatus.COMPLETED,
        )
    )
    with pytest.raises(IntegrityError):
        await integration_db_session.commit()
    await integration_db_session.rollback()


# --- real commit atomicity ---------------------------------------------------------------------


async def test_successful_activation_persists_document_and_cleanup_job_together(
    migrated_schema: None, postgres_url: str, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(integration_db_session)
    job = await _seed_ready_job(integration_db_session, document)

    result = await activate_reindexed_document(integration_db_session, job.id)
    assert result.outcome == ReindexActivationOutcome.ACTIVATED

    engine = create_async_engine(postgres_url, future=True)
    try:
        session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
        async with session_factory() as fresh_session:
            fresh_document = await fresh_session.get(Document, document.id)
            fresh_job = await fresh_session.get(ReindexJob, job.id)
            cleanup_rows = (
                await fresh_session.execute(
                    text("SELECT collection_name, status FROM vector_cleanup_jobs WHERE document_id = :id"),
                    {"id": document.id},
                )
            ).all()
    finally:
        await engine.dispose()

    assert fresh_document is not None
    assert fresh_document.collection_name == "target-collection"
    assert fresh_job is not None
    assert fresh_job.activated_at is not None
    assert fresh_job.status == ReindexJobStatus.COMPLETED
    assert len(cleanup_rows) == 1
    assert cleanup_rows[0].collection_name == "source-collection"
    assert cleanup_rows[0].status == VectorCleanupStatus.PENDING


async def test_activation_is_durable_across_sessions(
    migrated_schema: None, postgres_url: str, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(integration_db_session)
    job = await _seed_ready_job(integration_db_session, document)
    await activate_reindexed_document(integration_db_session, job.id)

    engine = create_async_engine(postgres_url, future=True)
    try:
        session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
        async with session_factory() as fresh_session:
            result = await activate_reindexed_document(fresh_session, job.id)
    finally:
        await engine.dispose()

    assert result.outcome == ReindexActivationOutcome.ALREADY_ACTIVATED


# --- real rollback --------------------------------------------------------------------------


async def test_blocked_activation_leaves_document_and_job_unmodified(
    migrated_schema: None, postgres_url: str, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(integration_db_session)
    job = await _seed_ready_job(integration_db_session, document)
    integration_db_session.add(
        DocumentDeletionJob(
            id=str(uuid.uuid4()),
            document_id=document.id,
            status=DocumentDeletionStatus.PENDING,
            vector_cleanup_completed=False,
            storage_cleanup_completed=False,
        )
    )
    await integration_db_session.commit()

    result = await activate_reindexed_document(integration_db_session, job.id)
    assert result.outcome == ReindexActivationOutcome.BLOCKED_BY_DELETION

    engine = create_async_engine(postgres_url, future=True)
    try:
        session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
        async with session_factory() as fresh_session:
            fresh_document = await fresh_session.get(Document, document.id)
            fresh_job = await fresh_session.get(ReindexJob, job.id)
            cleanup_count = (
                await fresh_session.execute(
                    text("SELECT count(*) FROM vector_cleanup_jobs WHERE document_id = :id"),
                    {"id": document.id},
                )
            ).scalar_one()
    finally:
        await engine.dispose()

    assert fresh_document is not None
    assert fresh_document.collection_name == "source-collection"
    assert fresh_job is not None
    assert fresh_job.activated_at is None
    assert cleanup_count == 0


# --- real concurrency: FOR UPDATE serialization -------------------------------------------------


@pytest.mark.parametrize("run", range(3))
async def test_concurrent_activation_of_the_same_job_activates_exactly_once(
    run: int, migrated_schema: None, postgres_url: str, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(integration_db_session, collection_name=f"source-collection-{run}")
    job = await _seed_ready_job(
        integration_db_session, document, target_collection_name=f"target-collection-{run}"
    )

    engine = create_async_engine(postgres_url, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def _activate_with_own_session() -> ReindexActivationOutcome:
        async with session_factory() as session:
            result = await activate_reindexed_document(session, job.id)
            return result.outcome

    try:
        outcomes = await asyncio.gather(_activate_with_own_session(), _activate_with_own_session())
    finally:
        await engine.dispose()

    assert sorted(outcomes) == sorted(
        [ReindexActivationOutcome.ACTIVATED, ReindexActivationOutcome.ALREADY_ACTIVATED]
    )

    cleanup_count = await integration_db_session.execute(
        text("SELECT count(*) FROM vector_cleanup_jobs WHERE document_id = :id"), {"id": document.id}
    )
    assert cleanup_count.scalar_one() == 1  # exactly one cleanup job, never duplicated by the race


async def test_second_reindex_job_for_the_same_document_sees_source_changed_after_first_activates(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    """Job 1 builds B from A; Job 2 (independently scheduled/built) also targets C from A. Job 1
    activates first, moving the document to B. Job 2 must then see SOURCE_CHANGED, never silently
    overwrite B with C."""
    document = await _seed_document(integration_db_session, collection_name="collection-a")
    job_one = await _seed_ready_job(
        integration_db_session,
        document,
        target_collection_name="collection-b",
        source_collection_name="collection-a",
    )

    result_one = await activate_reindexed_document(integration_db_session, job_one.id)
    assert result_one.outcome == ReindexActivationOutcome.ACTIVATED

    await _seed_index_collection(integration_db_session, "collection-c")
    job_two = ReindexJob(
        id=str(uuid.uuid4()),
        document_id=document.id,
        source_collection_name="collection-a",
        target_collection_name="collection-c",
        target_chunk_size=500,
        target_chunk_overlap=50,
        status=ReindexJobStatus.COMPLETED,
    )
    integration_db_session.add(job_two)
    await integration_db_session.commit()

    result_two = await activate_reindexed_document(integration_db_session, job_two.id)

    assert result_two.outcome == ReindexActivationOutcome.SOURCE_CHANGED

    refreshed_document = await integration_db_session.get(Document, document.id)
    assert refreshed_document is not None
    assert refreshed_document.collection_name == "collection-b"  # never overwritten by job two


async def test_activated_completed_job_coexists_with_a_new_active_job_under_the_partial_index(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    """The one-active-job-per-document partial unique index must not treat an activated COMPLETED
    row as blocking — a new PENDING job for the same document is still allowed afterward."""
    document = await _seed_document(integration_db_session)
    job = await _seed_ready_job(integration_db_session, document)
    result = await activate_reindexed_document(integration_db_session, job.id)
    assert result.outcome == ReindexActivationOutcome.ACTIVATED

    await _seed_index_collection(integration_db_session, "yet-another-target")
    integration_db_session.add(
        ReindexJob(
            id=str(uuid.uuid4()),
            document_id=document.id,
            source_collection_name="target-collection",
            target_collection_name="yet-another-target",
            target_chunk_size=500,
            target_chunk_overlap=50,
            status=ReindexJobStatus.PENDING,
        )
    )
    await integration_db_session.commit()  # must not raise
