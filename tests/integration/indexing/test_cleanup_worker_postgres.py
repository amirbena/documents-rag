"""Postgres integration tests for historical vector-cleanup execution (Phase 2.8.6, subtask 7) —
real Testcontainers Postgres, real row locks, no real Qdrant required.

Proves properties a fake session double cannot faithfully represent: real `SELECT ... FOR UPDATE
SKIP LOCKED` serialization under genuine concurrency, real cross-session durability of persisted
terminal status, and that historical terminal rows are never re-selected. Full orchestration/
decision-table coverage against a fake session lives in
tests/unit/services/indexing/test_cleanup_worker_service.py — this module only covers what a fake
cannot: real constraints and real races. `retry_cleanup_job()`'s own success/failure logic was
already covered by Phase 2.8.1's `test_cleanup_job_service.py`; this module does not repeat it.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.document import Document
from app.models.index_collection import IndexCollection, IndexCollectionStatus
from app.models.vector_cleanup_job import VectorCleanupJob, VectorCleanupStatus
from app.services.indexing.cleanup_job_service import (
    VectorCleanupWorkerOutcome,
    process_next_vector_cleanup_job,
)


class _RecordingVectorStore:
    """A VectorStore double recording every delete call — no real Qdrant needed for these tests,
    which verify Postgres claim/persistence behavior, not actual vector removal."""

    def __init__(self, fail_delete_for: set[str] | None = None) -> None:
        self.deleted: list[tuple[str, str]] = []
        self._fail_delete_for = fail_delete_for or set()

    async def delete_by_document_id(self, collection_name: str, document_id: str) -> None:
        if collection_name in self._fail_delete_for:
            raise RuntimeError(f"could not delete from {collection_name}")
        self.deleted.append((collection_name, document_id))


@pytest.fixture(autouse=True)
async def _clean_tables(migrated_schema: None, postgres_url: str) -> AsyncIterator[None]:
    engine = create_async_engine(postgres_url, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def _truncate() -> None:
        async with session_factory() as session:
            await session.execute(
                text(
                    "TRUNCATE TABLE vector_cleanup_jobs, reindex_jobs, index_collections, "
                    "ingestion_jobs, document_deletion_jobs, documents RESTART IDENTITY CASCADE"
                )
            )
            await session.commit()

    await _truncate()
    try:
        yield
    finally:
        await _truncate()
        await engine.dispose()


async def _ensure_index_collection(session: AsyncSession, collection_name: str) -> None:
    """Seed collection_name's IndexCollection row if it doesn't already exist.

    Document.collection_name carries a foreign key into index_collections (see the alembic
    baseline migration) — a document can never reference a collection that isn't itself persisted there.
    """
    existing = await session.get(IndexCollection, collection_name)
    if existing is None:
        session.add(
            IndexCollection(
                collection_name=collection_name,
                embedding_provider="ollama",
                embedding_model="test-model",
                embedding_dimension=768,
                embedding_version="v1",
                chunking_version="v1",
                status=IndexCollectionStatus.ACTIVE,
            )
        )
        await session.commit()


async def _seed_document(session: AsyncSession, **overrides: object) -> Document:
    fields: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        original_filename="report.pdf",
        stored_filename=f"{uuid.uuid4().hex}.pdf",
        content_type="application/pdf",
        file_size=10,
        stored_path="storage/documents/report.pdf",
        storage_provider="local",
        storage_key=f"storage/documents/{uuid.uuid4().hex}.pdf",
        collection_name="current-collection",
    )
    fields.update(overrides)

    collection_name = fields.get("collection_name")
    if collection_name is not None:
        await _ensure_index_collection(session, collection_name)  # type: ignore[arg-type]

    document = Document(**fields)  # type: ignore[arg-type]
    session.add(document)
    await session.commit()
    return document


async def _seed_cleanup_job(session: AsyncSession, document_id: str, **overrides: object) -> VectorCleanupJob:
    fields: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        document_id=document_id,
        collection_name="old-collection",
        status=VectorCleanupStatus.PENDING,
        attempts=0,
    )
    fields.update(overrides)
    job = VectorCleanupJob(**fields)  # type: ignore[arg-type]
    session.add(job)
    await session.commit()
    return job


async def test_one_eligible_job_is_claimed_and_completed(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(integration_db_session)
    job = await _seed_cleanup_job(integration_db_session, document.id)
    vector_store = _RecordingVectorStore()

    result = await process_next_vector_cleanup_job(integration_db_session, vector_store)

    assert result.outcome == VectorCleanupWorkerOutcome.COMPLETED
    assert result.job_id == job.id
    assert vector_store.deleted == [("old-collection", document.id)]


async def test_successful_cleanup_status_persists_across_sessions(
    migrated_schema: None, postgres_url: str, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(integration_db_session)
    job = await _seed_cleanup_job(integration_db_session, document.id)
    vector_store = _RecordingVectorStore()

    result = await process_next_vector_cleanup_job(integration_db_session, vector_store)
    assert result.outcome == VectorCleanupWorkerOutcome.COMPLETED

    engine = create_async_engine(postgres_url, future=True)
    try:
        session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
        async with session_factory() as fresh_session:
            fresh_job = await fresh_session.get(VectorCleanupJob, job.id)
    finally:
        await engine.dispose()

    assert fresh_job is not None
    assert fresh_job.status == VectorCleanupStatus.COMPLETED
    assert fresh_job.completed_at is not None
    assert fresh_job.last_error is None


async def test_failed_cleanup_status_and_bounded_error_persist(
    migrated_schema: None, postgres_url: str, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(integration_db_session)
    job = await _seed_cleanup_job(integration_db_session, document.id, collection_name="old-collection")
    vector_store = _RecordingVectorStore(fail_delete_for={"old-collection"})

    result = await process_next_vector_cleanup_job(integration_db_session, vector_store)
    assert result.outcome == VectorCleanupWorkerOutcome.FAILED

    engine = create_async_engine(postgres_url, future=True)
    try:
        session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
        async with session_factory() as fresh_session:
            fresh_job = await fresh_session.get(VectorCleanupJob, job.id)
    finally:
        await engine.dispose()

    assert fresh_job is not None
    assert fresh_job.status == VectorCleanupStatus.FAILED
    assert fresh_job.attempts == 1
    assert fresh_job.last_error is not None
    assert "old-collection" in fresh_job.last_error


async def test_historical_terminal_cleanup_jobs_remain_unchanged(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(integration_db_session)
    completed_job = await _seed_cleanup_job(
        integration_db_session,
        document.id,
        collection_name="already-done",
        status=VectorCleanupStatus.COMPLETED,
        completed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    pending_job = await _seed_cleanup_job(
        integration_db_session, document.id, collection_name="old-collection"
    )
    vector_store = _RecordingVectorStore()

    result = await process_next_vector_cleanup_job(integration_db_session, vector_store)

    assert result.outcome == VectorCleanupWorkerOutcome.COMPLETED
    assert result.job_id == pending_job.id

    refreshed_completed = await integration_db_session.get(VectorCleanupJob, completed_job.id)
    assert refreshed_completed is not None
    assert refreshed_completed.status == VectorCleanupStatus.COMPLETED
    assert refreshed_completed.completed_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert ("already-done", document.id) not in vector_store.deleted


@pytest.mark.parametrize("run", range(3))
async def test_concurrent_workers_cannot_claim_the_same_job(
    run: int, migrated_schema: None, postgres_url: str, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(integration_db_session, collection_name=f"current-{run}")
    job = await _seed_cleanup_job(integration_db_session, document.id, collection_name=f"old-{run}")

    engine = create_async_engine(postgres_url, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def _process_with_own_session() -> VectorCleanupWorkerOutcome:
        async with session_factory() as session:
            result = await process_next_vector_cleanup_job(session, _RecordingVectorStore())
            return result.outcome

    try:
        outcomes = await asyncio.gather(_process_with_own_session(), _process_with_own_session())

        # Exactly one call claimed and completed the job; the other found nothing left eligible.
        assert sorted(outcomes) == sorted(
            [VectorCleanupWorkerOutcome.COMPLETED, VectorCleanupWorkerOutcome.NO_JOB]
        )

        async with session_factory() as fresh_session:
            refreshed = await fresh_session.get(VectorCleanupJob, job.id)
        assert refreshed is not None
        assert refreshed.status == VectorCleanupStatus.COMPLETED
    finally:
        await engine.dispose()


async def test_concurrent_workers_claim_different_jobs_under_bounded_concurrency(
    migrated_schema: None, postgres_url: str, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(integration_db_session)
    job_a = await _seed_cleanup_job(
        integration_db_session,
        document.id,
        collection_name="old-a",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    job_b = await _seed_cleanup_job(
        integration_db_session,
        document.id,
        collection_name="old-b",
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    engine = create_async_engine(postgres_url, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def _process_with_own_session() -> str | None:
        async with session_factory() as session:
            result = await process_next_vector_cleanup_job(session, _RecordingVectorStore())
            return result.job_id

    try:
        claimed_ids = await asyncio.gather(_process_with_own_session(), _process_with_own_session())
    finally:
        await engine.dispose()

    assert set(claimed_ids) == {job_a.id, job_b.id}


async def test_one_pending_job_results_in_one_cleanup_invocation(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(integration_db_session)
    await _seed_cleanup_job(integration_db_session, document.id)
    vector_store = _RecordingVectorStore()

    await process_next_vector_cleanup_job(integration_db_session, vector_store)

    assert len(vector_store.deleted) == 1


async def test_active_serving_collection_protection_prevents_cleanup(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(integration_db_session, collection_name="still-active")
    job = await _seed_cleanup_job(integration_db_session, document.id, collection_name="still-active")
    vector_store = _RecordingVectorStore()

    result = await process_next_vector_cleanup_job(integration_db_session, vector_store)

    assert result.outcome == VectorCleanupWorkerOutcome.FAILED
    assert vector_store.deleted == []

    refreshed_job = await integration_db_session.get(VectorCleanupJob, job.id)
    refreshed_document = await integration_db_session.get(Document, document.id)
    assert refreshed_job is not None
    assert refreshed_job.status == VectorCleanupStatus.FAILED
    assert refreshed_document is not None
    assert refreshed_document.collection_name == "still-active"  # document itself untouched


async def test_worker_processes_one_job_per_invocation(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(integration_db_session)
    for index in range(3):
        await _seed_cleanup_job(
            integration_db_session,
            document.id,
            collection_name=f"old-{index}",
            created_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index),
        )
    vector_store = _RecordingVectorStore()

    result = await process_next_vector_cleanup_job(integration_db_session, vector_store)

    assert result.outcome == VectorCleanupWorkerOutcome.COMPLETED
    assert len(vector_store.deleted) == 1

    remaining_pending = await integration_db_session.execute(
        text("SELECT count(*) FROM vector_cleanup_jobs WHERE status = 'pending'")
    )
    assert remaining_pending.scalar_one() == 2
