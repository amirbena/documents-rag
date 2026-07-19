"""Postgres concurrency/lifecycle integration tests for ReindexWorker — real Testcontainers
Postgres, real row locking.

A fake `build_reindex_target()` delegate is monkeypatched at module level throughout (the same
technique the unit tests use), so these tests validate claiming/locking/lifecycle persistence, not
real extraction/embedding/Qdrant — see test_reindex_worker_build_postgres.py for the one real-build
scenario using genuine extraction/chunking/fake-embedding/real Qdrant.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.services.indexing.reindex_worker as reindex_worker_module
from app.core.config import get_settings
from app.models.document import Document
from app.models.index_collection import IndexCollection, IndexCollectionStatus
from app.models.reindex_job import ReindexJob, ReindexJobStatus
from app.rag.embedding_config import EmbeddingIndexConfig
from app.services.indexing.reindex_worker import ReindexWorker, ReindexWorkerOutcome

# The worker reconstructs EmbeddingIndexConfig from IndexCollection's fields and recomputes its
# .collection_name deterministically (see reindex_worker.py's "IndexCollection does not persist
# collection_prefix separately" docstring section) — an arbitrary collection_name string would
# never match, so every seeded IndexCollection/ReindexJob pair below is derived from a real config.
_SERVING_CONFIG = EmbeddingIndexConfig(
    collection_prefix="documents", provider="ollama", model="serving-model",
    dimension=3, embedding_version="v0", chunking_version="v0",
)
_TARGET_CONFIG = EmbeddingIndexConfig(
    collection_prefix="documents", provider="ollama", model="target-model",
    dimension=3, embedding_version="v1", chunking_version="v1",
)


class _FakeBuildDelegate:
    """A fake build_reindex_target() — no real Qdrant/embeddings, just records calls."""

    def __init__(self, raise_message: str | None = None) -> None:
        self.calls: list[str] = []
        self.raise_message = raise_message

    async def __call__(
        self,
        document: Document,
        session: AsyncSession,
        settings: object,
        file_storage: object,
        target_config: object,
        *,
        target_chunk_size: int,
        target_chunk_overlap: int,
    ) -> object:
        self.calls.append(document.id)
        if self.raise_message is not None:
            raise RuntimeError(self.raise_message)
        return object()


@pytest.fixture(autouse=True)
async def _clean_tables(migrated_schema: None, postgres_url: str) -> AsyncIterator[None]:
    engine = create_async_engine(postgres_url, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def _truncate() -> None:
        async with session_factory() as session:
            await session.execute(
                text(
                    "TRUNCATE TABLE reindex_jobs, index_collections, document_deletion_jobs, "
                    "ingestion_jobs, documents RESTART IDENTITY CASCADE"
                )
            )
            await session.commit()

    await _truncate()
    try:
        yield
    finally:
        await _truncate()
        await engine.dispose()


async def _seed_index_collection(session: AsyncSession, config: EmbeddingIndexConfig) -> None:
    existing = await session.get(IndexCollection, config.collection_name)
    if existing is None:
        session.add(
            IndexCollection(
                collection_name=config.collection_name,
                embedding_provider=config.provider,
                embedding_model=config.model,
                embedding_dimension=config.dimension,
                embedding_version=config.embedding_version,
                chunking_version=config.chunking_version,
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
        collection_name=_SERVING_CONFIG.collection_name,
    )
    fields.update(overrides)
    await _seed_index_collection(session, _SERVING_CONFIG)
    document = Document(**fields)  # type: ignore[arg-type]
    session.add(document)
    await session.commit()
    return document


async def _seed_reindex_job(
    session: AsyncSession, document_id: str, target_config: EmbeddingIndexConfig = _TARGET_CONFIG
) -> ReindexJob:
    await _seed_index_collection(session, target_config)
    job = ReindexJob(
        id=str(uuid.uuid4()),
        document_id=document_id,
        source_collection_name=_SERVING_CONFIG.collection_name,
        target_collection_name=target_config.collection_name,
        target_chunk_size=500,
        target_chunk_overlap=50,
        status=ReindexJobStatus.PENDING,
    )
    session.add(job)
    await session.commit()
    return job


@pytest.mark.parametrize("run", range(3))
async def test_concurrent_workers_claim_different_pending_jobs(
    run: int,
    migrated_schema: None,
    postgres_url: str,
    integration_db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two genuinely concurrent workers, two independent pending jobs -> each claims one."""
    document_a = await _seed_document(integration_db_session)
    document_b = await _seed_document(integration_db_session)
    job_a = await _seed_reindex_job(integration_db_session, document_a.id)
    job_b = await _seed_reindex_job(integration_db_session, document_b.id)

    monkeypatch.setattr(reindex_worker_module, "build_reindex_target", _FakeBuildDelegate())

    engine = create_async_engine(postgres_url, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def _claim_with_own_session() -> str | None:
        worker = ReindexWorker(file_storage=object())  # type: ignore[arg-type]
        async with session_factory() as session:
            result = await worker.process_next_job(session, get_settings())
            return result.job_id

    try:
        first_id, second_id = await asyncio.gather(
            _claim_with_own_session(), _claim_with_own_session()
        )
    finally:
        await engine.dispose()

    assert first_id is not None
    assert second_id is not None
    assert first_id != second_id
    assert {first_id, second_id} == {job_a.id, job_b.id}


@pytest.mark.parametrize("run", range(3))
async def test_concurrent_workers_cannot_claim_the_same_job(
    run: int,
    migrated_schema: None,
    postgres_url: str,
    integration_db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One pending job, two concurrent workers racing -> exactly one claims and builds it."""
    document = await _seed_document(integration_db_session)
    job = await _seed_reindex_job(integration_db_session, document.id)

    delegate = _FakeBuildDelegate()
    monkeypatch.setattr(reindex_worker_module, "build_reindex_target", delegate)

    engine = create_async_engine(postgres_url, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def _claim_with_own_session() -> str | None:
        worker = ReindexWorker(file_storage=object())  # type: ignore[arg-type]
        async with session_factory() as session:
            result = await worker.process_next_job(session, get_settings())
            return result.job_id

    try:
        first_id, second_id = await asyncio.gather(
            _claim_with_own_session(), _claim_with_own_session()
        )
    finally:
        await engine.dispose()

    claimed_ids = {job_id for job_id in (first_id, second_id) if job_id is not None}
    assert claimed_ids == {job.id}
    assert len(delegate.calls) == 1  # exactly one build invocation, not two


async def test_claimed_status_commits_before_build_execution(
    migrated_schema: None,
    postgres_url: str,
    integration_db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The PROCESSING claim must be visible to an independent connection before the build runs."""
    document = await _seed_document(integration_db_session)
    job = await _seed_reindex_job(integration_db_session, document.id)

    engine = create_async_engine(postgres_url, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def _assert_processing_from_independent_connection(*args: object, **kwargs: object) -> object:
        async with session_factory() as check_session:
            result = await check_session.execute(
                text("SELECT status FROM reindex_jobs WHERE id = :id"), {"id": job.id}
            )
            assert result.scalar_one() == "processing"
        return object()

    monkeypatch.setattr(
        reindex_worker_module, "build_reindex_target", _assert_processing_from_independent_connection
    )

    worker = ReindexWorker(file_storage=object())  # type: ignore[arg-type]
    async with session_factory() as session:
        result = await worker.process_next_job(session, get_settings())

    await engine.dispose()
    assert result.outcome == ReindexWorkerOutcome.COMPLETED


async def test_successful_execution_persists_completed_with_timestamp(
    migrated_schema: None, integration_db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = await _seed_document(integration_db_session)
    job = await _seed_reindex_job(integration_db_session, document.id)
    monkeypatch.setattr(reindex_worker_module, "build_reindex_target", _FakeBuildDelegate())

    worker = ReindexWorker(file_storage=object())  # type: ignore[arg-type]
    result = await worker.process_next_job(integration_db_session, get_settings())

    assert result.outcome == ReindexWorkerOutcome.COMPLETED
    row = await integration_db_session.execute(
        text("SELECT status, completed_at, error_message FROM reindex_jobs WHERE id = :id"),
        {"id": job.id},
    )
    record = row.one()
    assert record.status == "completed"
    assert record.completed_at is not None
    assert record.error_message is None


async def test_failed_execution_persists_failed_with_timestamp(
    migrated_schema: None, integration_db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = await _seed_document(integration_db_session)
    job = await _seed_reindex_job(integration_db_session, document.id)
    monkeypatch.setattr(
        reindex_worker_module, "build_reindex_target", _FakeBuildDelegate(raise_message="boom")
    )

    worker = ReindexWorker(file_storage=object())  # type: ignore[arg-type]
    result = await worker.process_next_job(integration_db_session, get_settings())

    assert result.outcome == ReindexWorkerOutcome.FAILED
    row = await integration_db_session.execute(
        text("SELECT status, completed_at, error_message FROM reindex_jobs WHERE id = :id"),
        {"id": job.id},
    )
    record = row.one()
    assert record.status == "failed"
    assert record.completed_at is not None
    assert record.error_message == "boom"


async def test_historical_terminal_jobs_remain_unchanged(
    migrated_schema: None, integration_db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = await _seed_document(integration_db_session)
    await _seed_index_collection(integration_db_session, _TARGET_CONFIG)
    old_completed = ReindexJob(
        id=str(uuid.uuid4()),
        document_id=document.id,
        source_collection_name=_SERVING_CONFIG.collection_name,
        target_collection_name=_TARGET_CONFIG.collection_name,
        target_chunk_size=500,
        target_chunk_overlap=50,
        status=ReindexJobStatus.COMPLETED,
    )
    integration_db_session.add(old_completed)
    await integration_db_session.commit()

    await _seed_reindex_job(integration_db_session, document.id)  # a fresh PENDING attempt
    monkeypatch.setattr(reindex_worker_module, "build_reindex_target", _FakeBuildDelegate())

    worker = ReindexWorker(file_storage=object())  # type: ignore[arg-type]
    await worker.process_next_job(integration_db_session, get_settings())

    refreshed_old = await integration_db_session.get(ReindexJob, old_completed.id)
    assert refreshed_old is not None
    assert refreshed_old.status == ReindexJobStatus.COMPLETED


async def test_worker_processes_one_job_per_call(
    migrated_schema: None, integration_db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    document_a = await _seed_document(integration_db_session)
    document_b = await _seed_document(integration_db_session)
    await _seed_reindex_job(integration_db_session, document_a.id)
    await _seed_reindex_job(integration_db_session, document_b.id)
    monkeypatch.setattr(reindex_worker_module, "build_reindex_target", _FakeBuildDelegate())

    worker = ReindexWorker(file_storage=object())  # type: ignore[arg-type]
    await worker.process_next_job(integration_db_session, get_settings())

    pending_count = await integration_db_session.execute(
        text("SELECT count(*) FROM reindex_jobs WHERE status = 'pending'")
    )
    assert pending_count.scalar_one() == 1
