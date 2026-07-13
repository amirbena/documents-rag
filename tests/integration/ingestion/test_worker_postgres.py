"""Integration tests for IngestionWorker against a real, ephemeral Postgres container.

Validates behavior that depends on genuine Postgres transaction/locking semantics —
`SELECT ... FOR UPDATE SKIP LOCKED` in particular — which SQLite does not represent correctly
even when it accepts the same SQLAlchemy call (see CLAUDE.md's database testing rules). Also
includes one end-to-end pipeline run against real Postgres + real Qdrant, using a fake
deterministic embedding provider instead of a real Ollama model.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.services.ingestion.worker as ingestion_worker_module
from app.core.config import get_settings
from app.models.document import Document
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.rag.embedding_config import get_active_embedding_config
from app.rag.providers.qdrant_vector_store import QdrantVectorStore
from app.services.ingestion.worker import IngestionWorker
from app.storage.local_storage import LocalFileStorage


@pytest.fixture(autouse=True)
async def _clean_tables(migrated_schema: None, postgres_url: str) -> AsyncIterator[None]:
    """Truncate documents/ingestion_jobs before each test so tests never see another test's rows.

    All tests in this module share one session-scoped Postgres container (starting a fresh
    container per test would be far too slow) — this keeps each test's data isolated anyway.
    """
    async with _new_session(postgres_url) as session:
        await session.execute(text('TRUNCATE TABLE ingestion_jobs, documents RESTART IDENTITY CASCADE'))
        await session.commit()
    yield


class _FakeEmbeddingProvider:
    """Returns one fixed-length deterministic vector per text — no real Ollama call."""

    def __init__(self, vector_size: int) -> None:
        self._vector = [0.1] * vector_size

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector for _ in texts]


async def _noop_process_document(
    document: Document | None, job: IngestionJob, session: AsyncSession
) -> None:
    """A trivial injectable processing step: succeeds without touching any provider."""


async def _failing_process_document(
    document: Document | None, job: IngestionJob, session: AsyncSession
) -> None:
    """A trivial injectable processing step: always raises, to exercise the failure path."""
    raise RuntimeError("simulated processing failure")


@asynccontextmanager
async def _new_session(postgres_url: str) -> AsyncIterator[AsyncSession]:
    """Open a fresh AsyncSession on its own dedicated engine/connection."""
    engine = create_async_engine(postgres_url, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with session_factory() as session:
        yield session
    await engine.dispose()


async def _create_pending_job(
    session: AsyncSession, storage_key: str = "x.txt"
) -> IngestionJob:
    document = Document(
        id=str(uuid.uuid4()),
        original_filename="notes.txt",
        stored_filename=f"{uuid.uuid4().hex}.txt",
        content_type="text/plain",
        file_size=11,
        stored_path=storage_key,
        storage_provider="local",
        storage_key=storage_key,
    )
    session.add(document)
    job = IngestionJob(id=str(uuid.uuid4()), document_id=document.id, status=IngestionStatus.PENDING)
    session.add(job)
    await session.commit()
    return job


async def test_pending_job_transitions_to_completed(migrated_schema: None, postgres_url: str) -> None:
    """A pending job processed successfully should end up completed."""
    async with _new_session(postgres_url) as session:
        job = await _create_pending_job(session)
        worker = IngestionWorker(process_document=_noop_process_document)

        result = await worker.process_next_job(session)

        assert result is not None
        assert result.id == job.id
        assert result.status == IngestionStatus.COMPLETED


async def test_processing_exception_marks_job_failed_with_error_message(
    migrated_schema: None, postgres_url: str
) -> None:
    """An exception in the processing step should mark the job failed with the error stored."""
    async with _new_session(postgres_url) as session:
        await _create_pending_job(session)
        worker = IngestionWorker(process_document=_failing_process_document)

        result = await worker.process_next_job(session)

        assert result is not None
        assert result.status == IngestionStatus.FAILED
        assert result.error_message == "simulated processing failure"


async def test_completed_and_failed_jobs_are_never_reclaimed(
    migrated_schema: None, postgres_url: str
) -> None:
    """Once a job resolves to completed/failed, it must never be selected again."""
    async with _new_session(postgres_url) as session:
        await _create_pending_job(session)
        worker = IngestionWorker(process_document=_noop_process_document)

        first = await worker.process_next_job(session)
        second = await worker.process_next_job(session)

        assert first is not None
        assert first.status == IngestionStatus.COMPLETED
        assert second is None


async def test_for_update_skip_locked_prevents_reclaim_until_lock_released(
    migrated_schema: None, postgres_url: str
) -> None:
    """A row locked by one open transaction must be skipped by another, and reclaimable after release."""
    worker = IngestionWorker(process_document=_noop_process_document)

    async with _new_session(postgres_url) as session_a:
        async with _new_session(postgres_url) as setup_session:
            job = await _create_pending_job(setup_session)

        claimed_by_a = await worker._claim_next_pending_job(session_a)
        assert claimed_by_a is not None
        assert claimed_by_a.id == job.id

        async with _new_session(postgres_url) as session_b:
            claimed_by_b = await worker._claim_next_pending_job(session_b)
            assert claimed_by_b is None, "a locked row must be skipped, not returned"

        await session_a.rollback()

    async with _new_session(postgres_url) as session_c:
        claimed_by_c = await worker._claim_next_pending_job(session_c)
        assert claimed_by_c is not None
        assert claimed_by_c.id == job.id, "releasing the lock should make the row claimable again"


async def test_two_concurrent_workers_cannot_claim_the_same_pending_job(
    migrated_schema: None, postgres_url: str
) -> None:
    """Two workers racing on process_next_job() with one pending job must never both claim it."""

    async def _claim_and_process() -> IngestionJob | None:
        async with _new_session(postgres_url) as session:
            worker = IngestionWorker(process_document=_noop_process_document)
            return await worker.process_next_job(session)

    async with _new_session(postgres_url) as setup_session:
        job = await _create_pending_job(setup_session)

    results = await asyncio.gather(_claim_and_process(), _claim_and_process())

    claimed = [result for result in results if result is not None]
    assert len(claimed) == 1
    assert claimed[0].id == job.id
    assert claimed[0].status == IngestionStatus.COMPLETED
    assert results.count(None) == 1


async def test_default_pipeline_against_real_postgres_and_qdrant_with_fake_embeddings(
    migrated_schema: None, postgres_url: str, qdrant_url: str, tmp_path: Path, monkeypatch
) -> None:
    """The real default pipeline should run end-to-end against real Postgres + real Qdrant.

    Uses a fake, deterministic embedding provider instead of a real Ollama model — this suite
    never pulls or calls a real LLM/embedding model.
    """
    settings = get_settings()
    collection_prefix = f"integration-worker-{uuid.uuid4().hex}"
    monkeypatch.setattr(settings, "qdrant_collection_name", collection_prefix)
    monkeypatch.setattr(
        ingestion_worker_module,
        "get_embedding_provider",
        lambda settings=None: _FakeEmbeddingProvider(get_settings().vector_size),
    )
    active_config = get_active_embedding_config(settings)

    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world " * 100, encoding="utf-8")

    async with _new_session(postgres_url) as session:
        await _create_pending_job(session, storage_key=file_path.name)
        worker = IngestionWorker(file_storage=LocalFileStorage(root=tmp_path))

        result = await worker.process_next_job(session)

        assert result is not None
        assert result.status == IngestionStatus.COMPLETED

        document = await session.get(Document, result.document_id)
        assert document is not None
        assert document.collection_name == active_config.collection_name
        assert document.indexed_at is not None

    vector_store = QdrantVectorStore(settings=settings)
    query_vector = [0.1] * settings.vector_size
    results = await vector_store.search_similar(active_config.collection_name, query_vector, limit=10)

    assert len(results) > 0
    assert all(result.document_id == document.id for result in results)
    assert all(result.text.strip() for result in results)


async def test_only_pending_jobs_are_selected(migrated_schema: None, postgres_url: str) -> None:
    """A completed job created before a pending one must not be re-selected instead of it."""
    async with _new_session(postgres_url) as session:
        completed_job = await _create_pending_job(session)
        completed_job.status = IngestionStatus.COMPLETED
        await session.commit()

        pending_job = await _create_pending_job(session)

        worker = IngestionWorker(process_document=_noop_process_document)
        result = await worker.process_next_job(session)

        assert result is not None
        assert result.id == pending_job.id

        refreshed_completed = await session.execute(
            select(IngestionJob).where(IngestionJob.id == completed_job.id)
        )
        assert refreshed_completed.scalar_one().status == IngestionStatus.COMPLETED
