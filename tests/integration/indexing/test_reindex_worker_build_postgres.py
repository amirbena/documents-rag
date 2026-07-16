"""Focused real-build integration test for ReindexWorker — real Postgres, real Qdrant, genuine
extraction/chunking, a deterministic fake embedding provider (never real Ollama).

One scenario: a document already indexed in collection A; a ReindexJob (created via the real
schedule_reindex()) targets collection B; ReindexWorker.process_next_job() builds it. Proves
vectors land in B, remain in A, and nothing about the document's serving identity, indexing
metadata, cleanup obligations, or retrieval configuration changes — see reindex_worker.py's module
docstring for why COMPLETED means only "the target build succeeded," never "the target is active."
"""

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.services.indexing.reindex_service as reindex_service_module
import app.services.ingestion.worker as ingestion_worker_module
from app.core.config import get_settings
from app.models.document import Document
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.models.reindex_job import ReindexJob, ReindexJobStatus
from app.rag.embedding_config import get_active_embedding_config
from app.rag.providers.qdrant_vector_store import QdrantVectorStore
from app.services.indexing.reindex_scheduling_service import ReindexSchedulingOutcome, schedule_reindex
from app.services.indexing.reindex_worker import ReindexWorker, ReindexWorkerOutcome
from app.services.ingestion.worker import IngestionWorker
from app.storage.local_storage import LocalFileStorage
from tests.multilingual_fixtures import MultilingualFakeEmbeddingProvider


@pytest.fixture(autouse=True)
async def _clean_tables(migrated_schema: None, postgres_url: str) -> AsyncIterator[None]:
    engine = create_async_engine(postgres_url, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def _truncate() -> None:
        async with session_factory() as session:
            await session.execute(
                text(
                    "TRUNCATE TABLE reindex_jobs, vector_cleanup_jobs, index_collections, "
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


@pytest.fixture(autouse=True)
def _unique_collection_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give this test its own Qdrant collection prefix — mirrors test_multilingual_indexing.py."""
    monkeypatch.setattr(get_settings(), "qdrant_collection_name", f"reindex-worker-{uuid.uuid4().hex}")


@asynccontextmanager
async def _new_session(postgres_url: str) -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(postgres_url, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with session_factory() as session:
        yield session
    await engine.dispose()


async def _create_pending_ingestion(session: AsyncSession, storage_key: str) -> IngestionJob:
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


async def test_worker_builds_target_while_serving_collection_and_metadata_stay_untouched(
    migrated_schema: None,
    postgres_url: str,
    qdrant_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = get_settings()
    fake = MultilingualFakeEmbeddingProvider(vector_size=settings.vector_size)
    monkeypatch.setattr(ingestion_worker_module, "get_embedding_provider", lambda settings=None: fake)

    file_storage = LocalFileStorage(root=tmp_path)
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world budget review " * 10, encoding="utf-8")

    # Step 1: real initial ingestion into collection A.
    async with _new_session(postgres_url) as session:
        job = await _create_pending_ingestion(session, file_path.name)
        result = await IngestionWorker(file_storage=file_storage).process_next_job(session)
        assert result is not None
        assert result.status == IngestionStatus.COMPLETED
        document_id = job.document_id

    original_config = get_active_embedding_config(settings)

    # Step 2: bump the embedding version -> a distinct target collection B -> schedule the build.
    monkeypatch.setattr(settings, "embedding_version", "v2-worker-test")
    target_config = get_active_embedding_config(settings)
    assert target_config.collection_name != original_config.collection_name
    monkeypatch.setattr(reindex_service_module, "get_embedding_provider", lambda settings=None: fake)

    async with _new_session(postgres_url) as session:
        document = await session.get(Document, document_id)
        assert document is not None
        vector_store = QdrantVectorStore(settings=settings)
        scheduling_result = await schedule_reindex(
            session,
            document,
            vector_store,
            target_config,
            target_chunk_size=settings.chunk_size,
            target_chunk_overlap=settings.chunk_overlap,
        )
        assert scheduling_result.outcome == ReindexSchedulingOutcome.CREATED
        assert scheduling_result.job is not None
        job_id = scheduling_result.job.id

    # Step 3: real build via ReindexWorker — genuine extraction/chunking, fake embedding, real Qdrant.
    worker = ReindexWorker(file_storage=file_storage)
    async with _new_session(postgres_url) as session:
        worker_result = await worker.process_next_job(session, settings)

    assert worker_result.outcome == ReindexWorkerOutcome.COMPLETED
    assert worker_result.job_id == job_id

    vector_store = QdrantVectorStore(settings=settings)
    query_vector = (await fake.embed(["hello world"]))[0]

    # Vectors are written to B (the target).
    results_in_target = await vector_store.search_similar(
        target_config.collection_name, query_vector, limit=10
    )
    assert any(r.document_id == document_id for r in results_in_target)

    # Vectors remain in A (the still-serving collection) — untouched by the build.
    results_in_original = await vector_store.search_similar(
        original_config.collection_name, query_vector, limit=10
    )
    assert any(r.document_id == document_id for r in results_in_original)

    async with _new_session(postgres_url) as session:
        refreshed_document = await session.get(Document, document_id)
        assert refreshed_document is not None
        # Document.collection_name remains A — the build never activates anything.
        assert refreshed_document.collection_name == original_config.collection_name
        # Document embedding metadata remains unchanged.
        assert refreshed_document.embedding_provider == original_config.provider
        assert refreshed_document.embedding_model == original_config.model
        assert refreshed_document.embedding_version == original_config.embedding_version
        assert refreshed_document.chunking_version == original_config.chunking_version

        # ReindexJob became COMPLETED.
        reindex_job = await session.get(ReindexJob, job_id)
        assert reindex_job is not None
        assert reindex_job.status == ReindexJobStatus.COMPLETED
        assert reindex_job.completed_at is not None
        assert reindex_job.error_message is None

        # No cleanup job for A (or anything else) exists — the worker never schedules cleanup.
        cleanup_count = await session.execute(
            text("SELECT count(*) FROM vector_cleanup_jobs WHERE document_id = :id"), {"id": document_id}
        )
        assert cleanup_count.scalar_one() == 0

    # Retrieval configuration remains unchanged by the build itself — the worker operates on an
    # isolated target-scoped Settings copy internally (Subtask 1) and never mutates the shared
    # `settings` object it was given; whatever it resolves to here is exactly what this test set,
    # not something the worker silently altered.
    assert get_active_embedding_config(settings).collection_name == target_config.collection_name
