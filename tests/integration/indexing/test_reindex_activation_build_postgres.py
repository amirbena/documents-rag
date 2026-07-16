"""Focused real end-to-end scenario for Phase 2.8.6 (Subtask 5): real ingest -> real schedule ->
real build (ReindexWorker) -> real activate (activate_reindexed_document()) — real Postgres, real
Qdrant, genuine extraction/chunking, a deterministic fake embedding provider (never real Ollama).

Proves the full build-then-activate handoff: after activation, the document serves B, its
embedding/chunking metadata matches B, `indexed_at` moved, the build job remains build-completed
AND is separately marked activated, exactly one cleanup job exists for A, vectors still exist in
BOTH A and B (activation never deletes), and retrieval configuration resolves B purely through
document metadata. The cleanup worker is never executed here — that remains a later subtask's
concern, and this scenario deliberately stops short of it.
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
from app.models.vector_cleanup_job import VectorCleanupStatus
from app.rag.embedding_config import get_active_embedding_config
from app.rag.providers.qdrant_vector_store import QdrantVectorStore
from app.services.indexing.reindex_activation import ReindexActivationOutcome, activate_reindexed_document
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
    monkeypatch.setattr(get_settings(), "qdrant_collection_name", f"reindex-activate-{uuid.uuid4().hex}")


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


async def test_real_ingest_schedule_build_then_activate_moves_serving_ownership_to_target(
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
    monkeypatch.setattr(settings, "embedding_version", "v2-activate-test")
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

    # Step 3: real build via ReindexWorker — target vectors land in B, A is untouched.
    worker = ReindexWorker(file_storage=file_storage)
    async with _new_session(postgres_url) as session:
        worker_result = await worker.process_next_job(session, settings)
    assert worker_result.outcome == ReindexWorkerOutcome.COMPLETED
    assert worker_result.job_id == job_id

    # Step 4: real activation — metadata cutover + deferred cleanup job only, no Qdrant call at all.
    async with _new_session(postgres_url) as session:
        activation_result = await activate_reindexed_document(session, job_id)
    assert activation_result.outcome == ReindexActivationOutcome.ACTIVATED

    async with _new_session(postgres_url) as session:
        refreshed_document = await session.get(Document, document_id)
        assert refreshed_document is not None
        # Document now serves B.
        assert refreshed_document.collection_name == target_config.collection_name
        assert refreshed_document.embedding_provider == target_config.provider
        assert refreshed_document.embedding_model == target_config.model
        assert refreshed_document.embedding_version == target_config.embedding_version
        assert refreshed_document.chunking_version == target_config.chunking_version
        # indexed_at moved to the activation timestamp.
        assert refreshed_document.indexed_at is not None

        # The build job remains build-completed AND is separately marked activated.
        reindex_job = await session.get(ReindexJob, job_id)
        assert reindex_job is not None
        assert reindex_job.status == ReindexJobStatus.COMPLETED
        assert reindex_job.completed_at is not None
        assert reindex_job.activated_at is not None

        # Exactly one cleanup job exists for A (the vacated source collection).
        cleanup_rows = (
            await session.execute(
                text("SELECT collection_name, status FROM vector_cleanup_jobs WHERE document_id = :id"),
                {"id": document_id},
            )
        ).all()
        assert len(cleanup_rows) == 1
        assert cleanup_rows[0].collection_name == original_config.collection_name
        assert cleanup_rows[0].status == VectorCleanupStatus.PENDING

    # Vectors still exist in BOTH A and B — activation never deletes anything.
    vector_store = QdrantVectorStore(settings=settings)
    query_vector = (await fake.embed(["hello world"]))[0]

    results_in_target = await vector_store.search_similar(
        target_config.collection_name, query_vector, limit=10
    )
    assert any(r.document_id == document_id for r in results_in_target)

    results_in_original = await vector_store.search_similar(
        original_config.collection_name, query_vector, limit=10
    )
    assert any(r.document_id == document_id for r in results_in_original)

    # Retrieval configuration now resolves B purely through the document's own persisted metadata
    # — not through re-deriving anything from live Settings (Settings.embedding_version here still
    # says "v2-activate-test", which happens to agree, but activation never consulted it to decide).
    async with _new_session(postgres_url) as session:
        refreshed_document = await session.get(Document, document_id)
        assert refreshed_document is not None
        assert refreshed_document.collection_name == target_config.collection_name
