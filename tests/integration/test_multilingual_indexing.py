"""Integration tests for the multilingual embedding/index foundation against real, ephemeral
Postgres and Qdrant containers.

Uses MultilingualFakeEmbeddingProvider (tests/multilingual_fixtures.py) instead of a real Ollama
model — this suite never pulls or calls a real embedding/LLM model.
"""

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.services.ingestion_worker as ingestion_worker_module
import app.services.reindex_service as reindex_service_module
from app.core.config import get_settings
from app.models.document import Document
from app.models.index_collection import IndexCollection
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.rag.embedding_config import get_active_embedding_config
from app.rag.providers.qdrant_vector_store import QdrantVectorStore
from app.rag.providers.vector_store import VectorPoint
from app.services.index_registry import (
    create_cleanup_job,
    delete_all_tracked_document_vectors,
    ensure_active_collection,
    get_pending_cleanup_jobs,
    is_document_stale,
    retry_cleanup_job,
)
from app.services.ingestion_worker import IngestionWorker
from app.services.reindex_service import ReindexOutcome, reindex_document
from app.storage.local_storage import LocalFileStorage
from tests.multilingual_fixtures import MIXED_TECHNICAL_DOCUMENT, MultilingualFakeEmbeddingProvider


@pytest.fixture(autouse=True)
async def _clean_tables(migrated_schema: None, postgres_url: str) -> AsyncIterator[None]:
    """Truncate documents/ingestion_jobs/index_collections before each test."""
    async with _new_session(postgres_url) as session:
        await session.execute(
            text(
                "TRUNCATE TABLE ingestion_jobs, documents, index_collections, vector_cleanup_jobs "
                "RESTART IDENTITY CASCADE"
            )
        )
        await session.commit()
    yield


@pytest.fixture(autouse=True)
def _unique_collection_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test its own Qdrant collection prefix.

    All tests in this module share one session-scoped Qdrant container — without this, every
    test resolving get_active_embedding_config() from the same default settings would derive the
    exact same collection name, and Qdrant collections (unlike Postgres rows) are never
    truncated between tests, so state would leak across tests within this file.
    """
    monkeypatch.setattr(get_settings(), "qdrant_collection_name", f"ml-test-{uuid.uuid4().hex}")


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


def _use_multilingual_fake_embeddings(monkeypatch, module) -> MultilingualFakeEmbeddingProvider:
    settings = get_settings()
    fake = MultilingualFakeEmbeddingProvider(vector_size=settings.vector_size)
    monkeypatch.setattr(module, "get_embedding_provider", lambda settings=None: fake)
    return fake


async def test_document_indexing_persists_embedding_metadata(
    migrated_schema: None, postgres_url: str, qdrant_url: str, tmp_path, monkeypatch
) -> None:
    """A successfully ingested document must have every indexing-metadata column populated."""
    _use_multilingual_fake_embeddings(monkeypatch, ingestion_worker_module)
    active_config = get_active_embedding_config()

    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world " * 20, encoding="utf-8")

    async with _new_session(postgres_url) as session:
        job = await _create_pending_job(session, storage_key=file_path.name)
        worker = IngestionWorker(file_storage=LocalFileStorage(root=tmp_path))
        result = await worker.process_next_job(session)

        assert result is not None
        assert result.status == IngestionStatus.COMPLETED

        document = await session.get(Document, job.document_id)
        assert document is not None
        assert document.embedding_provider == active_config.provider
        assert document.embedding_model == active_config.model
        assert document.embedding_dimension == active_config.dimension
        assert document.embedding_version == active_config.embedding_version
        assert document.chunking_version == active_config.chunking_version
        assert document.indexed_at is not None


async def test_active_collection_name_is_persisted_and_tracked(
    migrated_schema: None, postgres_url: str, qdrant_url: str, tmp_path, monkeypatch
) -> None:
    """The document's collection_name must match the active config, and be tracked in Postgres."""
    _use_multilingual_fake_embeddings(monkeypatch, ingestion_worker_module)
    active_config = get_active_embedding_config()

    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world " * 20, encoding="utf-8")

    async with _new_session(postgres_url) as session:
        await _create_pending_job(session, storage_key=file_path.name)
        await IngestionWorker(file_storage=LocalFileStorage(root=tmp_path)).process_next_job(session)

        document = (await session.execute(select(Document))).scalar_one()
        assert document.collection_name == active_config.collection_name

        tracked = await session.get(IndexCollection, active_config.collection_name)
        assert tracked is not None
        assert tracked.embedding_dimension == active_config.dimension


async def test_dimension_mismatch_is_rejected_against_real_qdrant(
    migrated_schema: None, postgres_url: str, qdrant_url: str, monkeypatch
) -> None:
    """A real, existing Qdrant collection with the wrong dimension must be rejected, not reused."""
    from app.services.index_registry import IncompatibleIndexConfigurationError

    settings = get_settings()
    active_config = get_active_embedding_config(settings)
    vector_store = QdrantVectorStore(settings=settings)

    # Pre-create the *exact* target collection name with a deliberately wrong dimension.
    await vector_store.create_collection_if_not_exists(
        active_config.collection_name, active_config.dimension + 1
    )

    async with _new_session(postgres_url) as session:
        with pytest.raises(IncompatibleIndexConfigurationError):
            await ensure_active_collection(vector_store, session, active_config)


async def test_stale_document_detection_after_embedding_version_change(
    migrated_schema: None, postgres_url: str, qdrant_url: str, tmp_path, monkeypatch
) -> None:
    """Bumping EMBEDDING_VERSION must make a previously-current document report stale."""
    _use_multilingual_fake_embeddings(monkeypatch, ingestion_worker_module)
    settings = get_settings()

    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world " * 20, encoding="utf-8")

    async with _new_session(postgres_url) as session:
        await _create_pending_job(session, storage_key=file_path.name)
        await IngestionWorker(file_storage=LocalFileStorage(root=tmp_path)).process_next_job(session)
        document = (await session.execute(select(Document))).scalar_one()

    original_config = get_active_embedding_config(settings)
    assert is_document_stale(document, original_config) is False

    monkeypatch.setattr(settings, "embedding_version", "v2-multilingual")
    bumped_config = get_active_embedding_config(settings)

    assert bumped_config.collection_name != original_config.collection_name
    assert is_document_stale(document, bumped_config) is True


async def test_reindex_writes_to_the_new_collection_and_updates_metadata(
    migrated_schema: None, postgres_url: str, qdrant_url: str, tmp_path, monkeypatch
) -> None:
    """A successful re-index after a version bump must write into the new collection."""
    _use_multilingual_fake_embeddings(monkeypatch, ingestion_worker_module)
    settings = get_settings()

    file_path = tmp_path / "notes.txt"
    file_path.write_text(MIXED_TECHNICAL_DOCUMENT.decode("utf-8"), encoding="utf-8")

    async with _new_session(postgres_url) as session:
        await _create_pending_job(session, storage_key=file_path.name)
        await IngestionWorker(file_storage=LocalFileStorage(root=tmp_path)).process_next_job(session)

    monkeypatch.setattr(settings, "embedding_version", "v2-multilingual")
    new_config = get_active_embedding_config(settings)
    _use_multilingual_fake_embeddings(monkeypatch, reindex_service_module)

    async with _new_session(postgres_url) as session:
        document = (await session.execute(select(Document))).scalar_one()
        assert is_document_stale(document, new_config) is True

        result = await reindex_document(
            document, session, settings, file_storage=LocalFileStorage(root=tmp_path)
        )

        assert result.outcome == ReindexOutcome.REINDEXED
        assert document.collection_name == new_config.collection_name
        assert document.embedding_version == "v2-multilingual"
        assert is_document_stale(document, new_config) is False

    vector_store = QdrantVectorStore(settings=settings)
    fake = MultilingualFakeEmbeddingProvider(vector_size=settings.vector_size)
    query_vector = (await fake.embed(["kafka kubernetes"]))[0]
    results = await vector_store.search_similar(new_config.collection_name, query_vector, limit=10)
    assert len(results) > 0


async def test_failed_reindex_does_not_mark_document_current(
    migrated_schema: None, postgres_url: str, qdrant_url: str, tmp_path, monkeypatch
) -> None:
    """A re-index failure must leave the document's stored indexing metadata untouched."""
    _use_multilingual_fake_embeddings(monkeypatch, ingestion_worker_module)
    settings = get_settings()

    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world " * 20, encoding="utf-8")

    async with _new_session(postgres_url) as session:
        await _create_pending_job(session, storage_key=file_path.name)
        await IngestionWorker(file_storage=LocalFileStorage(root=tmp_path)).process_next_job(session)

    monkeypatch.setattr(settings, "embedding_version", "v2-broken")
    new_config = get_active_embedding_config(settings)

    class _FailingEmbeddingProvider:
        async def embed(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("embedding provider unavailable")

    monkeypatch.setattr(
        reindex_service_module, "get_embedding_provider", lambda settings=None: _FailingEmbeddingProvider()
    )

    async with _new_session(postgres_url) as session:
        document = (await session.execute(select(Document))).scalar_one()
        original_collection_name = document.collection_name

        with pytest.raises(RuntimeError, match="embedding provider unavailable"):
            await reindex_document(document, session, settings, file_storage=LocalFileStorage(root=tmp_path))

        assert document.collection_name == original_collection_name
        assert is_document_stale(document, new_config) is True


async def test_reindex_cleanup_failure_persists_job_and_retry_succeeds_against_real_qdrant(
    migrated_schema: None, postgres_url: str, qdrant_url: str, tmp_path, monkeypatch
) -> None:
    """A legacy-collection delete failure is persisted, retryable, and idempotent against real Qdrant."""
    _use_multilingual_fake_embeddings(monkeypatch, ingestion_worker_module)
    settings = get_settings()

    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world " * 20, encoding="utf-8")

    async with _new_session(postgres_url) as session:
        await _create_pending_job(session, storage_key=file_path.name)
        await IngestionWorker(file_storage=LocalFileStorage(root=tmp_path)).process_next_job(session)

    original_config = get_active_embedding_config(settings)
    monkeypatch.setattr(settings, "embedding_version", "v2-multilingual")
    new_config = get_active_embedding_config(settings)
    _use_multilingual_fake_embeddings(monkeypatch, reindex_service_module)

    real_vector_store = QdrantVectorStore(settings=settings)

    class _FailOnceForOldCollection:
        """Wraps the real QdrantVectorStore, simulating one transient delete failure."""

        def __init__(self) -> None:
            self.delete_calls: list[str] = []

        async def create_collection_if_not_exists(self, collection_name: str, vector_size: int) -> None:
            await real_vector_store.create_collection_if_not_exists(collection_name, vector_size)

        async def upsert_vectors(self, collection_name: str, points: list) -> None:
            await real_vector_store.upsert_vectors(collection_name, points)

        async def get_collection_vector_size(self, collection_name: str) -> int | None:
            return await real_vector_store.get_collection_vector_size(collection_name)

        async def delete_by_document_id(self, collection_name: str, document_id: str) -> None:
            self.delete_calls.append(collection_name)
            if collection_name == original_config.collection_name:
                raise RuntimeError("simulated transient Qdrant failure")
            await real_vector_store.delete_by_document_id(collection_name, document_id)

    failing_store = _FailOnceForOldCollection()
    monkeypatch.setattr(reindex_service_module, "get_vector_store", lambda settings=None: failing_store)

    async with _new_session(postgres_url) as session:
        document = (await session.execute(select(Document))).scalar_one()
        result = await reindex_document(
            document, session, settings, file_storage=LocalFileStorage(root=tmp_path)
        )
        assert result.outcome == ReindexOutcome.REINDEXED_WITH_CLEANUP_PENDING
        assert document.collection_name == new_config.collection_name

    async with _new_session(postgres_url) as session:
        jobs = await get_pending_cleanup_jobs(session, document_id=document.id)
        assert len(jobs) == 1
        assert jobs[0].collection_name == original_config.collection_name
        assert jobs[0].attempts == 1

        succeeded = await retry_cleanup_job(session, real_vector_store, jobs[0])
        assert succeeded is True

        # Cleanup retry is retried even though the document itself is already current.
        assert is_document_stale(document, new_config) is False

    fake = MultilingualFakeEmbeddingProvider(vector_size=settings.vector_size)
    query_vector = (await fake.embed(["hello world"]))[0]
    remaining = await real_vector_store.search_similar(
        original_config.collection_name, query_vector, limit=10
    )
    assert not any(result.document_id == document.id for result in remaining)

    async with _new_session(postgres_url) as session:
        jobs_after_retry = await get_pending_cleanup_jobs(session, document_id=document.id)
        assert jobs_after_retry == []  # the completed job no longer shows up as pending/failed


async def test_full_document_deletion_cleans_pending_historical_collection_against_real_qdrant(
    migrated_schema: None, postgres_url: str, qdrant_url: str, tmp_path, monkeypatch
) -> None:
    """delete_all_tracked_document_vectors() must clean an outstanding pending legacy collection
    too, not just the document's current one — even without an explicit retry beforehand.
    """
    _use_multilingual_fake_embeddings(monkeypatch, ingestion_worker_module)
    settings = get_settings()

    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world " * 20, encoding="utf-8")

    async with _new_session(postgres_url) as session:
        await _create_pending_job(session, storage_key=file_path.name)
        await IngestionWorker(file_storage=LocalFileStorage(root=tmp_path)).process_next_job(session)

    original_config = get_active_embedding_config(settings)
    monkeypatch.setattr(settings, "embedding_version", "v3-multilingual")
    new_config = get_active_embedding_config(settings)
    _use_multilingual_fake_embeddings(monkeypatch, reindex_service_module)

    real_vector_store = QdrantVectorStore(settings=settings)

    class _AlwaysFailForOldCollection:
        async def create_collection_if_not_exists(self, collection_name: str, vector_size: int) -> None:
            await real_vector_store.create_collection_if_not_exists(collection_name, vector_size)

        async def upsert_vectors(self, collection_name: str, points: list) -> None:
            await real_vector_store.upsert_vectors(collection_name, points)

        async def get_collection_vector_size(self, collection_name: str) -> int | None:
            return await real_vector_store.get_collection_vector_size(collection_name)

        async def delete_by_document_id(self, collection_name: str, document_id: str) -> None:
            if collection_name == original_config.collection_name:
                raise RuntimeError("simulated persistent Qdrant failure")
            await real_vector_store.delete_by_document_id(collection_name, document_id)

    monkeypatch.setattr(
        reindex_service_module, "get_vector_store", lambda settings=None: _AlwaysFailForOldCollection()
    )

    async with _new_session(postgres_url) as session:
        document = (await session.execute(select(Document))).scalar_one()
        result = await reindex_document(
            document, session, settings, file_storage=LocalFileStorage(root=tmp_path)
        )
        assert result.outcome == ReindexOutcome.REINDEXED_WITH_CLEANUP_PENDING

    async with _new_session(postgres_url) as session:
        jobs = await get_pending_cleanup_jobs(session, document_id=document.id)
        assert len(jobs) == 1
        assert jobs[0].collection_name == original_config.collection_name

        # No retry attempted — go straight to full document deletion.
        await delete_all_tracked_document_vectors(document, real_vector_store, session)

    fake = MultilingualFakeEmbeddingProvider(vector_size=settings.vector_size)
    query_vector = (await fake.embed(["hello world"]))[0]

    remaining_in_old = await real_vector_store.search_similar(
        original_config.collection_name, query_vector, limit=10
    )
    assert not any(result.document_id == document.id for result in remaining_in_old)

    remaining_in_new = await real_vector_store.search_similar(
        new_config.collection_name, query_vector, limit=10
    )
    assert not any(result.document_id == document.id for result in remaining_in_new)


async def test_deleting_a_document_cleans_its_tracked_vectors(
    migrated_schema: None, postgres_url: str, qdrant_url: str, tmp_path, monkeypatch
) -> None:
    """Full document deletion (delete_all_tracked_document_vectors) removes its real Qdrant vectors."""
    fake = _use_multilingual_fake_embeddings(monkeypatch, ingestion_worker_module)
    settings = get_settings()
    active_config = get_active_embedding_config(settings)

    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world " * 20, encoding="utf-8")

    async with _new_session(postgres_url) as session:
        job = await _create_pending_job(session, storage_key=file_path.name)
        await IngestionWorker(file_storage=LocalFileStorage(root=tmp_path)).process_next_job(session)
        document = await session.get(Document, job.document_id)
        assert document is not None

    vector_store = QdrantVectorStore(settings=settings)
    query_vector = (await fake.embed(["hello world"]))[0]
    before = await vector_store.search_similar(active_config.collection_name, query_vector, limit=10)
    assert any(result.document_id == document.id for result in before)

    async with _new_session(postgres_url) as session:
        await delete_all_tracked_document_vectors(document, vector_store, session)

    after = await vector_store.search_similar(active_config.collection_name, query_vector, limit=10)
    assert not any(result.document_id == document.id for result in after)


def _fake_point(document_id: str, vector: list[float], text: str = "hello world") -> VectorPoint:
    """Build a real VectorPoint for a document, ready to upsert directly into any collection."""
    return VectorPoint(
        id=str(uuid.uuid4()),
        vector=vector,
        document_id=document_id,
        chunk_id=str(uuid.uuid4()),
        text=text,
        source="notes.txt",
    )


async def test_full_document_deletion_cleans_active_and_historical_without_touching_other_documents(
    migrated_schema: None, postgres_url: str, qdrant_url: str, tmp_path, monkeypatch
) -> None:
    """delete_all_tracked_document_vectors() must clean a document's vectors out of its active
    collection AND both of its tracked historical collections (one pending job, one failed job)
    against real Qdrant, while leaving a different document's vectors in a shared historical
    collection completely untouched.
    """
    fake = _use_multilingual_fake_embeddings(monkeypatch, ingestion_worker_module)
    settings = get_settings()
    active_config = get_active_embedding_config(settings)
    vector_store = QdrantVectorStore(settings=settings)

    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world " * 20, encoding="utf-8")

    async with _new_session(postgres_url) as session:
        job = await _create_pending_job(session, storage_key=file_path.name)
        await IngestionWorker(file_storage=LocalFileStorage(root=tmp_path)).process_next_job(session)
        document_a = await session.get(Document, job.document_id)
        assert document_a is not None

        other_job = await _create_pending_job(session, storage_key=file_path.name)
        document_b = await session.get(Document, other_job.document_id)
        assert document_b is not None

    query_vector = (await fake.embed(["hello world"]))[0]

    # Two historical collections for document A, seeded with real points directly (no need to
    # actually drive a re-index through them). One is tracked as PENDING, the other as FAILED.
    historical_collection_1 = f"{active_config.collection_name}-legacy-1"
    historical_collection_2 = f"{active_config.collection_name}-legacy-2"
    await vector_store.create_collection_if_not_exists(historical_collection_1, active_config.dimension)
    await vector_store.create_collection_if_not_exists(historical_collection_2, active_config.dimension)
    await vector_store.upsert_vectors(historical_collection_1, [_fake_point(document_a.id, query_vector)])
    # historical_collection_2 also holds a *different* document's vectors — this must survive.
    await vector_store.upsert_vectors(
        historical_collection_2,
        [_fake_point(document_a.id, query_vector), _fake_point(document_b.id, query_vector)],
    )

    async with _new_session(postgres_url) as session:
        await create_cleanup_job(session, document_a.id, historical_collection_1)  # PENDING
        await create_cleanup_job(session, document_a.id, historical_collection_2, error="boom")  # FAILED

        result = await delete_all_tracked_document_vectors(document_a, vector_store, session)

    assert result.fully_deleted is True
    assert set(result.attempted_collections) == {
        active_config.collection_name,
        historical_collection_1,
        historical_collection_2,
    }

    for collection_name in (active_config.collection_name, historical_collection_1, historical_collection_2):
        remaining = await vector_store.search_similar(collection_name, query_vector, limit=10)
        assert not any(r.document_id == document_a.id for r in remaining)

    # Document B's vectors in the shared historical collection must be untouched.
    remaining_in_shared = await vector_store.search_similar(historical_collection_2, query_vector, limit=10)
    assert any(r.document_id == document_b.id for r in remaining_in_shared)


async def test_partial_failure_still_cleans_the_collections_that_succeed_against_real_qdrant(
    migrated_schema: None, postgres_url: str, qdrant_url: str, tmp_path, monkeypatch
) -> None:
    """When one tracked collection's delete genuinely fails, delete_all_tracked_document_vectors()
    must still clean the other collections (verified via real search_similar) and report the
    partial outcome via the typed result — reusing the existing monkeypatched-real-store-except-
    one-collection pattern rather than inventing a new failure simulation.
    """
    fake = _use_multilingual_fake_embeddings(monkeypatch, ingestion_worker_module)
    settings = get_settings()
    active_config = get_active_embedding_config(settings)
    real_vector_store = QdrantVectorStore(settings=settings)

    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world " * 20, encoding="utf-8")

    async with _new_session(postgres_url) as session:
        job = await _create_pending_job(session, storage_key=file_path.name)
        await IngestionWorker(file_storage=LocalFileStorage(root=tmp_path)).process_next_job(session)
        document = await session.get(Document, job.document_id)
        assert document is not None

    query_vector = (await fake.embed(["hello world"]))[0]

    historical_collection = f"{active_config.collection_name}-legacy"
    await real_vector_store.create_collection_if_not_exists(historical_collection, active_config.dimension)
    await real_vector_store.upsert_vectors(historical_collection, [_fake_point(document.id, query_vector)])

    class _FailForHistoricalCollection:
        async def delete_by_document_id(self, collection_name: str, document_id: str) -> None:
            if collection_name == historical_collection:
                raise RuntimeError("simulated persistent Qdrant failure")
            await real_vector_store.delete_by_document_id(collection_name, document_id)

    partially_failing_store = _FailForHistoricalCollection()

    async with _new_session(postgres_url) as session:
        await create_cleanup_job(session, document.id, historical_collection, error="boom")

        result = await delete_all_tracked_document_vectors(document, partially_failing_store, session)  # type: ignore[arg-type]

    assert result.fully_deleted is False
    by_name = {r.collection_name: r for r in result.collection_results}
    assert by_name[active_config.collection_name].succeeded is True
    assert by_name[historical_collection].succeeded is False
    assert by_name[historical_collection].error is not None

    # The active collection's deletion genuinely happened against real Qdrant despite the other
    # collection's failure.
    remaining_active = await real_vector_store.search_similar(
        active_config.collection_name, query_vector, limit=10
    )
    assert not any(r.document_id == document.id for r in remaining_active)

    # The historical collection's vectors are still there — the simulated failure was real.
    remaining_historical = await real_vector_store.search_similar(
        historical_collection, query_vector, limit=10
    )
    assert any(r.document_id == document.id for r in remaining_historical)


async def test_mixed_hebrew_english_text_survives_persistence_and_retrieval(
    migrated_schema: None, postgres_url: str, qdrant_url: str, tmp_path, monkeypatch
) -> None:
    """A mixed Hebrew/English document's text must round-trip through real Qdrant unmangled."""
    fake = _use_multilingual_fake_embeddings(monkeypatch, ingestion_worker_module)
    settings = get_settings()
    active_config = get_active_embedding_config(settings)

    file_path = tmp_path / "mixed.txt"
    mixed_text = MIXED_TECHNICAL_DOCUMENT.decode("utf-8")
    file_path.write_text(mixed_text, encoding="utf-8")

    async with _new_session(postgres_url) as session:
        job = await _create_pending_job(session, storage_key=file_path.name)
        result = await IngestionWorker(file_storage=LocalFileStorage(root=tmp_path)).process_next_job(session)
        assert result is not None
        assert result.status == IngestionStatus.COMPLETED

    vector_store = QdrantVectorStore(settings=settings)
    query_vector = (await fake.embed(["kafka kubernetes qdrant langchain"]))[0]
    results = await vector_store.search_similar(active_config.collection_name, query_vector, limit=10)

    matching = [result for result in results if result.document_id == job.document_id]
    assert matching
    combined_text = "\n".join(result.text for result in matching)
    assert "Kafka" in combined_text
    assert "Kubernetes" in combined_text
    assert "עיבוד החזרים" in combined_text or "ארכיטקטורת" in combined_text
