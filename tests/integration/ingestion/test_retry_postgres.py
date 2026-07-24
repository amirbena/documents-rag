"""Postgres integration tests for ingestion retry — real Testcontainers Postgres, real locks.

Proves the properties a fake session double cannot faithfully represent: the partial unique index
actually rejecting a second concurrent active row, and that history (old FAILED/stale rows)
survives a retry. Mirrors tests/integration/test_alembic_migrations.py's and
tests/integration/documents/read/test_postgres.py's fixtures/style. Concurrent-retry/recovery races are
covered separately by tests/integration/ingestion/test_concurrency.py.
"""

import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.services.ingestion.worker as ingestion_worker_module
from app.core.config import get_settings
from app.models.document import Document
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.rag.embedding_config import get_active_embedding_config
from app.rag.providers.qdrant_vector_store import QdrantVectorStore
from app.services.ingestion.retry_service import RetryOutcome, retry_ingestion
from app.services.ingestion.worker import IngestionWorker
from app.storage.local_storage import LocalFileStorage

STALE_AFTER_SECONDS = 900


class _FailingEmbeddingProvider:
    """Always raises — simulates the first, failing ingestion attempt."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("simulated embedding provider failure")


class _FakeEmbeddingProvider:
    """Returns one fixed-length deterministic vector per text — no real Ollama call."""

    def __init__(self, vector_size: int) -> None:
        self._vector = [0.1] * vector_size

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector for _ in texts]


@pytest.fixture(autouse=True)
async def _clean_tables(migrated_schema: None, postgres_url: str) -> AsyncIterator[None]:
    """Truncate documents/ingestion_jobs before each test — see test_worker_postgres.py.

    Every test in this module shares one session-scoped Postgres container with every other
    integration test in the suite. Several tests here deliberately leave a fresh PENDING job
    uncommitted-to-a-worker (retry/recovery only *create* jobs, they never process them) — left
    behind, such a row would be claimed by an unrelated test's `IngestionWorker.process_next_job()`
    (which claims the oldest PENDING row globally, with no document_id filter), so this module
    must guarantee a clean slate before every test, not just isolate its own rows afterward.
    """
    engine = create_async_engine(postgres_url, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def _truncate() -> None:
        async with session_factory() as session:
            await session.execute(
                text("TRUNCATE TABLE ingestion_jobs, documents RESTART IDENTITY CASCADE")
            )
            await session.commit()

    await _truncate()
    try:
        yield
    finally:
        # Also truncate afterward (not just before): several tests in this module deliberately
        # create PENDING jobs that are never processed, and this module's tests run alongside
        # unrelated test modules sharing the same session-scoped container — some of which (e.g.
        # test_worker_minio.py) do not truncate before their own tests and would
        # otherwise have their worker claim one of this module's leftover PENDING rows.
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
    )
    session.add(document)
    await session.commit()
    return document


async def test_migration_creates_the_partial_unique_index(
    migrated_schema: None, postgres_url: str
) -> None:
    """alembic upgrade head must have created the one-active-job-per-document partial index."""
    from sqlalchemy import inspect
    from sqlalchemy.ext.asyncio import AsyncEngine

    engine: AsyncEngine = create_async_engine(postgres_url, future=True)
    try:
        async with engine.connect() as conn:
            index_names = await conn.run_sync(
                lambda sync_conn: {idx["name"] for idx in inspect(sync_conn).get_indexes("ingestion_jobs")}
            )
    finally:
        await engine.dispose()

    assert "ix_ingestion_jobs_one_active_per_document" in index_names


async def test_partial_unique_index_rejects_two_active_jobs_for_one_document(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    """A second PENDING/PROCESSING row for the same document must violate the unique index."""
    document = await _seed_document(integration_db_session)
    integration_db_session.add(
        IngestionJob(id=str(uuid.uuid4()), document_id=document.id, status=IngestionStatus.PENDING)
    )
    await integration_db_session.commit()

    integration_db_session.add(
        IngestionJob(id=str(uuid.uuid4()), document_id=document.id, status=IngestionStatus.PROCESSING)
    )
    try:
        await integration_db_session.commit()
        raised = False
    except IntegrityError:
        raised = True
        await integration_db_session.rollback()

    assert raised


async def test_partial_unique_index_allows_a_completed_and_a_pending_row(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    """A COMPLETED row plus one active row for the same document is allowed (history is append-only)."""
    document = await _seed_document(integration_db_session)
    integration_db_session.add(
        IngestionJob(id=str(uuid.uuid4()), document_id=document.id, status=IngestionStatus.COMPLETED)
    )
    integration_db_session.add(
        IngestionJob(id=str(uuid.uuid4()), document_id=document.id, status=IngestionStatus.PENDING)
    )
    await integration_db_session.commit()  # must not raise

    result = await integration_db_session.execute(
        text("SELECT count(*) FROM ingestion_jobs WHERE document_id = :id"), {"id": document.id}
    )
    assert result.scalar_one() == 2


async def test_retry_history_preserved_after_creating_new_job(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    """The old FAILED row must remain queryable (unchanged) after a retry creates a new job."""
    document = await _seed_document(integration_db_session)
    failed_job = IngestionJob(
        id=str(uuid.uuid4()), document_id=document.id, status=IngestionStatus.FAILED, error_message="boom"
    )
    integration_db_session.add(failed_job)
    await integration_db_session.commit()

    result = await retry_ingestion(
        integration_db_session, document.id, stale_after_seconds=STALE_AFTER_SECONDS
    )
    assert result.outcome == RetryOutcome.CREATED

    rows = await integration_db_session.execute(
        text(
            "SELECT id, status, error_message FROM ingestion_jobs "
            "WHERE document_id = :id ORDER BY created_at"
        ),
        {"id": document.id},
    )
    stored = rows.all()
    assert len(stored) == 2
    assert stored[0].id == failed_job.id
    assert stored[0].status == "failed"
    assert stored[0].error_message == "boom"
    assert stored[1].status == "pending"


async def test_retry_after_real_failure_writes_no_orphaned_vectors_then_succeeds(
    migrated_schema: None,
    postgres_url: str,
    qdrant_url: str,
    integration_db_session: AsyncSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proves, against real Postgres + real Qdrant, that a FAILED attempt writes zero vectors and
    a subsequent retry completes and its points become searchable — not just asserted from
    reading the worker's source, but actually observed.
    """
    settings = get_settings()
    collection_prefix = f"retry-idempotency-{uuid.uuid4().hex}"
    monkeypatch.setattr(settings, "qdrant_collection_name", collection_prefix)
    active_config = get_active_embedding_config(settings)

    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world " * 100, encoding="utf-8")

    document = Document(
        id=str(uuid.uuid4()),
        original_filename="notes.txt",
        stored_filename=file_path.name,
        content_type="text/plain",
        file_size=file_path.stat().st_size,
        stored_path=file_path.name,
        storage_provider="local",
        storage_key=file_path.name,
    )
    integration_db_session.add(document)
    integration_db_session.add(
        IngestionJob(id=str(uuid.uuid4()), document_id=document.id, status=IngestionStatus.PENDING)
    )
    await integration_db_session.commit()

    vector_store = QdrantVectorStore(settings=settings)
    # Pre-create the collection so a search against it before any successful upsert reliably
    # observes "zero points" rather than a 404 for a not-yet-created collection — a fresh attempt
    # ordinarily creates the collection itself in ensure_active_collection(), just later in the
    # pipeline (after embedding, which is exactly the step this test forces to fail first).
    await vector_store.create_collection_if_not_exists(active_config.collection_name, active_config.dimension)
    worker = IngestionWorker(file_storage=LocalFileStorage(root=tmp_path))

    # First attempt: embedding provider fails before upsert_vectors() is ever reached.
    monkeypatch.setattr(
        ingestion_worker_module, "get_embedding_provider", lambda settings=None: _FailingEmbeddingProvider()
    )
    first_result = await worker.process_next_job(integration_db_session)
    assert first_result is not None
    assert first_result.status == IngestionStatus.FAILED

    query_vector = [0.1] * settings.vector_size
    points_after_failure = await vector_store.search_similar(
        active_config.collection_name, query_vector, limit=10
    )
    assert points_after_failure == []

    # Retry creates a new PENDING job for the same document.
    retry_result = await retry_ingestion(
        integration_db_session, document.id, stale_after_seconds=STALE_AFTER_SECONDS
    )
    assert retry_result.outcome == RetryOutcome.CREATED

    # Second attempt: embedding provider now succeeds, mirroring a transient failure resolving.
    monkeypatch.setattr(
        ingestion_worker_module,
        "get_embedding_provider",
        lambda settings=None: _FakeEmbeddingProvider(get_settings().vector_size),
    )
    second_result = await worker.process_next_job(integration_db_session)
    assert second_result is not None
    assert second_result.status == IngestionStatus.COMPLETED

    points_after_retry = await vector_store.search_similar(
        active_config.collection_name, query_vector, limit=10
    )
    assert len(points_after_retry) > 0
    assert all(point.document_id == document.id for point in points_after_retry)
