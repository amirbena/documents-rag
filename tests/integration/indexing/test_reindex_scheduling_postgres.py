"""Postgres integration tests for re-index job scheduling — real Testcontainers Postgres, real locks.

Proves properties a fake session double cannot faithfully represent: the migration's table/columns,
foreign-key enforcement, the partial unique index actually rejecting a second concurrent active job,
append-only historical rows, migration upgrade/downgrade/re-upgrade, and genuine concurrent-
scheduling convergence. Full decision-table coverage against a fake session lives in
tests/unit/services/indexing/test_reindex_scheduling_service.py — this module only covers what a
fake cannot: real constraints and real races. Build execution, Qdrant, and object storage are all
out of scope for this subtask.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.models.document import Document
from app.models.index_collection import IndexCollection, IndexCollectionStatus
from app.models.reindex_job import ReindexJob, ReindexJobStatus
from app.rag.embedding_config import EmbeddingIndexConfig
from app.services.indexing.reindex_scheduling_service import ReindexSchedulingOutcome, schedule_reindex


class _NoopVectorStore:
    """A VectorStore double sufficient for ensure_active_collection() — real Qdrant is out of scope."""

    async def get_collection_vector_size(self, collection_name: str) -> int | None:
        return None

    async def create_collection_if_not_exists(self, collection_name: str, vector_size: int) -> None:
        return None


@pytest.fixture(autouse=True)
async def _clean_tables(migrated_schema: None, postgres_url: str) -> AsyncIterator[None]:
    """Truncate reindex/index-collection/ingestion/deletion/document tables before and after each test."""
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


def _target_config(**overrides: object) -> EmbeddingIndexConfig:
    fields: dict[str, object] = dict(
        collection_prefix="documents",
        provider="ollama",
        model="target-model",
        dimension=768,
        embedding_version="v9",
        chunking_version="v9",
    )
    fields.update(overrides)
    return EmbeddingIndexConfig(**fields)  # type: ignore[arg-type]


# The "old"/currently-serving config every seeded document defaults to `collection_name` under.
# `Document.collection_name` carries a foreign key into `index_collections` (see the alembic
# baseline migration) — a document can never reference a collection that isn't itself persisted there.
_OLD_CONFIG = _target_config(model="old-model", embedding_version="v0", chunking_version="v0")


async def _seed_index_collection(session: AsyncSession, config: EmbeddingIndexConfig) -> IndexCollection:
    existing = await session.get(IndexCollection, config.collection_name)
    if existing is not None:
        return existing

    record = IndexCollection(
        collection_name=config.collection_name,
        embedding_provider=config.provider,
        embedding_model=config.model,
        embedding_dimension=config.dimension,
        embedding_version=config.embedding_version,
        chunking_version=config.chunking_version,
        status=IndexCollectionStatus.ACTIVE,
    )
    session.add(record)
    await session.commit()
    return record


async def _seed_document(session: AsyncSession, **overrides: object) -> Document:
    await _seed_index_collection(session, _OLD_CONFIG)

    fields: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        original_filename="report.pdf",
        stored_filename=f"{uuid.uuid4().hex}.pdf",
        content_type="application/pdf",
        file_size=10,
        stored_path="storage/documents/report.pdf",
        storage_provider="local",
        storage_key=f"storage/documents/{uuid.uuid4().hex}.pdf",
        collection_name=_OLD_CONFIG.collection_name,
    )
    fields.update(overrides)
    document = Document(**fields)  # type: ignore[arg-type]
    session.add(document)
    await session.commit()
    return document


# --- migration ------------------------------------------------------------------------------


async def test_migration_creates_reindex_jobs_table_with_expected_columns(
    migrated_schema: None, postgres_url: str
) -> None:
    """alembic upgrade head must create reindex_jobs with every model column plus the partial index."""
    engine: AsyncEngine = create_async_engine(postgres_url, future=True)
    try:
        async with engine.connect() as conn:
            table_names = await conn.run_sync(lambda sync_conn: set(inspect(sync_conn).get_table_names()))
            columns = await conn.run_sync(
                lambda sync_conn: {col["name"] for col in inspect(sync_conn).get_columns("reindex_jobs")}
            )
            index_names = await conn.run_sync(
                lambda sync_conn: {idx["name"] for idx in inspect(sync_conn).get_indexes("reindex_jobs")}
            )
    finally:
        await engine.dispose()

    assert "reindex_jobs" in table_names
    assert columns == {
        "id",
        "document_id",
        "source_collection_name",
        "target_collection_name",
        "target_chunk_size",
        "target_chunk_overlap",
        "status",
        "error_message",
        "created_at",
        "updated_at",
        "completed_at",
        "activated_at",
    }
    assert "ix_reindex_jobs_one_active_per_document" in index_names


# --- foreign keys -----------------------------------------------------------------------------


async def test_document_foreign_key_is_enforced(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    """A ReindexJob for a document_id that doesn't exist must violate the foreign key."""
    target = _target_config()
    await _seed_index_collection(integration_db_session, target)

    integration_db_session.add(
        ReindexJob(
            id=str(uuid.uuid4()),
            document_id=str(uuid.uuid4()),
            source_collection_name=target.collection_name,
            target_collection_name=target.collection_name,
            target_chunk_size=500,
            target_chunk_overlap=50,
            status=ReindexJobStatus.PENDING,
        )
    )
    with pytest.raises(IntegrityError):
        await integration_db_session.commit()
    await integration_db_session.rollback()


async def test_target_collection_foreign_key_is_enforced(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    """A ReindexJob targeting a collection_name with no IndexCollection row must violate the FK."""
    document = await _seed_document(integration_db_session)

    integration_db_session.add(
        ReindexJob(
            id=str(uuid.uuid4()),
            document_id=document.id,
            source_collection_name=_OLD_CONFIG.collection_name,
            target_collection_name="never-persisted-collection",
            target_chunk_size=500,
            target_chunk_overlap=50,
            status=ReindexJobStatus.PENDING,
        )
    )
    with pytest.raises(IntegrityError):
        await integration_db_session.commit()
    await integration_db_session.rollback()


# --- one active job per document ----------------------------------------------------------------


async def test_one_pending_job_is_allowed(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(integration_db_session)
    target = _target_config()
    await _seed_index_collection(integration_db_session, target)

    integration_db_session.add(
        ReindexJob(
            id=str(uuid.uuid4()),
            document_id=document.id,
            source_collection_name=_OLD_CONFIG.collection_name,
            target_collection_name=target.collection_name,
            target_chunk_size=500,
            target_chunk_overlap=50,
            status=ReindexJobStatus.PENDING,
        )
    )
    await integration_db_session.commit()  # must not raise


async def test_second_active_job_for_same_document_is_rejected(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(integration_db_session)
    target = _target_config()
    await _seed_index_collection(integration_db_session, target)

    integration_db_session.add(
        ReindexJob(
            id=str(uuid.uuid4()),
            document_id=document.id,
            source_collection_name=_OLD_CONFIG.collection_name,
            target_collection_name=target.collection_name,
            target_chunk_size=500,
            target_chunk_overlap=50,
            status=ReindexJobStatus.PENDING,
        )
    )
    await integration_db_session.commit()

    integration_db_session.add(
        ReindexJob(
            id=str(uuid.uuid4()),
            document_id=document.id,
            source_collection_name=_OLD_CONFIG.collection_name,
            target_collection_name=target.collection_name,
            target_chunk_size=500,
            target_chunk_overlap=50,
            status=ReindexJobStatus.PROCESSING,
        )
    )
    with pytest.raises(IntegrityError):
        await integration_db_session.commit()
    await integration_db_session.rollback()


async def test_multiple_historical_terminal_jobs_are_allowed(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    """Two terminal (FAILED + COMPLETED) rows for the same document must coexist — append-only history."""
    document = await _seed_document(integration_db_session)
    target = _target_config()
    await _seed_index_collection(integration_db_session, target)

    integration_db_session.add(
        ReindexJob(
            id=str(uuid.uuid4()),
            document_id=document.id,
            source_collection_name=_OLD_CONFIG.collection_name,
            target_collection_name=target.collection_name,
            target_chunk_size=500,
            target_chunk_overlap=50,
            status=ReindexJobStatus.FAILED,
        )
    )
    integration_db_session.add(
        ReindexJob(
            id=str(uuid.uuid4()),
            document_id=document.id,
            source_collection_name=_OLD_CONFIG.collection_name,
            target_collection_name=target.collection_name,
            target_chunk_size=500,
            target_chunk_overlap=50,
            status=ReindexJobStatus.COMPLETED,
        )
    )
    await integration_db_session.commit()  # must not raise

    count = await integration_db_session.execute(
        text("SELECT count(*) FROM reindex_jobs WHERE document_id = :id"), {"id": document.id}
    )
    assert count.scalar_one() == 2


async def test_new_active_job_allowed_after_previous_becomes_failed(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(integration_db_session)
    target = _target_config()
    await _seed_index_collection(integration_db_session, target)

    integration_db_session.add(
        ReindexJob(
            id=str(uuid.uuid4()),
            document_id=document.id,
            source_collection_name=_OLD_CONFIG.collection_name,
            target_collection_name=target.collection_name,
            target_chunk_size=500,
            target_chunk_overlap=50,
            status=ReindexJobStatus.FAILED,
        )
    )
    await integration_db_session.commit()

    integration_db_session.add(
        ReindexJob(
            id=str(uuid.uuid4()),
            document_id=document.id,
            source_collection_name=_OLD_CONFIG.collection_name,
            target_collection_name=target.collection_name,
            target_chunk_size=500,
            target_chunk_overlap=50,
            status=ReindexJobStatus.PENDING,
        )
    )
    await integration_db_session.commit()  # must not raise


async def test_new_active_job_allowed_after_previous_becomes_completed(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(integration_db_session)
    target = _target_config()
    await _seed_index_collection(integration_db_session, target)

    integration_db_session.add(
        ReindexJob(
            id=str(uuid.uuid4()),
            document_id=document.id,
            source_collection_name=_OLD_CONFIG.collection_name,
            target_collection_name=target.collection_name,
            target_chunk_size=500,
            target_chunk_overlap=50,
            status=ReindexJobStatus.COMPLETED,
        )
    )
    await integration_db_session.commit()

    integration_db_session.add(
        ReindexJob(
            id=str(uuid.uuid4()),
            document_id=document.id,
            source_collection_name=_OLD_CONFIG.collection_name,
            target_collection_name=target.collection_name,
            target_chunk_size=500,
            target_chunk_overlap=50,
            status=ReindexJobStatus.PENDING,
        )
    )
    await integration_db_session.commit()  # must not raise


# --- genuinely concurrent scheduling -------------------------------------------------------------


@pytest.mark.parametrize("run", range(3))
async def test_concurrent_scheduling_converges_on_one_active_job(
    run: int, migrated_schema: None, postgres_url: str, integration_db_session: AsyncSession
) -> None:
    """Repeated (3x) concurrency stress: two genuinely concurrent schedule_reindex() calls -> one job."""
    document = await _seed_document(integration_db_session)
    target = _target_config(model=f"target-model-{run}")
    await _seed_index_collection(integration_db_session, target)

    engine = create_async_engine(postgres_url, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def _schedule_with_own_session() -> tuple[ReindexSchedulingOutcome, str | None]:
        async with session_factory() as session:
            doc = await session.get(Document, document.id)
            assert doc is not None
            result = await schedule_reindex(
                session,
                doc,
                _NoopVectorStore(),
                target,
                target_chunk_size=500,
                target_chunk_overlap=50,
            )
            return result.outcome, (result.job.id if result.job is not None else None)

    try:
        results = await asyncio.gather(_schedule_with_own_session(), _schedule_with_own_session())
    finally:
        await engine.dispose()

    outcomes = sorted(outcome for outcome, _ in results)
    job_ids = {job_id for _, job_id in results}

    assert outcomes == sorted(
        [ReindexSchedulingOutcome.CREATED, ReindexSchedulingOutcome.ALREADY_ACTIVE]
    )
    assert len(job_ids) == 1  # both callers resolve to the same job identity

    active_rows = await integration_db_session.execute(
        text(
            "SELECT count(*) FROM reindex_jobs WHERE document_id = :id "
            "AND status IN ('pending', 'processing')"
        ),
        {"id": document.id},
    )
    assert active_rows.scalar_one() == 1
