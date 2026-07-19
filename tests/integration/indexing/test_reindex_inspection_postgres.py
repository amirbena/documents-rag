"""Postgres integration tests for re-index inspection (Phase 2.8.6, subtask 6) — real Testcontainers
Postgres, real rows.

Drives the actual `schedule_reindex()`/`activate_reindexed_document()` services (never
reimplementing their decision tables here — see "Do not duplicate the complete scheduling, worker,
or activation concurrency suites already covered in earlier subtasks" in this subtask's spec) and
asserts `inspect_document_reindex_state()` reads the resulting persisted state back correctly. Full
scheduling/activation decision-table coverage lives in
tests/integration/indexing/test_reindex_scheduling_postgres.py and
test_reindex_activation_postgres.py — this module only covers the read path's own correctness
against real data.
"""

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.models.document import Document
from app.models.document_deletion_job import DocumentDeletionJob, DocumentDeletionStatus
from app.models.index_collection import IndexCollection, IndexCollectionStatus
from app.models.reindex_job import ReindexJob, ReindexJobStatus
from app.rag.embedding_config import EmbeddingIndexConfig, get_active_embedding_config
from app.schemas.reindex import ReindexLifecycleState
from app.services.indexing.reindex_activation import activate_reindexed_document
from app.services.indexing.reindex_inspection_service import inspect_document_reindex_state
from app.services.indexing.reindex_scheduling_service import ReindexSchedulingOutcome, schedule_reindex


class _NoopVectorStore:
    """A VectorStore double sufficient for ensure_active_collection() — real Qdrant is out of scope."""

    async def get_collection_vector_size(self, collection_name: str) -> int | None:
        return None

    async def create_collection_if_not_exists(self, collection_name: str, vector_size: int) -> None:
        return None


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


def _desired_config() -> EmbeddingIndexConfig:
    return get_active_embedding_config(get_settings())


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


async def _seed_document(
    session: AsyncSession, config: EmbeddingIndexConfig, **overrides: object
) -> Document:
    await _seed_index_collection(session, config)
    fields: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        original_filename="report.pdf",
        stored_filename=f"{uuid.uuid4().hex}.pdf",
        content_type="application/pdf",
        file_size=10,
        stored_path="storage/documents/report.pdf",
        storage_provider="local",
        storage_key=f"storage/documents/{uuid.uuid4().hex}.pdf",
        collection_name=config.collection_name,
        embedding_provider=config.provider,
        embedding_model=config.model,
        embedding_dimension=config.dimension,
        embedding_version=config.embedding_version,
        chunking_version=config.chunking_version,
        indexed_at=datetime.now(UTC),
    )
    fields.update(overrides)
    document = Document(**fields)  # type: ignore[arg-type]
    session.add(document)
    await session.commit()
    return document


async def test_inspection_returns_the_latest_attempt_deterministically(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    old_config = _desired_config()
    document = await _seed_document(integration_db_session, old_config, embedding_version="v0")
    target_a = EmbeddingIndexConfig(
        collection_prefix=old_config.collection_prefix,
        provider=old_config.provider,
        model="target-a",
        dimension=old_config.dimension,
        embedding_version=old_config.embedding_version,
        chunking_version=old_config.chunking_version,
    )
    target_b = EmbeddingIndexConfig(
        collection_prefix=old_config.collection_prefix,
        provider=old_config.provider,
        model="target-b",
        dimension=old_config.dimension,
        embedding_version=old_config.embedding_version,
        chunking_version=old_config.chunking_version,
    )
    await _seed_index_collection(integration_db_session, target_a)
    await _seed_index_collection(integration_db_session, target_b)

    now = datetime.now(UTC)
    older_job = ReindexJob(
        id=str(uuid.uuid4()),
        document_id=document.id,
        source_collection_name=old_config.collection_name,
        target_collection_name=target_a.collection_name,
        target_chunk_size=500,
        target_chunk_overlap=50,
        status=ReindexJobStatus.FAILED,
        created_at=now - timedelta(minutes=10),
    )
    newer_job = ReindexJob(
        id=str(uuid.uuid4()),
        document_id=document.id,
        source_collection_name=old_config.collection_name,
        target_collection_name=target_b.collection_name,
        target_chunk_size=500,
        target_chunk_overlap=50,
        status=ReindexJobStatus.PENDING,
        created_at=now,
    )
    integration_db_session.add(older_job)
    integration_db_session.add(newer_job)
    await integration_db_session.commit()

    result = await inspect_document_reindex_state(integration_db_session, document.id, get_settings())

    assert result is not None
    assert result.latest_job is not None
    assert result.latest_job.id == newer_job.id


async def test_scheduling_via_service_creates_one_pending_job_reflected_in_inspection(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    old_config = EmbeddingIndexConfig(
        collection_prefix=_desired_config().collection_prefix,
        provider=_desired_config().provider,
        model="old-model",
        dimension=_desired_config().dimension,
        embedding_version="v0",
        chunking_version="v0",
    )
    document = await _seed_document(integration_db_session, old_config)
    desired = _desired_config()

    scheduling_result = await schedule_reindex(
        integration_db_session,
        document,
        _NoopVectorStore(),
        desired,
        target_chunk_size=500,
        target_chunk_overlap=50,
    )
    assert scheduling_result.outcome == ReindexSchedulingOutcome.CREATED

    result = await inspect_document_reindex_state(integration_db_session, document.id, get_settings())

    assert result is not None
    assert result.state == ReindexLifecycleState.REINDEX_PENDING
    assert result.latest_job is not None
    assert result.latest_job.status == ReindexJobStatus.PENDING


async def test_repeated_scheduling_returns_one_active_attempt(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    old_config = EmbeddingIndexConfig(
        collection_prefix=_desired_config().collection_prefix,
        provider=_desired_config().provider,
        model="old-model",
        dimension=_desired_config().dimension,
        embedding_version="v0",
        chunking_version="v0",
    )
    document = await _seed_document(integration_db_session, old_config)
    desired = _desired_config()

    first = await schedule_reindex(
        integration_db_session, document, _NoopVectorStore(), desired,
        target_chunk_size=500, target_chunk_overlap=50,
    )
    second = await schedule_reindex(
        integration_db_session, document, _NoopVectorStore(), desired,
        target_chunk_size=500, target_chunk_overlap=50,
    )

    assert first.outcome == ReindexSchedulingOutcome.CREATED
    assert second.outcome == ReindexSchedulingOutcome.ALREADY_ACTIVE
    assert second.job is not None
    assert first.job is not None
    assert second.job.id == first.job.id

    result = await inspect_document_reindex_state(integration_db_session, document.id, get_settings())
    assert result is not None
    assert result.latest_job is not None
    assert result.latest_job.id == first.job.id


async def test_failed_historical_attempt_remains_unchanged_when_new_attempt_scheduled(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    old_config = EmbeddingIndexConfig(
        collection_prefix=_desired_config().collection_prefix,
        provider=_desired_config().provider,
        model="old-model",
        dimension=_desired_config().dimension,
        embedding_version="v0",
        chunking_version="v0",
    )
    document = await _seed_document(integration_db_session, old_config)
    abandoned_target = EmbeddingIndexConfig(
        collection_prefix=old_config.collection_prefix,
        provider=old_config.provider,
        model="abandoned-target",
        dimension=old_config.dimension,
        embedding_version=old_config.embedding_version,
        chunking_version=old_config.chunking_version,
    )
    await _seed_index_collection(integration_db_session, abandoned_target)
    failed_job = ReindexJob(
        id=str(uuid.uuid4()),
        document_id=document.id,
        source_collection_name=old_config.collection_name,
        target_collection_name=abandoned_target.collection_name,
        target_chunk_size=500,
        target_chunk_overlap=50,
        status=ReindexJobStatus.FAILED,
        error_message="a prior internal failure",
    )
    integration_db_session.add(failed_job)
    await integration_db_session.commit()
    failed_error_message_before = failed_job.error_message
    failed_status_before = failed_job.status

    desired = _desired_config()
    scheduling_result = await schedule_reindex(
        integration_db_session, document, _NoopVectorStore(), desired,
        target_chunk_size=500, target_chunk_overlap=50,
    )
    assert scheduling_result.outcome == ReindexSchedulingOutcome.CREATED

    refreshed_failed_job = await integration_db_session.get(ReindexJob, failed_job.id)
    assert refreshed_failed_job is not None
    assert refreshed_failed_job.status == failed_status_before
    assert refreshed_failed_job.error_message == failed_error_message_before

    result = await inspect_document_reindex_state(integration_db_session, document.id, get_settings())
    assert result is not None
    assert result.latest_job is not None
    assert result.latest_job.id == scheduling_result.job.id  # type: ignore[union-attr]
    assert result.latest_job.id != failed_job.id


async def test_completed_unactivated_job_is_reported_as_built_but_not_active(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    old_config = EmbeddingIndexConfig(
        collection_prefix=_desired_config().collection_prefix,
        provider=_desired_config().provider,
        model="old-model",
        dimension=_desired_config().dimension,
        embedding_version="v0",
        chunking_version="v0",
    )
    document = await _seed_document(integration_db_session, old_config)
    desired = _desired_config()
    await _seed_index_collection(integration_db_session, desired)
    job = ReindexJob(
        id=str(uuid.uuid4()),
        document_id=document.id,
        source_collection_name=old_config.collection_name,
        target_collection_name=desired.collection_name,
        target_chunk_size=500,
        target_chunk_overlap=50,
        status=ReindexJobStatus.COMPLETED,
        completed_at=datetime.now(UTC),
        activated_at=None,
    )
    integration_db_session.add(job)
    await integration_db_session.commit()

    result = await inspect_document_reindex_state(integration_db_session, document.id, get_settings())

    assert result is not None
    assert result.state == ReindexLifecycleState.TARGET_BUILT
    assert result.can_activate is True
    assert result.is_stale is True  # document.collection_name still points at A


async def test_successful_activation_updates_the_inspection_response(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    old_config = EmbeddingIndexConfig(
        collection_prefix=_desired_config().collection_prefix,
        provider=_desired_config().provider,
        model="old-model",
        dimension=_desired_config().dimension,
        embedding_version="v0",
        chunking_version="v0",
    )
    document = await _seed_document(integration_db_session, old_config)
    desired = _desired_config()
    await _seed_index_collection(integration_db_session, desired)
    job = ReindexJob(
        id=str(uuid.uuid4()),
        document_id=document.id,
        source_collection_name=old_config.collection_name,
        target_collection_name=desired.collection_name,
        target_chunk_size=500,
        target_chunk_overlap=50,
        status=ReindexJobStatus.COMPLETED,
        completed_at=datetime.now(UTC),
        activated_at=None,
    )
    integration_db_session.add(job)
    await integration_db_session.commit()

    activation_result = await activate_reindexed_document(integration_db_session, job.id)
    assert activation_result.outcome.value == "activated"

    result = await inspect_document_reindex_state(integration_db_session, document.id, get_settings())

    assert result is not None
    assert result.state == ReindexLifecycleState.ACTIVATED
    assert result.is_stale is False
    assert result.can_activate is False
    assert result.active_index.collection_name == desired.collection_name


async def test_deletion_lifecycle_blocks_scheduling(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    old_config = EmbeddingIndexConfig(
        collection_prefix=_desired_config().collection_prefix,
        provider=_desired_config().provider,
        model="old-model",
        dimension=_desired_config().dimension,
        embedding_version="v0",
        chunking_version="v0",
    )
    document = await _seed_document(integration_db_session, old_config)
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

    desired = _desired_config()
    scheduling_result = await schedule_reindex(
        integration_db_session, document, _NoopVectorStore(), desired,
        target_chunk_size=500, target_chunk_overlap=50,
    )
    assert scheduling_result.outcome == ReindexSchedulingOutcome.DELETION_ACTIVE

    result = await inspect_document_reindex_state(integration_db_session, document.id, get_settings())
    assert result is not None
    assert result.state == ReindexLifecycleState.DELETION_BLOCKED
    assert result.can_schedule is False


async def test_deletion_lifecycle_blocks_activation(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    old_config = EmbeddingIndexConfig(
        collection_prefix=_desired_config().collection_prefix,
        provider=_desired_config().provider,
        model="old-model",
        dimension=_desired_config().dimension,
        embedding_version="v0",
        chunking_version="v0",
    )
    document = await _seed_document(integration_db_session, old_config)
    desired = _desired_config()
    await _seed_index_collection(integration_db_session, desired)
    job = ReindexJob(
        id=str(uuid.uuid4()),
        document_id=document.id,
        source_collection_name=old_config.collection_name,
        target_collection_name=desired.collection_name,
        target_chunk_size=500,
        target_chunk_overlap=50,
        status=ReindexJobStatus.COMPLETED,
        completed_at=datetime.now(UTC),
        activated_at=None,
    )
    integration_db_session.add(job)
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

    activation_result = await activate_reindexed_document(integration_db_session, job.id)
    assert activation_result.outcome.value == "blocked_by_deletion"

    result = await inspect_document_reindex_state(integration_db_session, document.id, get_settings())
    assert result is not None
    assert result.state == ReindexLifecycleState.DELETION_BLOCKED
    assert result.can_activate is False
