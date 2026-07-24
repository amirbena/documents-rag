"""Postgres integration tests for full-deletion's completed-re-index-target ownership
(Phase 2.8.6, subtask 3) — real Testcontainers Postgres, no Qdrant required.

Proves `get_completed_reindex_target_collections()`'s query correctness and
`delete_all_tracked_document_vectors()`'s full three-source resolution (current collection +
pending/failed cleanup collections + completed re-index targets) against real rows and real
foreign keys — properties a fake session double cannot faithfully represent. Real-Qdrant vector
removal is covered separately by
tests/integration/documents/deletion/test_qdrant.py's completed-re-index-target scenario.
"""

import uuid
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.document import Document
from app.models.index_collection import IndexCollection, IndexCollectionStatus
from app.models.reindex_job import ReindexJob, ReindexJobStatus
from app.models.vector_cleanup_job import VectorCleanupJob, VectorCleanupStatus
from app.services.indexing.reindex_scheduling_service import get_completed_reindex_target_collections
from app.services.indexing.vector_deletion_service import delete_all_tracked_document_vectors


class _RecordingVectorStore:
    """A VectorStore double recording every delete call — no real Qdrant needed for these tests,
    which verify Postgres query ownership, not actual vector removal."""

    def __init__(self) -> None:
        self.deleted: list[tuple[str, str]] = []

    async def delete_by_document_id(self, collection_name: str, document_id: str) -> None:
        self.deleted.append((collection_name, document_id))


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


def _index_collection(collection_name: str) -> IndexCollection:
    return IndexCollection(
        collection_name=collection_name,
        embedding_provider="ollama",
        embedding_model="target-model",
        embedding_dimension=768,
        embedding_version="v1",
        chunking_version="v1",
        status=IndexCollectionStatus.ACTIVE,
    )


async def _ensure_index_collection(session: AsyncSession, collection_name: str) -> None:
    """Seed collection_name's IndexCollection row if it doesn't already exist.

    Document.collection_name carries a foreign key into index_collections (see the alembic
    baseline migration) — a document can never reference a collection that isn't itself persisted there.
    """
    existing = await session.get(IndexCollection, collection_name)
    if existing is None:
        session.add(_index_collection(collection_name))
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
    )
    fields.update(overrides)

    collection_name = fields.get("collection_name")
    if collection_name is not None:
        await _ensure_index_collection(session, collection_name)  # type: ignore[arg-type]

    document = Document(**fields)  # type: ignore[arg-type]
    session.add(document)
    await session.commit()
    return document


def _reindex_job(document_id: str, status: ReindexJobStatus, target_collection_name: str) -> ReindexJob:
    # source_collection_name only needs to reference a real, already-seeded IndexCollection row
    # for this file's tests (none of them exercise activation's source-staleness check) — reusing
    # the target itself is always valid here, since every call site seeds it before calling this.
    return ReindexJob(
        id=str(uuid.uuid4()),
        document_id=document_id,
        source_collection_name=target_collection_name,
        target_collection_name=target_collection_name,
        target_chunk_size=500,
        target_chunk_overlap=50,
        status=status,
    )


async def test_completed_reindex_job_contributes_its_target_collection(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(integration_db_session, collection_name=None)
    integration_db_session.add(_index_collection("target-b"))
    await integration_db_session.commit()
    integration_db_session.add(
        _reindex_job(document.id, ReindexJobStatus.COMPLETED, "target-b")
    )
    await integration_db_session.commit()

    result = await get_completed_reindex_target_collections(integration_db_session, document.id)

    assert result == ["target-b"]


@pytest.mark.parametrize(
    "status", [ReindexJobStatus.PENDING, ReindexJobStatus.PROCESSING, ReindexJobStatus.FAILED]
)
async def test_non_completed_reindex_jobs_do_not_contribute(
    status: ReindexJobStatus, migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(integration_db_session, collection_name=None)
    integration_db_session.add(_index_collection("target-b"))
    await integration_db_session.commit()
    integration_db_session.add(_reindex_job(document.id, status, "target-b"))
    await integration_db_session.commit()

    result = await get_completed_reindex_target_collections(integration_db_session, document.id)

    assert result == []


async def test_completed_jobs_belonging_to_another_document_do_not_contribute(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(integration_db_session, collection_name=None)
    other_document = await _seed_document(integration_db_session, collection_name=None)
    integration_db_session.add(_index_collection("target-b"))
    await integration_db_session.commit()
    integration_db_session.add(_reindex_job(other_document.id, ReindexJobStatus.COMPLETED, "target-b"))
    await integration_db_session.commit()

    result = await get_completed_reindex_target_collections(integration_db_session, document.id)

    assert result == []


async def test_multiple_completed_targets_are_all_returned(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(integration_db_session, collection_name="collection-a")
    integration_db_session.add(_index_collection("collection-b"))
    integration_db_session.add(_index_collection("collection-c"))
    await integration_db_session.commit()
    integration_db_session.add(_reindex_job(document.id, ReindexJobStatus.COMPLETED, "collection-b"))
    integration_db_session.add(_reindex_job(document.id, ReindexJobStatus.COMPLETED, "collection-c"))
    await integration_db_session.commit()

    result = await get_completed_reindex_target_collections(integration_db_session, document.id)

    assert set(result) == {"collection-b", "collection-c"}


async def test_target_collection_name_resolves_through_the_foreign_key(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    """target_collection_name is a real FK into index_collections — must round-trip exactly."""
    document = await _seed_document(integration_db_session, collection_name=None)
    collection = _index_collection("documents__ollama__target-model__ev1__cv1__d768")
    integration_db_session.add(collection)
    await integration_db_session.commit()
    integration_db_session.add(
        _reindex_job(document.id, ReindexJobStatus.COMPLETED, collection.collection_name)
    )
    await integration_db_session.commit()

    result = await get_completed_reindex_target_collections(integration_db_session, document.id)

    assert result == [collection.collection_name]


async def test_full_deletion_deduplicates_a_completed_target_seen_via_multiple_rows(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(integration_db_session, collection_name="collection-a")
    integration_db_session.add(_index_collection("collection-b"))
    await integration_db_session.commit()
    integration_db_session.add(_reindex_job(document.id, ReindexJobStatus.COMPLETED, "collection-b"))
    integration_db_session.add(_reindex_job(document.id, ReindexJobStatus.COMPLETED, "collection-b"))
    await integration_db_session.commit()

    vector_store = _RecordingVectorStore()
    result = await delete_all_tracked_document_vectors(document, vector_store, integration_db_session)

    assert result.attempted_collections.count("collection-b") == 1
    assert vector_store.deleted.count(("collection-b", document.id)) == 1


async def test_full_deletion_combines_current_cleanup_and_completed_target_collections(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    """Current collection + a pending cleanup collection + a completed re-index target must all
    appear exactly once in the full-deletion resolved set."""
    document = await _seed_document(integration_db_session, collection_name="collection-a")
    integration_db_session.add(_index_collection("collection-c"))
    await integration_db_session.commit()

    integration_db_session.add(
        VectorCleanupJob(
            id=str(uuid.uuid4()),
            document_id=document.id,
            collection_name="collection-b",
            status=VectorCleanupStatus.PENDING,
        )
    )
    integration_db_session.add(_reindex_job(document.id, ReindexJobStatus.COMPLETED, "collection-c"))
    await integration_db_session.commit()

    vector_store = _RecordingVectorStore()
    result = await delete_all_tracked_document_vectors(document, vector_store, integration_db_session)

    assert set(result.attempted_collections) == {"collection-a", "collection-b", "collection-c"}
    assert result.fully_deleted is True
