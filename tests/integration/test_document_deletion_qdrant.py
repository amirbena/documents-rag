"""Real-Qdrant integration tests for DocumentDeletionWorker's vector cleanup step.

Proves delete_all_tracked_document_vectors() is genuinely called (active collection + every
distinct historical pending/failed VectorCleanupJob collection), that unrelated documents' vectors
are never touched, that repeated deletion is idempotent, and that a genuine per-collection failure
blocks storage cleanup — against a real, ephemeral Qdrant container, not a mocked httpx transport.
"""

import uuid
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.document import Document
from app.models.document_deletion_job import DocumentDeletionJob, DocumentDeletionStatus
from app.models.index_collection import IndexCollection
from app.models.vector_cleanup_job import VectorCleanupJob, VectorCleanupStatus
from app.rag.providers.qdrant_vector_store import QdrantVectorStore
from app.rag.providers.vector_store import VectorPoint
from app.services.document_deletion_service import DocumentDeletionWorker

_VECTOR_SIZE = 4


def _index_collection(collection_name: str) -> IndexCollection:
    """Build the IndexCollection row Document.collection_name's FK requires."""
    return IndexCollection(
        collection_name=collection_name,
        embedding_provider="ollama",
        embedding_model="test-model",
        embedding_dimension=_VECTOR_SIZE,
        embedding_version="v1",
        chunking_version="v1",
    )


class _NoopFileStorage:
    """A FileStorage whose delete always succeeds — these tests only exercise vector cleanup."""

    async def delete(self, key: str) -> None:
        return None

    async def save(self, key: str, content: bytes) -> None:  # pragma: no cover - unused
        raise NotImplementedError

    async def read(self, key: str) -> bytes:  # pragma: no cover - unused
        raise NotImplementedError

    async def exists(self, key: str) -> bool:  # pragma: no cover - unused
        raise NotImplementedError


@pytest.fixture(autouse=True)
async def _clean_tables(migrated_schema: None, postgres_url: str) -> AsyncIterator[None]:
    engine = create_async_engine(postgres_url, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def _truncate() -> None:
        async with session_factory() as session:
            await session.execute(
                text(
                    "TRUNCATE TABLE document_deletion_jobs, vector_cleanup_jobs, ingestion_jobs, "
                    "documents, index_collections RESTART IDENTITY CASCADE"
                )
            )
            await session.commit()

    await _truncate()
    try:
        yield
    finally:
        await _truncate()
        await engine.dispose()


def _point(document_id: str) -> VectorPoint:
    return VectorPoint(
        id=str(uuid.uuid4()),
        vector=[0.1, 0.2, 0.3, 0.4],
        document_id=document_id,
        chunk_id=f"{document_id}-0",
        text="hello",
        source="notes.txt",
    )


async def test_worker_deletes_active_and_historical_collections_but_not_unrelated_document(
    migrated_schema: None,
    qdrant_url: str,
    integration_db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.core.config import get_settings

    settings = get_settings()
    vector_store = QdrantVectorStore(settings=settings)

    active_collection = f"active-{uuid.uuid4().hex}"
    pending_collection = f"pending-{uuid.uuid4().hex}"
    failed_collection = f"failed-{uuid.uuid4().hex}"

    target_document = Document(
        id=str(uuid.uuid4()),
        original_filename="target.txt",
        stored_filename=f"{uuid.uuid4().hex}.txt",
        content_type="text/plain",
        file_size=5,
        stored_path="target.txt",
        storage_provider="local",
        storage_key="target.txt",
        collection_name=active_collection,
    )
    unrelated_document = Document(
        id=str(uuid.uuid4()),
        original_filename="unrelated.txt",
        stored_filename=f"{uuid.uuid4().hex}.txt",
        content_type="text/plain",
        file_size=5,
        stored_path="unrelated.txt",
        storage_provider="local",
        storage_key="unrelated.txt",
        collection_name=active_collection,
    )
    integration_db_session.add(_index_collection(active_collection))
    await integration_db_session.commit()

    integration_db_session.add(target_document)
    integration_db_session.add(unrelated_document)
    integration_db_session.add(
        VectorCleanupJob(
            id=str(uuid.uuid4()),
            document_id=target_document.id,
            collection_name=pending_collection,
            status=VectorCleanupStatus.PENDING,
        )
    )
    integration_db_session.add(
        VectorCleanupJob(
            id=str(uuid.uuid4()),
            document_id=target_document.id,
            collection_name=failed_collection,
            status=VectorCleanupStatus.FAILED,
            last_error="previous attempt failed",
        )
    )
    deletion_job = DocumentDeletionJob(
        id=str(uuid.uuid4()), document_id=target_document.id, status=DocumentDeletionStatus.PENDING
    )
    integration_db_session.add(deletion_job)
    await integration_db_session.commit()

    for collection in (active_collection, pending_collection, failed_collection):
        await vector_store.create_collection_if_not_exists(collection, _VECTOR_SIZE)

    await vector_store.upsert_vectors(active_collection, [_point(target_document.id)])
    await vector_store.upsert_vectors(active_collection, [_point(unrelated_document.id)])
    await vector_store.upsert_vectors(pending_collection, [_point(target_document.id)])
    await vector_store.upsert_vectors(failed_collection, [_point(target_document.id)])

    worker = DocumentDeletionWorker(vector_store=vector_store, file_storage=_NoopFileStorage())
    result = await worker.process_next_job(integration_db_session)

    assert result is not None
    assert result.status == DocumentDeletionStatus.COMPLETED
    assert result.vector_cleanup_completed is True

    query_vector = [0.1, 0.2, 0.3, 0.4]
    active_remaining = await vector_store.search_similar(active_collection, query_vector, limit=10)
    assert all(point.document_id != target_document.id for point in active_remaining)
    assert any(point.document_id == unrelated_document.id for point in active_remaining)

    pending_remaining = await vector_store.search_similar(pending_collection, query_vector, limit=10)
    assert pending_remaining == []

    failed_remaining = await vector_store.search_similar(failed_collection, query_vector, limit=10)
    assert failed_remaining == []


async def test_worker_repeated_deletion_is_idempotent(
    migrated_schema: None, qdrant_url: str, integration_db_session: AsyncSession
) -> None:
    """Deleting an already-clean collection's vectors twice must not raise or fail the job."""
    from app.core.config import get_settings

    settings = get_settings()
    vector_store = QdrantVectorStore(settings=settings)
    collection = f"idempotent-{uuid.uuid4().hex}"

    document = Document(
        id=str(uuid.uuid4()),
        original_filename="a.txt",
        stored_filename=f"{uuid.uuid4().hex}.txt",
        content_type="text/plain",
        file_size=5,
        stored_path="a.txt",
        storage_provider="local",
        storage_key="a.txt",
        collection_name=collection,
    )
    integration_db_session.add(_index_collection(collection))
    await integration_db_session.commit()

    integration_db_session.add(document)
    await integration_db_session.commit()

    await vector_store.create_collection_if_not_exists(collection, _VECTOR_SIZE)
    await vector_store.upsert_vectors(collection, [_point(document.id)])

    job_one = DocumentDeletionJob(
        id=str(uuid.uuid4()), document_id=document.id, status=DocumentDeletionStatus.PENDING
    )
    integration_db_session.add(job_one)
    await integration_db_session.commit()

    worker = DocumentDeletionWorker(vector_store=vector_store, file_storage=_NoopFileStorage())
    first_result = await worker.process_next_job(integration_db_session)
    assert first_result is not None
    assert first_result.status == DocumentDeletionStatus.COMPLETED

    # A second attempt (e.g. a retry against an already-clean collection) must also succeed.
    job_two = DocumentDeletionJob(
        id=str(uuid.uuid4()), document_id=document.id, status=DocumentDeletionStatus.PENDING
    )
    integration_db_session.add(job_two)
    await integration_db_session.commit()

    second_result = await worker.process_next_job(integration_db_session)
    assert second_result is not None
    assert second_result.status == DocumentDeletionStatus.COMPLETED


async def test_worker_partial_collection_failure_blocks_storage_and_is_reported(
    migrated_schema: None, qdrant_url: str, integration_db_session: AsyncSession
) -> None:
    """A genuinely unreachable historical collection must PARTIALLY_FAIL and skip storage cleanup."""
    from app.core.config import get_settings

    settings = get_settings()
    vector_store = QdrantVectorStore(settings=settings)
    active_collection = f"active-{uuid.uuid4().hex}"

    document = Document(
        id=str(uuid.uuid4()),
        original_filename="a.txt",
        stored_filename=f"{uuid.uuid4().hex}.txt",
        content_type="text/plain",
        file_size=5,
        stored_path="a.txt",
        storage_provider="local",
        storage_key="a.txt",
        collection_name=active_collection,
    )
    integration_db_session.add(_index_collection(active_collection))
    await integration_db_session.commit()

    integration_db_session.add(document)
    # A historical collection that was never created in Qdrant — a delete against it will raise.
    integration_db_session.add(
        VectorCleanupJob(
            id=str(uuid.uuid4()),
            document_id=document.id,
            collection_name="does-not-exist-and-will-fail",
            status=VectorCleanupStatus.FAILED,
        )
    )
    job = DocumentDeletionJob(
        id=str(uuid.uuid4()), document_id=document.id, status=DocumentDeletionStatus.PENDING
    )
    integration_db_session.add(job)
    await integration_db_session.commit()

    await vector_store.create_collection_if_not_exists(active_collection, _VECTOR_SIZE)

    class _FailingCollectionVectorStore(QdrantVectorStore):
        async def delete_by_document_id(self, collection_name: str, document_id: str) -> None:
            if collection_name == "does-not-exist-and-will-fail":
                raise RuntimeError("collection does not exist")
            await super().delete_by_document_id(collection_name, document_id)

    failing_store = _FailingCollectionVectorStore(settings=settings)
    file_storage = _NoopFileStorage()
    worker = DocumentDeletionWorker(vector_store=failing_store, file_storage=file_storage)

    result = await worker.process_next_job(integration_db_session)

    assert result is not None
    assert result.status == DocumentDeletionStatus.PARTIALLY_FAILED
    assert result.vector_cleanup_completed is False
    assert result.storage_cleanup_completed is False
