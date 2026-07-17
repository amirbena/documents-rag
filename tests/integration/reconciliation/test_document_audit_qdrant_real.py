"""Focused real-Qdrant tests for the document lifecycle audit (Phase 2.8.7, subtask 1) — real
Testcontainers Postgres + real Qdrant, no fake vector-store adapter.

Proves the audit's Qdrant inspection (`get_collection_vector_size`/`count_document_vectors`) is
correct against a real Qdrant instance. Object Storage is faked in this module — see
`test_document_audit_storage_real.py` for the real-Object-Storage counterpart.
"""

import uuid
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.models.document import Document
from app.models.index_collection import IndexCollection, IndexCollectionStatus
from app.rag.providers.qdrant_vector_store import QdrantVectorStore
from app.rag.providers.vector_store import VectorPoint
from app.services.reconciliation.document_audit_service import (
    DocumentLifecycleFindingCode,
    audit_document_lifecycle,
)


class _FakeFileStorage:
    async def exists(self, key: str) -> bool:
        return True


@pytest.fixture(autouse=True)
async def _clean_tables(migrated_schema: None, postgres_url: str) -> AsyncIterator[None]:
    engine = create_async_engine(postgres_url, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def _truncate() -> None:
        async with session_factory() as session:
            await session.execute(
                text("TRUNCATE TABLE ingestion_jobs, index_collections, documents RESTART IDENTITY CASCADE")
            )
            await session.commit()

    await _truncate()
    try:
        yield
    finally:
        await _truncate()
        await engine.dispose()


async def _seed_document(session: AsyncSession, collection_name: str, **overrides: object) -> Document:
    existing = await session.get(IndexCollection, collection_name)
    if existing is None:
        session.add(
            IndexCollection(
                collection_name=collection_name,
                embedding_provider="ollama",
                embedding_model="test-model",
                embedding_dimension=4,
                embedding_version="v1",
                chunking_version="v1",
                status=IndexCollectionStatus.ACTIVE,
            )
        )
        await session.commit()

    fields: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        original_filename="a.txt",
        stored_filename=f"{uuid.uuid4().hex}.txt",
        content_type="text/plain",
        file_size=5,
        stored_path="documents/a/a.txt",
        storage_provider="local",
        storage_key="documents/a/a.txt",
        collection_name=collection_name,
        embedding_provider="ollama",
        embedding_model="test-model",
        embedding_dimension=4,
        embedding_version="v1",
        chunking_version="v1",
    )
    fields.update(overrides)
    document = Document(**fields)  # type: ignore[arg-type]
    session.add(document)
    await session.commit()
    return document


async def test_active_collection_with_document_vectors_is_healthy(
    migrated_schema: None, qdrant_url: str, integration_db_session: AsyncSession
) -> None:
    settings = get_settings()
    vector_store = QdrantVectorStore(settings=settings)
    collection_name = f"audit-healthy-{uuid.uuid4().hex}"

    document = await _seed_document(integration_db_session, collection_name)
    unrelated_document_id = str(uuid.uuid4())

    await vector_store.create_collection_if_not_exists(collection_name, 4)
    await vector_store.upsert_vectors(
        collection_name,
        [
            VectorPoint(
                id=str(uuid.uuid4()),
                vector=[0.1, 0.2, 0.3, 0.4],
                document_id=document.id,
                chunk_id="chunk-1",
                text="hello",
                source="a.txt",
            ),
            VectorPoint(
                id=str(uuid.uuid4()),
                vector=[0.5, 0.6, 0.7, 0.8],
                document_id=unrelated_document_id,
                chunk_id="chunk-1",
                text="unrelated",
                source="b.txt",
            ),
        ],
    )

    result = await audit_document_lifecycle(
        integration_db_session, document.id, settings, _FakeFileStorage(), vector_store
    )

    assert result.vector_state is not None
    assert result.vector_state.collection_exists is True
    assert result.vector_state.has_vectors is True
    codes = {f.code for f in result.findings}
    assert DocumentLifecycleFindingCode.ACTIVE_COLLECTION_MISSING not in codes
    assert DocumentLifecycleFindingCode.ACTIVE_VECTORS_MISSING not in codes

    # Preserve unrelated vectors — the audit never deletes anything.
    unrelated_count = await vector_store.count_document_vectors(collection_name, unrelated_document_id)
    assert unrelated_count == 1
    own_count = await vector_store.count_document_vectors(collection_name, document.id)
    assert own_count == 1


async def test_active_collection_without_document_vectors_produces_active_vectors_missing(
    migrated_schema: None, qdrant_url: str, integration_db_session: AsyncSession
) -> None:
    settings = get_settings()
    vector_store = QdrantVectorStore(settings=settings)
    collection_name = f"audit-empty-{uuid.uuid4().hex}"

    document = await _seed_document(integration_db_session, collection_name)
    unrelated_document_id = str(uuid.uuid4())

    await vector_store.create_collection_if_not_exists(collection_name, 4)
    # Only an unrelated document's vectors exist — none for the audited document.
    await vector_store.upsert_vectors(
        collection_name,
        [
            VectorPoint(
                id=str(uuid.uuid4()),
                vector=[0.5, 0.6, 0.7, 0.8],
                document_id=unrelated_document_id,
                chunk_id="chunk-1",
                text="unrelated",
                source="b.txt",
            )
        ],
    )

    result = await audit_document_lifecycle(
        integration_db_session, document.id, settings, _FakeFileStorage(), vector_store
    )

    assert result.vector_state is not None
    assert result.vector_state.collection_exists is True
    assert result.vector_state.has_vectors is False
    assert DocumentLifecycleFindingCode.ACTIVE_VECTORS_MISSING in {f.code for f in result.findings}

    # Preserve unrelated vectors.
    unrelated_count = await vector_store.count_document_vectors(collection_name, unrelated_document_id)
    assert unrelated_count == 1


async def test_missing_active_collection_produces_active_collection_missing(
    migrated_schema: None, qdrant_url: str, integration_db_session: AsyncSession
) -> None:
    settings = get_settings()
    vector_store = QdrantVectorStore(settings=settings)
    collection_name = f"audit-never-created-{uuid.uuid4().hex}"

    document = await _seed_document(integration_db_session, collection_name)

    result = await audit_document_lifecycle(
        integration_db_session, document.id, settings, _FakeFileStorage(), vector_store
    )

    assert result.vector_state is not None
    assert result.vector_state.collection_exists is False
    assert DocumentLifecycleFindingCode.ACTIVE_COLLECTION_MISSING in {f.code for f in result.findings}
