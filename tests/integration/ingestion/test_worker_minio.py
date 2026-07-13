"""Integration test for the full ingestion chain against real Postgres + real Qdrant + real MinIO.

Upload (Document row + MinIO object) -> IngestionWorker reads via MinioFileStorage -> extraction
-> chunking -> fake embeddings (never a real Ollama call) -> real Qdrant upsert. Confirms the
worker never takes a local-filesystem shortcut to read content that was saved to MinIO.
"""

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.services.ingestion.worker as ingestion_worker_module
from app.core.config import Settings, get_settings
from app.models.document import Document
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.rag.embedding_config import get_active_embedding_config
from app.rag.providers.qdrant_vector_store import QdrantVectorStore
from app.services.ingestion.worker import IngestionWorker
from app.storage.keys import generate_object_key
from app.storage.minio_storage import MinioFileStorage


class _FakeEmbeddingProvider:
    """Returns one fixed-length deterministic vector per text — no real Ollama call."""

    def __init__(self, vector_size: int) -> None:
        self._vector = [0.1] * vector_size

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector for _ in texts]


@asynccontextmanager
async def _new_session(postgres_url: str) -> AsyncIterator[AsyncSession]:
    """Open a fresh AsyncSession on its own dedicated engine/connection."""
    engine = create_async_engine(postgres_url, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with session_factory() as session:
        yield session
    await engine.dispose()


@pytest.fixture
def minio_storage(minio_endpoint: str, minio_credentials: tuple[str, str]) -> MinioFileStorage:
    """A MinioFileStorage pointed at the ephemeral container, with its bucket ensured lazily."""
    access_key, secret_key = minio_credentials
    settings = Settings(
        FILE_STORAGE_PROVIDER="minio",
        MINIO_ENDPOINT=minio_endpoint,
        MINIO_ACCESS_KEY=access_key,
        MINIO_SECRET_KEY=secret_key,
        MINIO_BUCKET="documents-ingestion-test",
        MINIO_SECURE=False,
    )
    return MinioFileStorage(settings=settings)


async def test_ingestion_reads_content_from_minio_not_the_local_filesystem(
    migrated_schema: None,
    postgres_url: str,
    qdrant_url: str,
    minio_storage: MinioFileStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full pipeline should extract/chunk/embed/upsert content that only ever lived in MinIO."""
    await minio_storage.ensure_bucket()

    settings = get_settings()
    collection_prefix = f"minio-ingestion-{uuid.uuid4().hex}"
    monkeypatch.setattr(settings, "qdrant_collection_name", collection_prefix)
    monkeypatch.setattr(
        ingestion_worker_module,
        "get_embedding_provider",
        lambda settings=None: _FakeEmbeddingProvider(get_settings().vector_size),
    )
    active_config = get_active_embedding_config(settings)

    document_id = str(uuid.uuid4())
    key = generate_object_key(document_id, "notes.txt")
    content = b"hello from minio " * 50
    await minio_storage.save(key, content, content_type="text/plain")

    async with _new_session(postgres_url) as session:
        document = Document(
            id=document_id,
            original_filename="notes.txt",
            stored_filename=key.rsplit("/", 1)[-1],
            content_type="text/plain",
            file_size=len(content),
            stored_path=key,
            storage_provider="minio",
            storage_bucket="documents-ingestion-test",
            storage_key=key,
        )
        session.add(document)
        job = IngestionJob(id=str(uuid.uuid4()), document_id=document.id, status=IngestionStatus.PENDING)
        session.add(job)
        await session.commit()

        worker = IngestionWorker(file_storage=minio_storage)
        result = await worker.process_next_job(session)

        assert result is not None
        assert result.status == IngestionStatus.COMPLETED
        assert result.error_message is None

        indexed_document = await session.get(Document, document_id)
        assert indexed_document is not None
        assert indexed_document.indexed_at is not None
        assert indexed_document.collection_name == active_config.collection_name

    vector_store = QdrantVectorStore(settings=settings)
    query_vector = [0.1] * settings.vector_size
    results = await vector_store.search_similar(active_config.collection_name, query_vector, limit=10)

    assert len(results) > 0
    assert all(result.document_id == document_id for result in results)
