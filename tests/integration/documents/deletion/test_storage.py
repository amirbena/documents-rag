"""Storage-cleanup-step tests for DocumentDeletionWorker: real LocalFileStorage + real MinIO.

Proves Local and MinIO both satisfy the same FileStorage contract from the worker's point of
view: the original object is deleted, an already-missing object is treated idempotently
(success, not failure), and a genuinely unreachable storage backend produces a PARTIALLY_FAILED
job rather than a false success — never a raw provider exception surfacing to the job record's
public-facing fields.
"""

import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings
from app.models.document import Document
from app.models.document_deletion_job import DocumentDeletionJob, DocumentDeletionStatus
from app.services.documents.deletion_service import DeletionErrorCode
from app.services.documents.deletion_worker import DocumentDeletionWorker
from app.storage.local_storage import LocalFileStorage
from app.storage.minio_storage import MinioFileStorage


class _NoopVectorStore:
    """Vectors are out of scope for these tests — every delete call succeeds trivially."""

    async def delete_by_document_id(self, collection_name: str, document_id: str) -> None:
        return None


def _document(storage_key: str, **overrides: object) -> Document:
    fields: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        original_filename="a.txt",
        stored_filename=f"{uuid.uuid4().hex}.txt",
        content_type="text/plain",
        file_size=5,
        stored_path=storage_key,
        storage_provider="local",
        storage_key=storage_key,
    )
    fields.update(overrides)
    return Document(**fields)  # type: ignore[arg-type]


def _deletion_job(document_id: str) -> DocumentDeletionJob:
    return DocumentDeletionJob(
        id=str(uuid.uuid4()), document_id=document_id, status=DocumentDeletionStatus.PENDING
    )


@pytest.fixture(autouse=True)
async def _clean_tables(migrated_schema: None, postgres_url: str) -> AsyncIterator[None]:
    engine = create_async_engine(postgres_url, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def _truncate() -> None:
        async with session_factory() as session:
            await session.execute(
                text(
                    "TRUNCATE TABLE document_deletion_jobs, ingestion_jobs, documents "
                    "RESTART IDENTITY CASCADE"
                )
            )
            await session.commit()

    await _truncate()
    try:
        yield
    finally:
        await _truncate()
        await engine.dispose()


async def test_worker_deletes_local_storage_object(
    migrated_schema: None, integration_db_session, tmp_path: Path
) -> None:
    storage = LocalFileStorage(root=tmp_path)
    key = "documents/a/notes.txt"
    await storage.save(key, b"hello world")

    document = _document(key)
    integration_db_session.add(document)
    job = _deletion_job(document.id)
    integration_db_session.add(job)
    await integration_db_session.commit()

    worker = DocumentDeletionWorker(vector_store=_NoopVectorStore(), file_storage=storage)
    result = await worker.process_next_job(integration_db_session)

    assert result is not None
    assert result.status == DocumentDeletionStatus.COMPLETED
    assert result.storage_cleanup_completed is True
    assert await storage.exists(key) is False


async def test_worker_local_missing_object_is_idempotent_success(
    migrated_schema: None, integration_db_session, tmp_path: Path
) -> None:
    """A previously-deleted (or never-saved) local object must not block COMPLETED."""
    storage = LocalFileStorage(root=tmp_path)
    key = "documents/a/already-gone.txt"

    document = _document(key)
    integration_db_session.add(document)
    job = _deletion_job(document.id)
    integration_db_session.add(job)
    await integration_db_session.commit()

    worker = DocumentDeletionWorker(vector_store=_NoopVectorStore(), file_storage=storage)
    result = await worker.process_next_job(integration_db_session)

    assert result is not None
    assert result.status == DocumentDeletionStatus.COMPLETED


async def test_worker_deletes_minio_storage_object(
    migrated_schema: None, integration_db_session, minio_endpoint: str, minio_credentials: tuple[str, str]
) -> None:
    access_key, secret_key = minio_credentials
    settings = Settings(
        FILE_STORAGE_PROVIDER="minio",
        MINIO_ENDPOINT=minio_endpoint,
        MINIO_ACCESS_KEY=access_key,
        MINIO_SECRET_KEY=secret_key,
        MINIO_BUCKET="document-deletion-integration-test",
        MINIO_SECURE=False,
    )
    storage = MinioFileStorage(settings=settings)
    await storage.ensure_bucket()

    key = f"documents/{uuid.uuid4().hex}/notes.txt"
    await storage.save(key, b"hello world")

    document = _document(key)
    integration_db_session.add(document)
    job = _deletion_job(document.id)
    integration_db_session.add(job)
    await integration_db_session.commit()

    worker = DocumentDeletionWorker(vector_store=_NoopVectorStore(), file_storage=storage)
    result = await worker.process_next_job(integration_db_session)

    assert result is not None
    assert result.status == DocumentDeletionStatus.COMPLETED
    assert result.storage_cleanup_completed is True
    assert await storage.exists(key) is False


async def test_worker_minio_missing_object_is_idempotent_success(
    migrated_schema: None, integration_db_session, minio_endpoint: str, minio_credentials: tuple[str, str]
) -> None:
    access_key, secret_key = minio_credentials
    settings = Settings(
        FILE_STORAGE_PROVIDER="minio",
        MINIO_ENDPOINT=minio_endpoint,
        MINIO_ACCESS_KEY=access_key,
        MINIO_SECRET_KEY=secret_key,
        MINIO_BUCKET="document-deletion-integration-test",
        MINIO_SECURE=False,
    )
    storage = MinioFileStorage(settings=settings)
    await storage.ensure_bucket()

    key = f"documents/{uuid.uuid4().hex}/already-gone.txt"
    document = _document(key)
    integration_db_session.add(document)
    job = _deletion_job(document.id)
    integration_db_session.add(job)
    await integration_db_session.commit()

    worker = DocumentDeletionWorker(vector_store=_NoopVectorStore(), file_storage=storage)
    result = await worker.process_next_job(integration_db_session)

    assert result is not None
    assert result.status == DocumentDeletionStatus.COMPLETED


async def test_worker_minio_unreachable_marks_partially_failed_not_completed(
    migrated_schema: None, integration_db_session
) -> None:
    """A genuinely unreachable MinIO endpoint must PARTIALLY_FAIL after vectors, never COMPLETE."""
    settings = Settings(
        FILE_STORAGE_PROVIDER="minio",
        MINIO_ENDPOINT="127.0.0.1:1",  # nothing listens here
        MINIO_ACCESS_KEY="test",
        MINIO_SECRET_KEY="testtest",
        MINIO_BUCKET="document-deletion-integration-test",
        MINIO_SECURE=False,
    )
    storage = MinioFileStorage(settings=settings)

    document = _document("documents/a/unreachable.txt", content_hash="a" * 64)
    integration_db_session.add(document)
    job = _deletion_job(document.id)
    integration_db_session.add(job)
    await integration_db_session.commit()

    worker = DocumentDeletionWorker(vector_store=_NoopVectorStore(), file_storage=storage)
    result = await worker.process_next_job(integration_db_session)

    assert result is not None
    assert result.status == DocumentDeletionStatus.PARTIALLY_FAILED
    assert result.vector_cleanup_completed is True
    assert result.storage_cleanup_completed is False
    assert result.error_code == DeletionErrorCode.DOCUMENT_STORAGE_CLEANUP_FAILED
    # A storage-cleanup failure (PARTIALLY_FAILED, not COMPLETED) must never release the hash.
    assert document.content_hash == "a" * 64


async def test_worker_completed_deletion_clears_content_hash(
    migrated_schema: None, integration_db_session, tmp_path: Path
) -> None:
    """The COMPLETED transition must release content_hash back to NULL, against real Postgres."""
    storage = LocalFileStorage(root=tmp_path)
    key = "documents/a/notes.txt"
    await storage.save(key, b"hello world")

    document = _document(key, content_hash="b" * 64)
    integration_db_session.add(document)
    job = _deletion_job(document.id)
    integration_db_session.add(job)
    await integration_db_session.commit()

    worker = DocumentDeletionWorker(vector_store=_NoopVectorStore(), file_storage=storage)
    result = await worker.process_next_job(integration_db_session)

    assert result is not None
    assert result.status == DocumentDeletionStatus.COMPLETED
    assert document.content_hash is None

    stored_value = await integration_db_session.execute(
        text("SELECT content_hash FROM documents WHERE id = :id"), {"id": document.id}
    )
    assert stored_value.scalar_one() is None


async def test_completed_deletion_allows_a_later_document_to_claim_the_same_hash(
    migrated_schema: None, integration_db_session, tmp_path: Path
) -> None:
    """Once a document's deletion COMPLETEs and releases its hash, a genuinely new Document may
    claim that same content_hash without violating uq_documents_content_hash.
    """
    storage = LocalFileStorage(root=tmp_path)
    key = "documents/a/notes.txt"
    await storage.save(key, b"hello world")
    shared_hash = "c" * 64

    document = _document(key, content_hash=shared_hash)
    integration_db_session.add(document)
    job = _deletion_job(document.id)
    integration_db_session.add(job)
    await integration_db_session.commit()

    worker = DocumentDeletionWorker(vector_store=_NoopVectorStore(), file_storage=storage)
    result = await worker.process_next_job(integration_db_session)
    assert result is not None
    assert result.status == DocumentDeletionStatus.COMPLETED

    later_document = _document("documents/b/notes-again.txt", content_hash=shared_hash)
    integration_db_session.add(later_document)
    await integration_db_session.commit()  # must not raise IntegrityError

    hash_count = await integration_db_session.execute(
        text("SELECT count(*) FROM documents WHERE content_hash = :hash"), {"hash": shared_hash}
    )
    assert hash_count.scalar_one() == 1
