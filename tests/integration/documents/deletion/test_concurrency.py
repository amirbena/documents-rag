"""Concurrency stress tests for document deletion — real Testcontainers Postgres, genuine races.

Kept separate from test_postgres.py's ordinary persistence/lifecycle tests, per this repo's
convention (mirrors tests/integration/test_ingestion_retry_postgres.py's equivalent split): these
tests use two independent AsyncSessions (separate connections) racing via `asyncio.gather`, not
two calls sharing one session/transaction, which would prove nothing about real locking. Each
scenario is internally parametrized and repeated (5x for scheduling, 3x for worker claims) to
catch any residual flakiness rather than relying on a single lucky run.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.document import Document
from app.models.document_deletion_job import DocumentDeletionJob, DocumentDeletionStatus
from app.services.documents.deletion_service import DeletionRequestOutcome, request_document_deletion
from app.services.documents.deletion_worker import DocumentDeletionWorker
from app.storage.errors import StorageObjectNotFoundError


class _NoopVectorStore:
    """A VectorStore whose deletes always succeed instantly — worker-claim tests don't need real Qdrant."""

    async def delete_by_document_id(self, collection_name: str, document_id: str) -> None:
        return None


class _NoopFileStorage:
    """A FileStorage whose delete always succeeds instantly (idempotent no-op, per contract)."""

    async def delete(self, key: str) -> None:
        return None

    async def save(self, key: str, content: bytes) -> None:  # pragma: no cover - unused
        raise NotImplementedError

    async def read(self, key: str) -> bytes:
        raise StorageObjectNotFoundError(key)

    async def exists(self, key: str) -> bool:  # pragma: no cover - unused
        return False


@pytest.fixture(autouse=True)
async def _clean_tables(migrated_schema: None, postgres_url: str) -> AsyncIterator[None]:
    """Truncate deletion/ingestion/document tables before and after each test in this module."""
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


async def _seed_document(session: AsyncSession) -> Document:
    document = Document(
        id=str(uuid.uuid4()),
        original_filename="report.pdf",
        stored_filename=f"{uuid.uuid4().hex}.pdf",
        content_type="application/pdf",
        file_size=10,
        stored_path="storage/documents/report.pdf",
        storage_provider="local",
        storage_key=f"storage/documents/{uuid.uuid4().hex}.pdf",
    )
    session.add(document)
    await session.commit()
    return document


@pytest.mark.parametrize("run", range(5))
async def test_concurrent_delete_requests_produce_exactly_one_active_job(
    run: int, migrated_schema: None, postgres_url: str, integration_db_session: AsyncSession
) -> None:
    """Repeated (5x) concurrency stress: two genuinely concurrent DELETE schedulings -> one job."""
    document = await _seed_document(integration_db_session)

    engine = create_async_engine(postgres_url, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def _request_with_own_session() -> DeletionRequestOutcome:
        async with session_factory() as session:
            result = await request_document_deletion(session, document.id)
            return result.outcome

    try:
        outcomes = await asyncio.gather(_request_with_own_session(), _request_with_own_session())
    finally:
        await engine.dispose()

    assert sorted(outcomes) == sorted(
        [DeletionRequestOutcome.CREATED, DeletionRequestOutcome.ALREADY_ACTIVE]
    )

    active_rows = await integration_db_session.execute(
        text(
            "SELECT count(*) FROM document_deletion_jobs WHERE document_id = :id "
            "AND status IN ('pending', 'processing')"
        ),
        {"id": document.id},
    )
    assert active_rows.scalar_one() == 1


@pytest.mark.parametrize("run", range(3))
async def test_concurrent_worker_claims_never_claim_the_same_job_twice(
    run: int, migrated_schema: None, postgres_url: str, integration_db_session: AsyncSession
) -> None:
    """Repeated (3x) concurrency stress: two workers racing to claim distinct PENDING jobs."""
    doc_a = await _seed_document(integration_db_session)
    doc_b = await _seed_document(integration_db_session)
    integration_db_session.add(
        DocumentDeletionJob(id=str(uuid.uuid4()), document_id=doc_a.id, status=DocumentDeletionStatus.PENDING)
    )
    integration_db_session.add(
        DocumentDeletionJob(id=str(uuid.uuid4()), document_id=doc_b.id, status=DocumentDeletionStatus.PENDING)
    )
    await integration_db_session.commit()

    engine = create_async_engine(postgres_url, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def _claim_with_own_session() -> str | None:
        worker = DocumentDeletionWorker(vector_store=_NoopVectorStore(), file_storage=_NoopFileStorage())
        async with session_factory() as session:
            job = await worker.process_next_job(session)
            return job.id if job is not None else None

    try:
        first_id, second_id = await asyncio.gather(
            _claim_with_own_session(), _claim_with_own_session()
        )
    finally:
        await engine.dispose()

    assert first_id is not None
    assert second_id is not None
    assert first_id != second_id

    completed = await integration_db_session.execute(
        text("SELECT count(*) FROM document_deletion_jobs WHERE status = 'completed'")
    )
    assert completed.scalar_one() == 2
