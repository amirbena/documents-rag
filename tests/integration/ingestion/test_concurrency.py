"""Postgres concurrency integration tests for ingestion retry and stale-job recovery.

Separated from tests/integration/ingestion/test_retry_postgres.py to isolate the genuinely
concurrent (`asyncio.gather` over independent sessions/connections) tests from the single-session
persistence/migration tests — mirrors tests/integration/documents/deletion/test_concurrency.py's
separation rationale.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.document import Document
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.services.ingestion.retry_service import RetryOutcome, retry_ingestion
from app.services.ingestion.stale_recovery_service import recover_stale_ingestion_jobs

STALE_AFTER_SECONDS = 900


@pytest.fixture(autouse=True)
async def _clean_tables(migrated_schema: None, postgres_url: str) -> AsyncIterator[None]:
    """Truncate documents/ingestion_jobs before and after each test — see test_retry_postgres.py."""
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


async def test_two_concurrent_retries_produce_exactly_one_new_active_job(
    migrated_schema: None, postgres_url: str, integration_db_session: AsyncSession
) -> None:
    """The single most important test: two genuinely concurrent retries -> exactly one new job.

    Uses two independent AsyncSessions (independent connections) so the two `retry_ingestion()`
    calls run as real, separate Postgres transactions racing each other via `asyncio.gather` —
    not two calls sharing one session/transaction, which would prove nothing about real locking.
    """
    document = await _seed_document(integration_db_session)
    integration_db_session.add(
        IngestionJob(id=str(uuid.uuid4()), document_id=document.id, status=IngestionStatus.FAILED)
    )
    await integration_db_session.commit()

    engine = create_async_engine(postgres_url, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def _retry_with_own_session() -> RetryOutcome:
        async with session_factory() as session:
            result = await retry_ingestion(
                session, document.id, stale_after_seconds=STALE_AFTER_SECONDS
            )
            return result.outcome

    try:
        outcomes = await asyncio.gather(_retry_with_own_session(), _retry_with_own_session())
    finally:
        await engine.dispose()

    # Exactly one of the two concurrent calls created a new job; the other observed it as active.
    assert sorted(outcomes) == sorted([RetryOutcome.CREATED, RetryOutcome.ALREADY_ACTIVE])

    active_rows = await integration_db_session.execute(
        text(
            "SELECT count(*) FROM ingestion_jobs WHERE document_id = :id "
            "AND status IN ('pending', 'processing')"
        ),
        {"id": document.id},
    )
    assert active_rows.scalar_one() == 1


async def test_two_concurrent_recoveries_never_recover_the_same_stale_row_twice(
    migrated_schema: None, postgres_url: str, integration_db_session: AsyncSession
) -> None:
    """Two concurrent recovery batches must never both create a replacement for the same stale row."""
    document = await _seed_document(integration_db_session)
    stale_updated_at = datetime.now(UTC) - timedelta(seconds=STALE_AFTER_SECONDS + 100)
    integration_db_session.add(
        IngestionJob(
            id=str(uuid.uuid4()),
            document_id=document.id,
            status=IngestionStatus.PROCESSING,
            updated_at=stale_updated_at,
        )
    )
    await integration_db_session.commit()
    # Force the just-inserted row's server-generated updated_at (onupdate=func.now() only fires
    # on UPDATE, not INSERT's server_default) back to a genuinely stale timestamp.
    await integration_db_session.execute(
        text("UPDATE ingestion_jobs SET updated_at = :ts WHERE document_id = :id"),
        {"ts": stale_updated_at, "id": document.id},
    )
    await integration_db_session.commit()

    engine = create_async_engine(postgres_url, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def _recover_with_own_session():
        async with session_factory() as session:
            return await recover_stale_ingestion_jobs(
                session, batch_size=10, stale_after_seconds=STALE_AFTER_SECONDS
            )

    try:
        results = await asyncio.gather(_recover_with_own_session(), _recover_with_own_session())
    finally:
        await engine.dispose()

    total_recovered = sum(result.count for result in results)
    assert total_recovered == 1

    replacement_count = await integration_db_session.execute(
        text(
            "SELECT count(*) FROM ingestion_jobs WHERE document_id = :id AND status = 'pending'"
        ),
        {"id": document.id},
    )
    assert replacement_count.scalar_one() == 1
