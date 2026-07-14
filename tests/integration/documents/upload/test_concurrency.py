"""Postgres concurrency integration tests for content-hash-deduplicated upload.

Mirrors tests/integration/ingestion/test_concurrency.py's and
tests/integration/documents/deletion/test_concurrency.py's separation rationale: genuinely
concurrent (`asyncio.gather` over independent sessions/connections) tests, isolated from
single-session persistence tests. Real `LocalFileStorage` on a shared `tmp_path` root stands in
for both concurrent callers' storage backend — the database's `uq_documents_content_hash` unique
index is what's actually under test, not the storage layer.
"""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.services.documents.dedup_service import UploadOutcome
from app.services.documents.upload_service import UploadResult, upload_document
from app.storage.local_storage import LocalFileStorage


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


@pytest.mark.parametrize("run", range(5))
async def test_two_concurrent_identical_uploads_converge_on_one_document(
    migrated_schema: None, postgres_url: str, tmp_path: Path, run: int
) -> None:
    """The single most important test: two genuinely concurrent identical uploads -> exactly one
    document, one ingestion job, both callers reporting the same identities.

    Uses two independent AsyncSessions (independent connections) and independent LocalFileStorage
    instances sharing the same root, so the two `upload_document()` calls run as real, separate
    Postgres transactions racing each other via `asyncio.gather` — not two calls sharing one
    session/transaction, which would prove nothing about the real unique-index race.
    """
    content = f"identical bytes for run {run}".encode()

    engine = create_async_engine(postgres_url, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def _upload_with_own_session() -> UploadResult:
        async with session_factory() as session:
            storage = LocalFileStorage(root=tmp_path)
            return await upload_document(
                content=content,
                original_filename=f"report-{run}.pdf",
                content_type="application/pdf",
                storage=storage,
                session=session,
            )

    try:
        results = await asyncio.gather(_upload_with_own_session(), _upload_with_own_session())
    finally:
        await engine.dispose()

    document_ids = {result.document.id for result in results}
    job_ids = {result.ingestion_job.id for result in results}
    outcomes = sorted(result.outcome for result in results)

    assert len(document_ids) == 1, "both uploads must report the same document_id"
    assert len(job_ids) == 1, "both uploads must report the same ingestion job identity"
    assert outcomes[0] == UploadOutcome.CREATED
    assert outcomes[1] in (
        UploadOutcome.REUSED_ACTIVE,
        UploadOutcome.REUSED_INDEXED,
        UploadOutcome.REUSED_FAILED,
    )

    async with session_factory() as verify_session:
        content_hash = results[0].document.content_hash
        doc_count = await verify_session.execute(
            text("SELECT count(*) FROM documents WHERE content_hash = :hash"), {"hash": content_hash}
        )
        assert doc_count.scalar_one() == 1

        job_count = await verify_session.execute(
            text("SELECT count(*) FROM ingestion_jobs WHERE document_id = :id"),
            {"id": results[0].document.id},
        )
        assert job_count.scalar_one() == 1


async def test_distinct_uploads_create_two_documents(
    migrated_schema: None, postgres_url: str, tmp_path: Path, integration_db_session: AsyncSession
) -> None:
    """Two different byte sequences must never be deduplicated against each other."""
    storage = LocalFileStorage(root=tmp_path)

    first = await upload_document(
        content=b"first document's bytes",
        original_filename="first.pdf",
        content_type="application/pdf",
        storage=storage,
        session=integration_db_session,
    )
    second = await upload_document(
        content=b"second document's bytes",
        original_filename="second.pdf",
        content_type="application/pdf",
        storage=storage,
        session=integration_db_session,
    )

    assert first.outcome == UploadOutcome.CREATED
    assert second.outcome == UploadOutcome.CREATED
    assert first.document.id != second.document.id
    assert first.ingestion_job.id != second.ingestion_job.id
    assert first.document.content_hash != second.document.content_hash

    doc_count = await integration_db_session.execute(text("SELECT count(*) FROM documents"))
    assert doc_count.scalar_one() == 2

    job_count = await integration_db_session.execute(text("SELECT count(*) FROM ingestion_jobs"))
    assert job_count.scalar_one() == 2
