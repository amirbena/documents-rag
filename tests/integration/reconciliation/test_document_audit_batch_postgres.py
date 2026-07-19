"""Postgres integration tests for the batch document lifecycle audit (Phase 2.8.7, subtask 2) —
real Testcontainers Postgres, fake Object Storage/Qdrant adapters.

Proves the batch service's keyset-pagination query correctness (ascending ordering, id
tiebreaking, cursor boundary correctness, cross-document isolation, no persisted mutation)
against real rows — properties a fake session double cannot faithfully represent. Object
Storage/Qdrant are faked here deliberately, matching test_document_audit_postgres.py; this module
does not repeat that module's single-document finding matrix.
"""

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.models.document import Document
from app.models.document_deletion_job import DocumentDeletionJob, DocumentDeletionStatus
from app.models.index_collection import IndexCollection, IndexCollectionStatus
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.services.reconciliation.document_audit_batch_service import (
    InvalidAuditCursorError,
    audit_document_lifecycle_batch,
    decode_audit_cursor,
)


class _FakeFileStorage:
    async def exists(self, key: str) -> bool:
        return True


class _FakeVectorStore:
    async def get_collection_vector_size(self, collection_name: str) -> int | None:
        return None

    async def count_document_vectors(self, collection_name: str, document_id: str) -> int:
        return 0


@pytest.fixture(autouse=True)
async def _clean_tables(migrated_schema: None, postgres_url: str) -> AsyncIterator[None]:
    engine = create_async_engine(postgres_url, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def _truncate() -> None:
        async with session_factory() as session:
            await session.execute(
                text(
                    "TRUNCATE TABLE vector_cleanup_jobs, reindex_jobs, document_deletion_jobs, "
                    "ingestion_jobs, index_collections, documents RESTART IDENTITY CASCADE"
                )
            )
            await session.commit()

    await _truncate()
    try:
        yield
    finally:
        await _truncate()
        await engine.dispose()


async def _seed_document(session: AsyncSession, *, created_at: datetime, **overrides: object) -> Document:
    fields: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        original_filename="report.pdf",
        stored_filename=f"{uuid.uuid4().hex}.pdf",
        content_type="application/pdf",
        file_size=10,
        stored_path="storage/documents/report.pdf",
        storage_provider="local",
        storage_key=f"storage/documents/{uuid.uuid4().hex}.pdf",
        created_at=created_at,
    )
    fields.update(overrides)
    document = Document(**fields)  # type: ignore[arg-type]
    session.add(document)
    await session.commit()
    return document


async def _row_counts(session: AsyncSession) -> dict[str, int]:
    counts = {}
    for table in (
        "documents",
        "ingestion_jobs",
        "document_deletion_jobs",
        "reindex_jobs",
        "vector_cleanup_jobs",
    ):
        result = await session.execute(text(f"SELECT count(*) FROM {table}"))
        counts[table] = result.scalar_one()
    return counts


async def test_ascending_keyset_ordering_persists_across_real_rows(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    doc_1 = await _seed_document(integration_db_session, created_at=datetime(2026, 1, 1, tzinfo=UTC))
    doc_3 = await _seed_document(integration_db_session, created_at=datetime(2026, 1, 3, tzinfo=UTC))
    doc_2 = await _seed_document(integration_db_session, created_at=datetime(2026, 1, 2, tzinfo=UTC))

    result = await audit_document_lifecycle_batch(
        integration_db_session, get_settings(), _FakeFileStorage(), _FakeVectorStore(), limit=10
    )

    assert [summary.document_id for summary in result.documents] == [doc_1.id, doc_2.id, doc_3.id]


async def test_equal_created_at_values_are_ordered_by_id(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    same_time = datetime(2026, 1, 1, tzinfo=UTC)
    doc_a = await _seed_document(integration_db_session, created_at=same_time)
    doc_b = await _seed_document(integration_db_session, created_at=same_time)
    expected_order = sorted([doc_a.id, doc_b.id])

    result = await audit_document_lifecycle_batch(
        integration_db_session, get_settings(), _FakeFileStorage(), _FakeVectorStore(), limit=10
    )

    assert [summary.document_id for summary in result.documents] == expected_order


async def test_cursor_boundary_on_duplicated_timestamp_skips_no_rows(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    same_time = datetime(2026, 1, 1, tzinfo=UTC)
    docs = sorted(
        [
            await _seed_document(integration_db_session, created_at=same_time)
            for _ in range(3)
        ],
        key=lambda d: d.id,
    )

    first_page = await audit_document_lifecycle_batch(
        integration_db_session, get_settings(), _FakeFileStorage(), _FakeVectorStore(), limit=2
    )
    assert [s.document_id for s in first_page.documents] == [docs[0].id, docs[1].id]
    assert first_page.has_more is True

    second_page = await audit_document_lifecycle_batch(
        integration_db_session,
        get_settings(),
        _FakeFileStorage(),
        _FakeVectorStore(),
        limit=2,
        cursor=first_page.next_cursor,
    )
    assert [s.document_id for s in second_page.documents] == [docs[2].id]
    assert second_page.has_more is False


async def test_consecutive_pages_contain_no_duplicate_document(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    docs = [
        await _seed_document(integration_db_session, created_at=datetime(2026, 1, i + 1, tzinfo=UTC))
        for i in range(5)
    ]
    seen: list[str] = []
    cursor: str | None = None
    while True:
        page = await audit_document_lifecycle_batch(
            integration_db_session,
            get_settings(),
            _FakeFileStorage(),
            _FakeVectorStore(),
            limit=2,
            cursor=cursor,
        )
        seen.extend(summary.document_id for summary in page.documents)
        if not page.has_more:
            break
        cursor = page.next_cursor

    assert seen == [d.id for d in docs]
    assert len(seen) == len(set(seen))


async def test_consecutive_pages_cover_the_expected_full_set(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    docs = {
        (await _seed_document(integration_db_session, created_at=datetime(2026, 1, i + 1, tzinfo=UTC))).id
        for i in range(7)
    }
    seen: set[str] = set()
    cursor: str | None = None
    while True:
        page = await audit_document_lifecycle_batch(
            integration_db_session,
            get_settings(),
            _FakeFileStorage(),
            _FakeVectorStore(),
            limit=3,
            cursor=cursor,
        )
        seen.update(summary.document_id for summary in page.documents)
        if not page.has_more:
            break
        cursor = page.next_cursor

    assert seen == docs


async def test_lookahead_produces_correct_has_more(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    for i in range(4):
        await _seed_document(integration_db_session, created_at=datetime(2026, 1, i + 1, tzinfo=UTC))

    exact_page = await audit_document_lifecycle_batch(
        integration_db_session, get_settings(), _FakeFileStorage(), _FakeVectorStore(), limit=4
    )
    assert exact_page.has_more is False

    short_page = await audit_document_lifecycle_batch(
        integration_db_session, get_settings(), _FakeFileStorage(), _FakeVectorStore(), limit=3
    )
    assert short_page.has_more is True
    assert short_page.next_cursor is not None
    decode_audit_cursor(short_page.next_cursor)  # must decode without error


async def test_documents_outside_the_selected_page_are_not_audited(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    docs = [
        await _seed_document(integration_db_session, created_at=datetime(2026, 1, i + 1, tzinfo=UTC))
        for i in range(4)
    ]

    result = await audit_document_lifecycle_batch(
        integration_db_session, get_settings(), _FakeFileStorage(), _FakeVectorStore(), limit=2
    )

    returned_ids = {summary.document_id for summary in result.documents}
    assert returned_ids == {docs[0].id, docs[1].id}
    assert docs[2].id not in returned_ids
    assert docs[3].id not in returned_ids


async def test_retained_completed_deletion_documents_remain_eligible(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(integration_db_session, created_at=datetime(2026, 1, 1, tzinfo=UTC))
    integration_db_session.add(
        DocumentDeletionJob(
            id=str(uuid.uuid4()),
            document_id=document.id,
            status=DocumentDeletionStatus.COMPLETED,
            vector_cleanup_completed=True,
            storage_cleanup_completed=True,
        )
    )
    await integration_db_session.commit()

    result = await audit_document_lifecycle_batch(
        integration_db_session, get_settings(), _FakeFileStorage(), _FakeVectorStore(), limit=10
    )

    assert result.scanned_count == 1
    assert result.documents[0].document_id == document.id


async def test_one_documents_lifecycle_state_does_not_affect_another_documents_summary(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    document_a = await _seed_document(
        integration_db_session, created_at=datetime(2026, 1, 1, tzinfo=UTC), collection_name=None
    )
    document_b = await _seed_document(
        integration_db_session, created_at=datetime(2026, 1, 2, tzinfo=UTC), collection_name=None
    )
    integration_db_session.add(
        IngestionJob(
            id=str(uuid.uuid4()),
            document_id=document_a.id,
            status=IngestionStatus.COMPLETED,
        )
    )
    await integration_db_session.commit()

    result = await audit_document_lifecycle_batch(
        integration_db_session, get_settings(), _FakeFileStorage(), _FakeVectorStore(), limit=10
    )

    summary_a = next(s for s in result.documents if s.document_id == document_a.id)
    summary_b = next(s for s in result.documents if s.document_id == document_b.id)
    assert summary_a.overall_status.value == "inconsistent"  # completed ingestion, no index metadata
    assert summary_b.overall_status.value == "consistent"  # untouched by document_a's job


async def test_batch_audit_creates_no_rows_and_modifies_no_document_or_job_fields(
    migrated_schema: None, postgres_url: str, integration_db_session: AsyncSession
) -> None:
    integration_db_session.add(
        IndexCollection(
            collection_name="collection-a",
            embedding_provider="ollama",
            embedding_model="test-model",
            embedding_dimension=768,
            embedding_version="v1",
            chunking_version="v1",
            status=IndexCollectionStatus.ACTIVE,
        )
    )
    await integration_db_session.commit()

    document = await _seed_document(
        integration_db_session,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        collection_name="collection-a",
        embedding_provider="ollama",
        embedding_model="test-model",
        embedding_dimension=768,
        embedding_version="v1",
        chunking_version="v1",
    )

    before = await _row_counts(integration_db_session)

    await audit_document_lifecycle_batch(
        integration_db_session, get_settings(), _FakeFileStorage(), _FakeVectorStore(), limit=10
    )

    engine = create_async_engine(postgres_url, future=True)
    try:
        session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
        async with session_factory() as fresh_session:
            after = await _row_counts(fresh_session)
            fresh_document = await fresh_session.get(Document, document.id)
    finally:
        await engine.dispose()

    assert after == before
    assert fresh_document is not None
    assert fresh_document.collection_name == "collection-a"


async def test_invalid_cursor_fails_before_any_lifecycle_mutation(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    await _seed_document(integration_db_session, created_at=datetime(2026, 1, 1, tzinfo=UTC))
    before = await _row_counts(integration_db_session)

    with pytest.raises(InvalidAuditCursorError):
        await audit_document_lifecycle_batch(
            integration_db_session,
            get_settings(),
            _FakeFileStorage(),
            _FakeVectorStore(),
            limit=10,
            cursor="not-a-valid-cursor!!!",
        )

    after = await _row_counts(integration_db_session)
    assert after == before
