"""Postgres integration tests for the document lifecycle audit (Phase 2.8.7, subtask 1) — real
Testcontainers Postgres, fake Object Storage/Qdrant adapters.

Proves the audit's PostgreSQL-side query correctness (latest-job selection, deterministic cleanup
scoping, cross-document isolation, no persisted mutation) against real rows and real foreign keys —
properties a fake session double cannot faithfully represent. Object Storage/Qdrant are faked here
deliberately (see module docstring of test_document_audit_storage_real.py /
test_document_audit_qdrant_real.py for the focused real-dependency coverage) — this module is
PostgreSQL-focused only.
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
from app.models.reindex_job import ReindexJob, ReindexJobStatus
from app.models.vector_cleanup_job import VectorCleanupJob, VectorCleanupStatus
from app.services.reconciliation.document_audit_service import (
    AuditOverallStatus,
    DocumentLifecycleFindingCode,
    audit_document_lifecycle,
)


class _FakeFileStorage:
    def __init__(self, *, existing_keys: set[str] | None = None) -> None:
        self._existing_keys = existing_keys or set()

    async def exists(self, key: str) -> bool:
        return key in self._existing_keys


class _FakeVectorStore:
    def __init__(
        self,
        *,
        collection_sizes: dict[str, int] | None = None,
        vector_counts: dict[tuple[str, str], int] | None = None,
    ) -> None:
        self._collection_sizes = collection_sizes or {}
        self._vector_counts = vector_counts or {}

    async def get_collection_vector_size(self, collection_name: str) -> int | None:
        return self._collection_sizes.get(collection_name)

    async def count_document_vectors(self, collection_name: str, document_id: str) -> int:
        return self._vector_counts.get((collection_name, document_id), 0)


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


async def _ensure_index_collection(session: AsyncSession, collection_name: str) -> None:
    existing = await session.get(IndexCollection, collection_name)
    if existing is None:
        session.add(
            IndexCollection(
                collection_name=collection_name,
                embedding_provider="ollama",
                embedding_model="test-model",
                embedding_dimension=768,
                embedding_version="v1",
                chunking_version="v1",
                status=IndexCollectionStatus.ACTIVE,
            )
        )
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


async def _seed_ingestion_job(
    session: AsyncSession, document_id: str, status: IngestionStatus, **overrides: object
) -> IngestionJob:
    fields: dict[str, object] = dict(id=str(uuid.uuid4()), document_id=document_id, status=status)
    fields.update(overrides)
    job = IngestionJob(**fields)  # type: ignore[arg-type]
    session.add(job)
    await session.commit()
    return job


async def _seed_deletion_job(
    session: AsyncSession, document_id: str, status: DocumentDeletionStatus, **overrides: object
) -> DocumentDeletionJob:
    fields: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        document_id=document_id,
        status=status,
        vector_cleanup_completed=False,
        storage_cleanup_completed=False,
    )
    fields.update(overrides)
    job = DocumentDeletionJob(**fields)  # type: ignore[arg-type]
    session.add(job)
    await session.commit()
    return job


async def _seed_reindex_job(
    session: AsyncSession,
    document_id: str,
    source_collection_name: str,
    target_collection_name: str,
    **overrides: object,
) -> ReindexJob:
    await _ensure_index_collection(session, source_collection_name)
    await _ensure_index_collection(session, target_collection_name)
    fields: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        document_id=document_id,
        source_collection_name=source_collection_name,
        target_collection_name=target_collection_name,
        target_chunk_size=500,
        target_chunk_overlap=50,
        status=ReindexJobStatus.COMPLETED,
    )
    fields.update(overrides)
    job = ReindexJob(**fields)  # type: ignore[arg-type]
    session.add(job)
    await session.commit()
    return job


async def _seed_cleanup_job(
    session: AsyncSession, document_id: str, collection_name: str, **overrides: object
) -> VectorCleanupJob:
    fields: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        document_id=document_id,
        collection_name=collection_name,
        status=VectorCleanupStatus.PENDING,
        attempts=0,
    )
    fields.update(overrides)
    job = VectorCleanupJob(**fields)  # type: ignore[arg-type]
    session.add(job)
    await session.commit()
    return job


async def test_audit_loads_the_correct_latest_ingestion_job(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(integration_db_session)
    older = await _seed_ingestion_job(
        integration_db_session,
        document.id,
        IngestionStatus.FAILED,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    newer = await _seed_ingestion_job(
        integration_db_session,
        document.id,
        IngestionStatus.PROCESSING,
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
        updated_at=datetime.now(UTC),
    )

    result = await audit_document_lifecycle(
        integration_db_session, document.id, get_settings(), _FakeFileStorage(), _FakeVectorStore()
    )

    assert result.postgres_state is not None
    assert result.postgres_state.latest_ingestion_status == newer.status
    assert result.postgres_state.latest_ingestion_status != older.status


async def test_audit_loads_the_correct_latest_deletion_job(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(integration_db_session)
    await _seed_deletion_job(
        integration_db_session,
        document.id,
        DocumentDeletionStatus.PARTIALLY_FAILED,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    newer = await _seed_deletion_job(
        integration_db_session,
        document.id,
        DocumentDeletionStatus.PENDING,
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    result = await audit_document_lifecycle(
        integration_db_session,
        document.id,
        get_settings(),
        _FakeFileStorage(existing_keys={document.storage_key}),
        _FakeVectorStore(),
    )

    assert result.postgres_state is not None
    assert result.postgres_state.latest_deletion_status == newer.status


async def test_audit_loads_the_correct_latest_reindex_job(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(integration_db_session, collection_name=None)
    await _seed_reindex_job(
        integration_db_session,
        document.id,
        "collection-a",
        "collection-b",
        status=ReindexJobStatus.FAILED,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    newer = await _seed_reindex_job(
        integration_db_session,
        document.id,
        "collection-a",
        "collection-c",
        status=ReindexJobStatus.PENDING,
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    result = await audit_document_lifecycle(
        integration_db_session, document.id, get_settings(), _FakeFileStorage(), _FakeVectorStore()
    )

    assert result.postgres_state is not None
    assert result.postgres_state.latest_reindex_status == newer.status


async def test_audit_loads_relevant_cleanup_records_deterministically(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(integration_db_session, collection_name=None)
    await _seed_cleanup_job(integration_db_session, document.id, "old-a", status=VectorCleanupStatus.PENDING)
    await _seed_cleanup_job(integration_db_session, document.id, "old-b", status=VectorCleanupStatus.FAILED)
    await _seed_cleanup_job(
        integration_db_session, document.id, "old-c", status=VectorCleanupStatus.COMPLETED
    )

    result = await audit_document_lifecycle(
        integration_db_session, document.id, get_settings(), _FakeFileStorage(), _FakeVectorStore()
    )

    assert result.postgres_state is not None
    assert set(result.postgres_state.pending_cleanup_collections) == {"old-a", "old-b"}


async def test_historical_terminal_jobs_do_not_override_newer_lifecycle_state(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(integration_db_session, collection_name=None)
    await _seed_ingestion_job(
        integration_db_session,
        document.id,
        IngestionStatus.FAILED,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    await _seed_ingestion_job(
        integration_db_session,
        document.id,
        IngestionStatus.COMPLETED,
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
        updated_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    result = await audit_document_lifecycle(
        integration_db_session,
        document.id,
        get_settings(),
        _FakeFileStorage(existing_keys={document.storage_key}),
        _FakeVectorStore(),
    )

    assert result.postgres_state is not None
    assert result.postgres_state.latest_ingestion_status == IngestionStatus.COMPLETED
    # A COMPLETED-but-no-collection_name document is a genuine inconsistency, not FAILED-derived.
    assert DocumentLifecycleFindingCode.INDEX_METADATA_INCOMPLETE in {f.code for f in result.findings}


async def test_build_completed_but_unactivated_reindex_is_reported_correctly(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(
        integration_db_session,
        collection_name="collection-a",
        embedding_provider="ollama",
        embedding_model="test-model",
        embedding_dimension=768,
        embedding_version="v1",
        chunking_version="v1",
    )
    await _seed_ingestion_job(integration_db_session, document.id, IngestionStatus.COMPLETED)
    await _seed_reindex_job(
        integration_db_session,
        document.id,
        "collection-a",
        "collection-b",
        status=ReindexJobStatus.COMPLETED,
        activated_at=None,
    )

    result = await audit_document_lifecycle(
        integration_db_session,
        document.id,
        get_settings(),
        _FakeFileStorage(existing_keys={document.storage_key}),
        _FakeVectorStore(
            collection_sizes={"collection-a": 768}, vector_counts={("collection-a", document.id): 1}
        ),
    )

    codes = {f.code for f in result.findings}
    assert DocumentLifecycleFindingCode.REINDEX_TARGET_BUILT_NOT_ACTIVATED in codes
    assert result.overall_status == AuditOverallStatus.CONSISTENT


async def test_activated_reindex_with_unresolved_cleanup_is_reported_correctly(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(
        integration_db_session,
        collection_name="collection-b",
        embedding_provider="ollama",
        embedding_model="test-model",
        embedding_dimension=768,
        embedding_version="v1",
        chunking_version="v1",
    )
    await _seed_ingestion_job(integration_db_session, document.id, IngestionStatus.COMPLETED)
    await _seed_reindex_job(
        integration_db_session,
        document.id,
        "collection-a",
        "collection-b",
        status=ReindexJobStatus.COMPLETED,
        activated_at=datetime.now(UTC),
    )
    await _seed_cleanup_job(
        integration_db_session, document.id, "collection-a", status=VectorCleanupStatus.PENDING
    )

    result = await audit_document_lifecycle(
        integration_db_session,
        document.id,
        get_settings(),
        _FakeFileStorage(existing_keys={document.storage_key}),
        _FakeVectorStore(
            collection_sizes={"collection-b": 768}, vector_counts={("collection-b", document.id): 1}
        ),
    )

    codes = {f.code for f in result.findings}
    assert DocumentLifecycleFindingCode.REINDEX_CLEANUP_PENDING in codes
    assert DocumentLifecycleFindingCode.VECTOR_CLEANUP_INCOMPLETE not in codes


async def test_completed_cleanup_removes_the_corresponding_warning(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(
        integration_db_session,
        collection_name="collection-b",
        embedding_provider="ollama",
        embedding_model="test-model",
        embedding_dimension=768,
        embedding_version="v1",
        chunking_version="v1",
    )
    await _seed_ingestion_job(integration_db_session, document.id, IngestionStatus.COMPLETED)
    await _seed_reindex_job(
        integration_db_session,
        document.id,
        "collection-a",
        "collection-b",
        status=ReindexJobStatus.COMPLETED,
        activated_at=datetime.now(UTC),
    )
    await _seed_cleanup_job(
        integration_db_session, document.id, "collection-a", status=VectorCleanupStatus.COMPLETED
    )

    result = await audit_document_lifecycle(
        integration_db_session,
        document.id,
        get_settings(),
        _FakeFileStorage(existing_keys={document.storage_key}),
        _FakeVectorStore(
            collection_sizes={"collection-b": 768}, vector_counts={("collection-b", document.id): 1}
        ),
    )

    codes = {f.code for f in result.findings}
    assert DocumentLifecycleFindingCode.REINDEX_CLEANUP_PENDING not in codes
    assert DocumentLifecycleFindingCode.VECTOR_CLEANUP_INCOMPLETE not in codes


async def test_one_documents_jobs_do_not_affect_another_documents_audit(
    migrated_schema: None, integration_db_session: AsyncSession
) -> None:
    document_a = await _seed_document(integration_db_session, collection_name=None)
    document_b = await _seed_document(integration_db_session, collection_name=None)
    await _seed_ingestion_job(integration_db_session, document_a.id, IngestionStatus.FAILED)
    await _seed_ingestion_job(integration_db_session, document_b.id, IngestionStatus.COMPLETED)
    await _seed_cleanup_job(integration_db_session, document_a.id, "old-a")

    result_b = await audit_document_lifecycle(
        integration_db_session, document_b.id, get_settings(), _FakeFileStorage(), _FakeVectorStore()
    )

    assert result_b.postgres_state is not None
    assert result_b.postgres_state.latest_ingestion_status == IngestionStatus.COMPLETED
    assert result_b.postgres_state.pending_cleanup_collections == ()


async def test_audit_creates_no_database_rows_and_modifies_no_persisted_state(
    migrated_schema: None, postgres_url: str, integration_db_session: AsyncSession
) -> None:
    document = await _seed_document(integration_db_session, collection_name="collection-a")
    await _seed_ingestion_job(integration_db_session, document.id, IngestionStatus.COMPLETED)

    async def _row_counts(session: AsyncSession) -> dict[str, int]:
        counts = {}
        tables = (
            "documents",
            "ingestion_jobs",
            "document_deletion_jobs",
            "reindex_jobs",
            "vector_cleanup_jobs",
        )
        for table in tables:
            result = await session.execute(text(f"SELECT count(*) FROM {table}"))
            counts[table] = result.scalar_one()
        return counts

    before = await _row_counts(integration_db_session)

    await audit_document_lifecycle(
        integration_db_session,
        document.id,
        get_settings(),
        _FakeFileStorage(existing_keys={document.storage_key}),
        _FakeVectorStore(
            collection_sizes={"collection-a": 768}, vector_counts={("collection-a", document.id): 1}
        ),
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
