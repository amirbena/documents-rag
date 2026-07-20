"""Focused real-Object-Storage tests for the document lifecycle audit (Phase 2.8.7, subtask 1) —
real Testcontainers Postgres + real LocalFileStorage, no fake storage adapter.

Proves the audit's Object Storage inspection is correct against a real `FileStorage`
implementation — `LocalFileStorage` is the one existing storage provider integration stack used
here (mirrors `tests/integration/documents/deletion/test_storage.py`'s Local coverage; MinIO
coverage is not duplicated since `FileStorage.exists()`'s contract is already proven identical
across providers there). Qdrant is faked in this module — see
`test_document_audit_qdrant_real.py` for the real-Qdrant counterpart.
"""

import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.models.document import Document
from app.services.reconciliation.document_audit_service import (
    DocumentLifecycleFindingCode,
    audit_document_lifecycle,
)
from app.storage.local_storage import LocalFileStorage


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
            await session.execute(text("TRUNCATE TABLE ingestion_jobs, documents RESTART IDENTITY CASCADE"))
            await session.commit()

    await _truncate()
    try:
        yield
    finally:
        await _truncate()
        await engine.dispose()


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


async def test_persisted_object_exists_produces_no_object_missing_finding(
    migrated_schema: None, integration_db_session: AsyncSession, tmp_path: Path
) -> None:
    storage = LocalFileStorage(root=tmp_path)
    key = "documents/a/notes.txt"
    await storage.save(key, b"hello world")

    document = _document(key)
    integration_db_session.add(document)
    await integration_db_session.commit()

    result = await audit_document_lifecycle(
        integration_db_session, document.id, get_settings(), storage, _FakeVectorStore()
    )

    assert result.storage_state is not None
    assert result.storage_state.inspected is True
    assert result.storage_state.exists is True
    assert DocumentLifecycleFindingCode.OBJECT_MISSING not in {f.code for f in result.findings}


async def test_persisted_object_absent_produces_object_missing(
    migrated_schema: None, integration_db_session: AsyncSession, tmp_path: Path
) -> None:
    storage = LocalFileStorage(root=tmp_path)
    key = "documents/a/never-saved.txt"

    document = _document(key)
    integration_db_session.add(document)
    await integration_db_session.commit()

    result = await audit_document_lifecycle(
        integration_db_session, document.id, get_settings(), storage, _FakeVectorStore()
    )

    assert result.storage_state is not None
    assert result.storage_state.inspected is True
    assert result.storage_state.exists is False
    assert DocumentLifecycleFindingCode.OBJECT_MISSING in {f.code for f in result.findings}
