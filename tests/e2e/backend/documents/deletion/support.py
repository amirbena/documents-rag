"""Shared Backend E2E helpers for document-deletion scenarios, used by two or more test modules.

Infrastructure-global fixtures (Postgres/Qdrant/MinIO container startup, the fake AI providers)
stay in tests/e2e/backend/conftest.py/fakes.py — nothing here duplicates that. This module only
holds the deletion-feature-specific "drive a document through ingestion" helper and the
"execute one pending deletion job for real" fixture that test_successful_deletion.py,
test_partial_failures.py, and test_deleted_document_behavior.py all need.
"""

from pathlib import Path

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.models.ingestion_job import IngestionStatus
from app.rag.providers.qdrant_vector_store import QdrantVectorStore
from app.services.documents.deletion_worker import DocumentDeletionWorker
from app.storage.local_storage import LocalFileStorage

VALID_CONTENT = b"Plain text content for the document deletion E2E test.\n"


async def upload_and_ingest(app_client: httpx.AsyncClient, process_pending_job) -> str:
    """Upload one document and drive it to COMPLETED through the real IngestionWorker."""
    upload = await app_client.post(
        "/api/v1/documents", files={"file": ("notes.txt", VALID_CONTENT, "text/plain")}
    )
    assert upload.status_code == 202
    document_id = upload.json()["document_id"]

    result = await process_pending_job()
    assert result is not None
    assert result.status == IngestionStatus.COMPLETED
    return document_id


@pytest.fixture
def process_pending_deletion(
    e2e_session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
    isolated_test_state: None,
):
    """Run a real DocumentDeletionWorker against one pending deletion job (real Qdrant + storage)."""
    settings = get_settings()
    file_storage = LocalFileStorage(root=tmp_path)
    vector_store = QdrantVectorStore(settings=settings)

    async def _process():
        async with e2e_session_factory() as session:
            worker = DocumentDeletionWorker(vector_store=vector_store, file_storage=file_storage)
            return await worker.process_next_job(session)

    return _process
