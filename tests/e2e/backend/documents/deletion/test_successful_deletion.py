"""Backend E2E: the successful full-document-deletion workflow, through the real HTTP boundary.

Same real-HTTP, real-Postgres/Qdrant, fake-AI-provider setup as
test_upload_to_streaming_chat.py/test_ingestion_retry_recovery.py (see tests/e2e/backend/
conftest.py). Executes the actual cross-system cleanup via a real DocumentDeletionWorker (real
Qdrant, real tmp_path-rooted LocalFileStorage) — never a mock.
"""

import httpx
import pytest

from app.core.config import get_settings
from app.rag.embedding_config import get_active_embedding_config
from app.rag.providers.qdrant_vector_store import QdrantVectorStore
from tests.e2e.backend.documents.deletion.support import process_pending_deletion, upload_and_ingest
from tests.e2e.backend.fakes import FakeEmbeddingProvider

pytestmark = pytest.mark.e2e

__all__ = ["process_pending_deletion"]  # re-exported fixture, used via pytest fixture injection


async def test_successful_deletion_removes_content_and_makes_it_unsearchable(
    app_client: httpx.AsyncClient,
    process_pending_job,
    process_pending_deletion,
    fake_embedding_provider: FakeEmbeddingProvider,
) -> None:
    """Upload -> ingest -> searchable -> DELETE -> worker runs -> deleted, 410, not searchable."""
    document_id = await upload_and_ingest(app_client, process_pending_job)

    settings = get_settings()
    active_config = get_active_embedding_config(settings)
    vector_store = QdrantVectorStore(settings=settings)
    query_vector = (await fake_embedding_provider.embed(["notes"]))[0]

    before = await vector_store.search_similar(active_config.collection_name, query_vector, limit=10)
    assert any(result.document_id == document_id for result in before)

    delete_response = await app_client.delete(f"/api/v1/documents/{document_id}")
    assert delete_response.status_code == 202
    assert delete_response.json()["created"] is True

    deletion_result = await process_pending_deletion()
    assert deletion_result is not None
    assert deletion_result.status.value == "completed"

    detail = await app_client.get(f"/api/v1/documents/{document_id}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "deleted"

    download = await app_client.get(f"/api/v1/documents/{document_id}/download")
    assert download.status_code == 410

    after = await vector_store.search_similar(active_config.collection_name, query_vector, limit=10)
    assert all(result.document_id != document_id for result in after)

    status_response = await app_client.get(f"/api/v1/documents/{document_id}/deletion")
    assert status_response.status_code == 200
    body = status_response.json()
    assert body["status"] == "completed"
    assert body["vector_cleanup_completed"] is True
    assert body["storage_cleanup_completed"] is True
