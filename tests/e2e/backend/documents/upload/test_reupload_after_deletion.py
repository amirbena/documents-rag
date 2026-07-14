"""Backend E2E: re-uploading identical content after its prior document was fully deleted.

Same real-HTTP, real-Postgres/Qdrant, fake-AI-provider setup as
documents/deletion/test_successful_deletion.py (see tests/e2e/backend/conftest.py). This is the
load-bearing acceptance test for Phase 2.8.5's hash-release-on-COMPLETED-deletion behavior: proves
a `content_hash` genuinely becomes available for a brand-new document once its old owner's
deletion reaches COMPLETED, rather than staying permanently reserved.
"""

import httpx
import pytest

from app.core.config import get_settings
from app.models.ingestion_job import IngestionStatus
from app.rag.embedding_config import get_active_embedding_config
from app.rag.providers.qdrant_vector_store import QdrantVectorStore
from tests.e2e.backend.documents.deletion.support import process_pending_deletion
from tests.e2e.backend.fakes import FakeEmbeddingProvider

pytestmark = pytest.mark.e2e

__all__ = ["process_pending_deletion"]  # re-exported fixture, used via pytest fixture injection

_RELEASED_CONTENT = b"This document proves hash release marker after completed deletion.\n"


async def test_completed_deletion_releases_the_hash_for_a_new_upload(
    app_client: httpx.AsyncClient,
    process_pending_job,
    process_pending_deletion,
    fake_embedding_provider: FakeEmbeddingProvider,
) -> None:
    """Identical bytes may be uploaded again — as a genuinely new document — once deletion completes."""
    first = await app_client.post(
        "/api/v1/documents",
        files={"file": ("marker.txt", _RELEASED_CONTENT, "text/plain")},
    )
    assert first.status_code == 202
    old_document_id = first.json()["document_id"]

    first_ingestion = await process_pending_job()
    assert first_ingestion is not None
    assert first_ingestion.status == IngestionStatus.COMPLETED

    delete_response = await app_client.delete(f"/api/v1/documents/{old_document_id}")
    assert delete_response.status_code == 202

    deletion_result = await process_pending_deletion()
    assert deletion_result is not None
    assert deletion_result.status.value == "completed"

    old_detail = await app_client.get(f"/api/v1/documents/{old_document_id}")
    assert old_detail.status_code == 200
    assert old_detail.json()["status"] == "deleted"

    old_download = await app_client.get(f"/api/v1/documents/{old_document_id}/download")
    assert old_download.status_code == 410

    reupload = await app_client.post(
        "/api/v1/documents",
        files={"file": ("marker-again.txt", _RELEASED_CONTENT, "text/plain")},
    )

    assert reupload.status_code == 202
    body = reupload.json()
    assert body["outcome"] == "CREATED"
    new_document_id = body["document_id"]
    assert new_document_id != old_document_id

    second_ingestion = await process_pending_job()
    assert second_ingestion is not None
    assert second_ingestion.status == IngestionStatus.COMPLETED

    settings = get_settings()
    active_config = get_active_embedding_config(settings)
    vector_store = QdrantVectorStore(settings=settings)
    query_vector = (await fake_embedding_provider.embed(["marker"]))[0]
    search_results = await vector_store.search_similar(active_config.collection_name, query_vector, limit=50)

    assert all(result.document_id != old_document_id for result in search_results), (
        "the deleted document's vectors must remain absent"
    )
    assert any(result.document_id == new_document_id for result in search_results), (
        "the new document must own the rebuilt vectors"
    )
