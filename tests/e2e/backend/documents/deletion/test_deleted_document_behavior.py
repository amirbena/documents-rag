"""Backend E2E: cross-endpoint behavior for an already-deleted document, real HTTP boundary.

Same real-HTTP, real-Postgres/Qdrant, fake-AI-provider setup as
test_upload_to_streaming_chat.py/test_ingestion_retry_recovery.py (see tests/e2e/backend/
conftest.py). Proves a deleted document is never implicitly resurrected by another mutating
endpoint.
"""

import httpx
import pytest

from tests.e2e.backend.documents.deletion.support import process_pending_deletion, upload_and_ingest

pytestmark = pytest.mark.e2e

__all__ = ["process_pending_deletion"]  # re-exported fixture, used via pytest fixture injection


async def test_deleted_document_rejects_ingestion_retry(
    app_client: httpx.AsyncClient, process_pending_job, process_pending_deletion
) -> None:
    """A fully deleted document must reject POST .../ingestion/retry with 409, never resurrect it."""
    document_id = await upload_and_ingest(app_client, process_pending_job)

    delete_response = await app_client.delete(f"/api/v1/documents/{document_id}")
    assert delete_response.status_code == 202

    deletion_result = await process_pending_deletion()
    assert deletion_result is not None
    assert deletion_result.status.value == "completed"

    retry_response = await app_client.post(f"/api/v1/documents/{document_id}/ingestion/retry")
    assert retry_response.status_code == 409
