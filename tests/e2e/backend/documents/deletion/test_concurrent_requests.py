"""Backend E2E: concurrent DELETE requests for the same document, through the real HTTP boundary.

Same real-HTTP, real-Postgres/Qdrant, fake-AI-provider setup as
test_upload_to_streaming_chat.py/test_ingestion_retry_recovery.py (see tests/e2e/backend/
conftest.py). Two genuinely concurrent HTTP requests (via `asyncio.gather`), not two calls
sharing one session — proves the same one-active-job invariant the Postgres integration tier
proves at the service layer, but through the real public API this time.
"""

import asyncio

import httpx
import pytest

from tests.e2e.backend.documents.deletion.support import upload_and_ingest

pytestmark = pytest.mark.e2e


async def test_concurrent_delete_requests_reference_the_same_deletion_job(
    app_client: httpx.AsyncClient, process_pending_job
) -> None:
    """Two DELETE calls for the same document must reference exactly one active deletion job."""
    document_id = await upload_and_ingest(app_client, process_pending_job)

    first, second = await asyncio.gather(
        app_client.delete(f"/api/v1/documents/{document_id}"),
        app_client.delete(f"/api/v1/documents/{document_id}"),
    )

    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["deletion_job_id"] == second.json()["deletion_job_id"]
    # Exactly one of the two responses reports having created the job.
    assert sorted([first.json()["created"], second.json()["created"]]) == [False, True]
