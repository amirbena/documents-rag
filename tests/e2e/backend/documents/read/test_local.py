"""Backend E2E: upload -> ingestion -> list -> detail -> ingestion-status -> download, over the
real public HTTP boundary, for FILE_STORAGE_PROVIDER=local (see test_minio.py
for the MinIO-flavored equivalent). These are read-only document APIs — they touch only
Document/IngestionJob/FileStorage, never RagEngine, so there is no need to run this under both
RAG_ENGINE settings (unlike tests/e2e/backend/test_rag_engine_parity.py).
"""

from urllib.parse import unquote

import httpx
import pytest

from app.models.ingestion_job import IngestionStatus

pytestmark = pytest.mark.e2e

_CONTENT = b"Plain text content for the document read API E2E test.\n"
_HEBREW_FILENAME = "מסמך-בדיקה.txt"


async def test_full_read_lifecycle_over_http(
    app_client: httpx.AsyncClient, process_pending_job
) -> None:
    """Upload, then walk every read endpoint through the pending -> indexed lifecycle."""
    upload = await app_client.post(
        "/api/v1/documents", files={"file": (_HEBREW_FILENAME, _CONTENT, "text/plain")}
    )
    assert upload.status_code == 202
    document_id = upload.json()["document_id"]

    # Immediately after upload: pending, listed, detail visible.
    ingestion = await app_client.get(f"/api/v1/documents/{document_id}/ingestion")
    assert ingestion.status_code == 200
    assert ingestion.json()["status"] == "pending"

    listing = await app_client.get("/api/v1/documents")
    assert listing.status_code == 200
    ids = [item["id"] for item in listing.json()["items"]]
    assert document_id in ids

    detail = await app_client.get(f"/api/v1/documents/{document_id}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "pending"
    for forbidden in ("storage_key", "storage_bucket", "storage_etag"):
        assert forbidden not in detail.json()

    # Process the job -> indexed.
    result = await process_pending_job()
    assert result is not None
    assert result.status == IngestionStatus.COMPLETED

    detail_after = await app_client.get(f"/api/v1/documents/{document_id}")
    assert detail_after.status_code == 200
    assert detail_after.json()["status"] == "indexed"
    assert detail_after.json()["indexed_at"] is not None

    ingestion_after = await app_client.get(f"/api/v1/documents/{document_id}/ingestion")
    assert ingestion_after.status_code == 200
    assert ingestion_after.json()["status"] == "completed"

    # Download: exact bytes, correct content-type, Hebrew filename round-trips.
    download = await app_client.get(f"/api/v1/documents/{document_id}/download")
    assert download.status_code == 200
    assert download.content == _CONTENT
    # Starlette appends a default charset to text/* media types; the original type is preserved.
    assert download.headers["content-type"].startswith("text/plain")
    disposition = download.headers["content-disposition"]
    encoded = disposition.split("filename*=UTF-8''", 1)[1]
    assert unquote(encoded) == _HEBREW_FILENAME

    # No failure to report for a successfully-indexed document.
    failure = await app_client.get(f"/api/v1/documents/{document_id}/failure")
    assert failure.status_code == 404


async def test_failed_ingestion_is_visible_through_the_failure_endpoint(
    app_client: httpx.AsyncClient, process_pending_job
) -> None:
    """A file that fails extraction surfaces through GET .../failure with a sanitized message."""
    upload = await app_client.post(
        "/api/v1/documents",
        files={"file": ("not-really-a.pdf", b"this is not a valid pdf file", "application/pdf")},
    )
    assert upload.status_code == 202
    document_id = upload.json()["document_id"]

    result = await process_pending_job()
    assert result is not None
    assert result.status == IngestionStatus.FAILED

    failure = await app_client.get(f"/api/v1/documents/{document_id}/failure")
    assert failure.status_code == 200
    body = failure.json()
    assert body["status"] == "failed"
    assert body["job_id"] == result.id
    assert "valid PDF" not in body["safe_message"], "raw error text must never leak into safe_message"
    assert "retryable" not in body

    detail = await app_client.get(f"/api/v1/documents/{document_id}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "failed"


async def test_download_missing_document_is_404(app_client: httpx.AsyncClient) -> None:
    response = await app_client.get("/api/v1/documents/does-not-exist/download")
    assert response.status_code == 404


async def test_ingestion_status_missing_document_is_404(app_client: httpx.AsyncClient) -> None:
    response = await app_client.get("/api/v1/documents/does-not-exist/ingestion")
    assert response.status_code == 404
