"""Backend E2E: the same document read/download lifecycle as test_document_read_api.py, for
FILE_STORAGE_PROVIDER=minio — mirrors test_minio_e2e.py's minio_storage_settings/minio_app_client/
process_pending_job_minio pattern so storage provider selection goes through the app's real
get_file_storage()/create_file_storage() dependency chain, never a hand-substituted instance.
"""

import uuid

import httpx
import pytest

from app.core.config import get_settings
from app.db.session import get_db_session
from app.main import app as fastapi_app
from app.models.ingestion_job import IngestionStatus
from app.services.document_query_service import get_document
from app.storage.factory import create_file_storage
from app.storage.keys import resolve_document_storage_key
from app.storage.minio_storage import MinioFileStorage

pytestmark = pytest.mark.e2e

_CONTENT = b"MinIO-backed document read API E2E content.\n"
_HEBREW_FILENAME = "קובץ-בדיקה.txt"


@pytest.fixture
def minio_bucket_name() -> str:
    """A unique bucket name per test invocation."""
    return f"e2e-minio-read-{uuid.uuid4().hex}"


@pytest.fixture
def minio_storage_settings(
    minio_endpoint: str,
    minio_credentials: tuple[str, str],
    minio_bucket_name: str,
    monkeypatch: pytest.MonkeyPatch,
    isolated_test_state: None,
) -> None:
    """Point the app's cached Settings singleton at FILE_STORAGE_PROVIDER=minio for this test."""
    access_key, secret_key = minio_credentials
    settings = get_settings()
    monkeypatch.setattr(settings, "file_storage_provider", "minio")
    monkeypatch.setattr(settings, "minio_endpoint", minio_endpoint)
    monkeypatch.setattr(settings, "minio_access_key", access_key)
    monkeypatch.setattr(settings, "minio_secret_key", secret_key)
    monkeypatch.setattr(settings, "minio_bucket", minio_bucket_name)
    monkeypatch.setattr(settings, "minio_secure", False)
    monkeypatch.setattr(settings, "minio_create_bucket_if_missing", True)


async def _ensure_minio_bucket() -> None:
    """Create the configured MinIO bucket via the real create_file_storage() dependency chain.

    Uses the same production code path app/services/platform_health.py's readiness probe uses
    (MinioFileStorage.ensure_bucket()) — but calls it directly rather than through GET
    /health/ready, since readiness also gates on unrelated required dependencies (e.g. the
    configured Ollama embedding model actually being pulled) that this read-API test suite has
    no need to depend on.
    """
    storage = create_file_storage()
    assert isinstance(storage, MinioFileStorage)
    await storage.ensure_bucket()


async def test_full_read_lifecycle_over_http_via_minio(
    minio_storage_settings: None,
    minio_app_client: httpx.AsyncClient,
    process_pending_job_minio,
) -> None:
    """Upload -> ingest -> list/detail/ingestion -> download, all through real MinIO."""
    await _ensure_minio_bucket()

    upload = await minio_app_client.post(
        "/api/v1/documents", files={"file": (_HEBREW_FILENAME, _CONTENT, "text/plain")}
    )
    assert upload.status_code == 202
    document_id = upload.json()["document_id"]

    detail = await minio_app_client.get(f"/api/v1/documents/{document_id}")
    assert detail.status_code == 200
    assert detail.json()["storage_provider"] == "minio"

    result = await process_pending_job_minio()
    assert result is not None
    assert result.status == IngestionStatus.COMPLETED

    listing = await minio_app_client.get("/api/v1/documents")
    assert document_id in [item["id"] for item in listing.json()["items"]]

    download = await minio_app_client.get(f"/api/v1/documents/{document_id}/download")
    assert download.status_code == 200
    assert download.content == _CONTENT


async def test_download_missing_object_is_409_via_minio(
    minio_storage_settings: None,
    minio_app_client: httpx.AsyncClient,
) -> None:
    """A document row that exists but has no corresponding MinIO object is a 409, not a 404."""
    await _ensure_minio_bucket()

    upload = await minio_app_client.post(
        "/api/v1/documents", files={"file": ("gone.txt", b"will be deleted", "text/plain")}
    )
    assert upload.status_code == 202
    document_id = upload.json()["document_id"]

    # Simulate real-world storage/DB drift: delete the MinIO object out from under the row.
    session_dep = fastapi_app.dependency_overrides[get_db_session]
    async for session in session_dep():
        document = await get_document(session, document_id)
        assert document is not None
        key = resolve_document_storage_key(document)
        break

    storage = create_file_storage()
    await storage.delete(key)

    response = await minio_app_client.get(f"/api/v1/documents/{document_id}/download")
    assert response.status_code == 409
