"""HTTP-boundary unit tests for the document read/download routes, using a fake DB session and
local temp storage, matching tests/test_document_upload.py's dependency-override style."""

import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote

import pytest
from fastapi.testclient import TestClient

from app.api.v1.routes.documents import get_file_storage
from app.db.session import get_db_session
from app.main import app
from app.models.document import Document
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.storage.local_storage import LocalFileStorage
from tests.support.fake_document_session import FakeDocumentQuerySession

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def _document(**overrides: object) -> Document:
    defaults: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        original_filename="report.pdf",
        stored_filename="stored.pdf",
        content_type="application/pdf",
        file_size=11,
        stored_path="documents/x/stored.pdf",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        storage_provider="local",
        storage_bucket=None,
        storage_key="documents/x/stored.pdf",
        storage_etag="etag",
        collection_name=None,
        embedding_provider=None,
        embedding_model=None,
        embedding_dimension=None,
        embedding_version=None,
        chunking_version=None,
        indexed_at=None,
    )
    defaults.update(overrides)
    return Document(**defaults)  # type: ignore[arg-type]


def _job(document_id: str, status: IngestionStatus, **overrides: object) -> IngestionJob:
    defaults: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        document_id=document_id,
        status=status,
        error_message=None,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    defaults.update(overrides)
    return IngestionJob(**defaults)  # type: ignore[arg-type]


def _override_session() -> FakeDocumentQuerySession:
    session = FakeDocumentQuerySession()

    async def _fake_db_session():
        yield session

    app.dependency_overrides[get_db_session] = _fake_db_session
    return session


def _override_storage(tmp_path: Path) -> LocalFileStorage:
    storage = LocalFileStorage(root=tmp_path)
    app.dependency_overrides[get_file_storage] = lambda: storage
    return storage


# --- GET /api/v1/documents ------------------------------------------------------------------


def test_list_documents_empty() -> None:
    _override_session()
    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    body = response.json()
    assert body == {"items": [], "total": 0, "limit": 20, "offset": 0}


def test_list_documents_pagination_query_params() -> None:
    session = _override_session()
    for _ in range(3):
        session.add(_document())

    response = client.get("/api/v1/documents", params={"limit": 2, "offset": 1})
    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 2
    assert body["total"] == 3
    assert body["limit"] == 2
    assert body["offset"] == 1


def test_list_documents_rejects_limit_above_max() -> None:
    _override_session()
    response = client.get("/api/v1/documents", params={"limit": 1000})
    assert response.status_code == 422


# --- GET /api/v1/documents/{id} -------------------------------------------------------------


def test_get_document_detail_404_for_missing_document() -> None:
    _override_session()
    response = client.get(f"/api/v1/documents/{uuid.uuid4()}")
    assert response.status_code == 404


def test_get_document_detail_never_leaks_storage_internals() -> None:
    session = _override_session()
    doc = _document(storage_bucket="documents", storage_key="documents/x/y.pdf", storage_etag="secret-etag")
    session.add(doc)

    response = client.get(f"/api/v1/documents/{doc.id}")

    assert response.status_code == 200
    body = response.json()
    assert body["storage_provider"] == "local"
    for forbidden in ("storage_key", "storage_bucket", "storage_etag"):
        assert forbidden not in body


# --- GET /api/v1/documents/{id}/ingestion ----------------------------------------------------


def test_ingestion_status_404_for_missing_document() -> None:
    _override_session()
    response = client.get(f"/api/v1/documents/{uuid.uuid4()}/ingestion")
    assert response.status_code == 404


def test_ingestion_status_200_with_nulls_when_no_job() -> None:
    session = _override_session()
    doc = _document()
    session.add(doc)

    response = client.get(f"/api/v1/documents/{doc.id}/ingestion")

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == doc.id
    assert body["job_id"] is None
    assert body["status"] is None


def test_ingestion_status_reflects_latest_job() -> None:
    session = _override_session()
    doc = _document()
    session.add(doc)
    session.add(_job(doc.id, IngestionStatus.PROCESSING))

    response = client.get(f"/api/v1/documents/{doc.id}/ingestion")

    assert response.status_code == 200
    assert response.json()["status"] == "processing"


# --- GET /api/v1/documents/{id}/failure ------------------------------------------------------


def test_failure_404_for_missing_document() -> None:
    _override_session()
    response = client.get(f"/api/v1/documents/{uuid.uuid4()}/failure")
    assert response.status_code == 404


def test_failure_404_when_no_failed_job() -> None:
    session = _override_session()
    doc = _document()
    session.add(doc)
    session.add(_job(doc.id, IngestionStatus.COMPLETED))

    response = client.get(f"/api/v1/documents/{doc.id}/failure")
    assert response.status_code == 404


def test_failure_returns_sanitized_message_not_raw_error() -> None:
    session = _override_session()
    doc = _document()
    session.add(doc)
    session.add(
        _job(
            doc.id,
            IngestionStatus.FAILED,
            error_message="Qdrant unreachable at http://internal-host:6333/collections",
        )
    )

    response = client.get(f"/api/v1/documents/{doc.id}/failure")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert "internal-host" not in body["safe_message"]
    assert "6333" not in body["safe_message"]
    assert "retryable" not in body


# --- GET /api/v1/documents/{id}/download -----------------------------------------------------


def test_download_404_for_missing_document(tmp_path: Path) -> None:
    _override_session()
    _override_storage(tmp_path)

    response = client.get(f"/api/v1/documents/{uuid.uuid4()}/download")
    assert response.status_code == 404


def test_download_success_local_storage(tmp_path: Path) -> None:
    session = _override_session()
    _override_storage(tmp_path)

    doc = _document(storage_key="documents/x/report.pdf", content_type="application/pdf")
    session.add(doc)
    (tmp_path / "documents" / "x").mkdir(parents=True)
    (tmp_path / "documents" / "x" / "report.pdf").write_bytes(b"%PDF-1.4 exact bytes")

    response = client.get(f"/api/v1/documents/{doc.id}/download")

    assert response.status_code == 200
    assert response.content == b"%PDF-1.4 exact bytes"
    assert response.headers["content-type"] == "application/pdf"
    assert 'filename="report.pdf"' in response.headers["content-disposition"]


def test_download_unicode_filename_content_disposition(tmp_path: Path) -> None:
    session = _override_session()
    _override_storage(tmp_path)

    hebrew_name = "חוזה_עבודה.pdf"
    doc = _document(original_filename=hebrew_name, storage_key="documents/x/stored.pdf")
    session.add(doc)
    (tmp_path / "documents" / "x").mkdir(parents=True)
    (tmp_path / "documents" / "x" / "stored.pdf").write_bytes(b"content")

    response = client.get(f"/api/v1/documents/{doc.id}/download")

    assert response.status_code == 200
    disposition = response.headers["content-disposition"]
    # ASCII fallback form must be present and header-safe (no raw non-ASCII bytes).
    assert disposition.isascii()
    assert "filename=" in disposition
    # UTF-8 percent-encoded form must decode back to the exact original filename.
    assert "filename*=UTF-8''" in disposition
    encoded = disposition.split("filename*=UTF-8''", 1)[1]
    assert unquote(encoded) == hebrew_name


def test_download_missing_object_is_409(tmp_path: Path) -> None:
    session = _override_session()
    _override_storage(tmp_path)

    doc = _document(storage_key="documents/x/missing.pdf")
    session.add(doc)
    # Note: no file actually written under storage_key.

    response = client.get(f"/api/v1/documents/{doc.id}/download")
    assert response.status_code == 409


def test_download_does_not_mutate_anything(tmp_path: Path) -> None:
    session = _override_session()
    _override_storage(tmp_path)

    doc = _document(storage_key="documents/x/report.pdf")
    session.add(doc)
    (tmp_path / "documents" / "x").mkdir(parents=True)
    (tmp_path / "documents" / "x" / "report.pdf").write_bytes(b"content")

    client.get(f"/api/v1/documents/{doc.id}/download")

    # No new Document/IngestionJob rows were added, no commit was triggered as a write.
    assert len(session.documents) == 1
    assert len(session.jobs) == 0
