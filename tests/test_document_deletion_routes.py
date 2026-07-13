"""HTTP-boundary unit tests for DELETE /api/v1/documents/{id} and GET .../deletion, fake DB only.

Matches tests/test_ingestion_retry_routes.py's dependency-override style. Both routes only ever
touch Postgres (via the fake session) — no FileStorage/vector-store dependency is exercised here.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.db.session import get_db_session
from app.main import app
from app.models.document import Document
from app.models.document_deletion_job import DocumentDeletionJob, DocumentDeletionStatus
from app.models.ingestion_job import IngestionJob, IngestionStatus
from tests.support.fake_document_deletion_session import FakeDocumentDeletionSession

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def _document(document_id: str | None = None) -> Document:
    return Document(
        id=document_id or str(uuid.uuid4()),
        original_filename="report.pdf",
        stored_filename="stored.pdf",
        content_type="application/pdf",
        file_size=11,
        stored_path="documents/x/stored.pdf",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        storage_provider="local",
        storage_key="documents/x/stored.pdf",
    )


def _ingestion_job(document_id: str, status: IngestionStatus) -> IngestionJob:
    now = datetime.now(UTC)
    return IngestionJob(
        id=str(uuid.uuid4()), document_id=document_id, status=status, created_at=now, updated_at=now
    )


def _deletion_job(
    document_id: str, status: DocumentDeletionStatus, **overrides: object
) -> DocumentDeletionJob:
    now = datetime.now(UTC)
    fields: dict[str, object] = {
        "id": str(uuid.uuid4()),
        "document_id": document_id,
        "status": status,
        "vector_cleanup_completed": False,
        "storage_cleanup_completed": False,
        "created_at": now - timedelta(minutes=5),
        "updated_at": now,
    }
    fields.update(overrides)
    return DocumentDeletionJob(**fields)  # type: ignore[arg-type]


def _override_session() -> FakeDocumentDeletionSession:
    session = FakeDocumentDeletionSession()

    async def _fake_db_session():
        yield session

    app.dependency_overrides[get_db_session] = _fake_db_session
    return session


def _delete(document_id: str):
    return client.delete(f"/api/v1/documents/{document_id}")


def _get_deletion(document_id: str):
    return client.get(f"/api/v1/documents/{document_id}/deletion")


def test_delete_missing_document_returns_404() -> None:
    _override_session()
    response = _delete(str(uuid.uuid4()))
    assert response.status_code == 404


def test_delete_new_document_returns_202_and_created_true() -> None:
    session = _override_session()
    document = _document()
    session.documents[document.id] = document

    response = _delete(document.id)

    assert response.status_code == 202
    body = response.json()
    assert body["document_id"] == document.id
    assert body["created"] is True
    assert body["status"] == "pending"


def test_delete_active_ingestion_returns_409() -> None:
    session = _override_session()
    document = _document()
    session.documents[document.id] = document
    session.ingestion_jobs[str(uuid.uuid4())] = _ingestion_job(document.id, IngestionStatus.PENDING)

    response = _delete(document.id)

    assert response.status_code == 409


def test_delete_existing_pending_job_returns_202_created_false() -> None:
    session = _override_session()
    document = _document()
    session.documents[document.id] = document
    existing = _deletion_job(document.id, DocumentDeletionStatus.PENDING)
    session.deletion_jobs[existing.id] = existing

    response = _delete(document.id)

    assert response.status_code == 202
    body = response.json()
    assert body["created"] is False
    assert body["deletion_job_id"] == existing.id


def test_delete_already_deleted_document_returns_200_idempotent() -> None:
    session = _override_session()
    document = _document()
    session.documents[document.id] = document
    completed = _deletion_job(document.id, DocumentDeletionStatus.COMPLETED)
    session.deletion_jobs[completed.id] = completed

    response = _delete(document.id)

    assert response.status_code == 200
    body = response.json()
    assert body["created"] is False
    assert body["status"] == "completed"


def test_delete_two_concurrent_requests_reference_the_same_job() -> None:
    """Both of two 'concurrent' DELETE calls against the same fake session converge on one job."""
    session = _override_session()
    document = _document()
    session.documents[document.id] = document

    first = _delete(document.id)
    second = _delete(document.id)

    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["deletion_job_id"] == second.json()["deletion_job_id"]
    assert first.json()["created"] is True
    assert second.json()["created"] is False


def test_get_deletion_status_returns_404_when_no_attempt_exists() -> None:
    _override_session()
    response = _get_deletion(str(uuid.uuid4()))
    assert response.status_code == 404


def test_get_deletion_status_returns_sanitized_partial_failure() -> None:
    session = _override_session()
    document = _document()
    session.documents[document.id] = document
    job = _deletion_job(
        document.id,
        DocumentDeletionStatus.PARTIALLY_FAILED,
        vector_cleanup_completed=True,
        error_code="document_storage_cleanup_failed",
        error_message="MinIO endpoint http://internal-minio:9000 unreachable: connection refused",
    )
    session.deletion_jobs[job.id] = job

    response = _get_deletion(document.id)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "partially_failed"
    assert body["vector_cleanup_completed"] is True
    assert body["storage_cleanup_completed"] is False
    assert "internal-minio" not in body["safe_message"]
    assert "9000" not in body["safe_message"]


def test_get_deletion_status_never_exposes_storage_or_provider_details() -> None:
    session = _override_session()
    document = _document()
    session.documents[document.id] = document
    job = _deletion_job(
        document.id,
        DocumentDeletionStatus.COMPLETED,
        storage_cleanup_completed=True,
        vector_cleanup_completed=True,
    )
    session.deletion_jobs[job.id] = job

    response = _get_deletion(document.id)

    body = response.json()
    assert "storage_key" not in body
    assert "storage_bucket" not in body
    assert "collection_name" not in body
