"""HTTP-boundary unit tests for POST /api/v1/documents/{id}/ingestion/retry, fake DB session only.

Matches tests/test_document_read_routes.py's dependency-override style. Also asserts retry never
touches storage/embedding/vector-store code paths — it only ever reads/writes Postgres.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.db.session import get_db_session
from app.main import app
from app.models.document import Document
from app.models.document_deletion_job import DocumentDeletionJob, DocumentDeletionStatus
from app.models.ingestion_job import IngestionJob, IngestionStatus
from tests.support.fake_ingestion_retry_session import FakeIngestionRetrySession

client = TestClient(app)
STALE_AFTER = get_settings().ingestion_stale_after_seconds


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


def _job(document_id: str, status: IngestionStatus, *, updated_at: datetime | None = None) -> IngestionJob:
    now = datetime.now(UTC)
    return IngestionJob(
        id=str(uuid.uuid4()),
        document_id=document_id,
        status=status,
        created_at=now - timedelta(minutes=5),
        updated_at=updated_at or now,
    )


def _override_session() -> FakeIngestionRetrySession:
    session = FakeIngestionRetrySession()

    async def _fake_db_session():
        yield session

    app.dependency_overrides[get_db_session] = _fake_db_session
    return session


def _retry(document_id: str):
    return client.post(f"/api/v1/documents/{document_id}/ingestion/retry")


def test_retry_missing_document_returns_404() -> None:
    _override_session()
    response = _retry(str(uuid.uuid4()))
    assert response.status_code == 404


def test_retry_failed_job_returns_202_and_new_job() -> None:
    session = _override_session()
    document = _document()
    session.documents[document.id] = document
    failed_job = _job(document.id, IngestionStatus.FAILED)
    session.jobs[failed_job.id] = failed_job

    response = _retry(document.id)

    assert response.status_code == 202
    body = response.json()
    assert body["document_id"] == document.id
    assert body["created"] is True
    assert body["status"] == "pending"
    assert body["job_id"] != failed_job.id


def test_retry_pending_job_returns_200_no_new_job() -> None:
    session = _override_session()
    document = _document()
    session.documents[document.id] = document
    pending_job = _job(document.id, IngestionStatus.PENDING)
    session.jobs[pending_job.id] = pending_job

    response = _retry(document.id)

    assert response.status_code == 200
    body = response.json()
    assert body["created"] is False
    assert body["job_id"] == pending_job.id


def test_retry_fresh_processing_job_returns_200_no_new_job() -> None:
    session = _override_session()
    document = _document()
    session.documents[document.id] = document
    processing_job = _job(document.id, IngestionStatus.PROCESSING, updated_at=datetime.now(UTC))
    session.jobs[processing_job.id] = processing_job

    response = _retry(document.id)

    assert response.status_code == 200
    assert response.json()["created"] is False


def test_retry_stale_processing_job_returns_202_and_new_job() -> None:
    session = _override_session()
    document = _document()
    session.documents[document.id] = document
    stale_job = _job(
        document.id,
        IngestionStatus.PROCESSING,
        updated_at=datetime.now(UTC) - timedelta(seconds=STALE_AFTER + 1),
    )
    session.jobs[stale_job.id] = stale_job

    response = _retry(document.id)

    assert response.status_code == 202
    assert response.json()["created"] is True


def test_retry_completed_job_returns_409() -> None:
    session = _override_session()
    document = _document()
    session.documents[document.id] = document
    completed_job = _job(document.id, IngestionStatus.COMPLETED)
    session.jobs[completed_job.id] = completed_job

    response = _retry(document.id)

    assert response.status_code == 409


def test_retry_returns_409_when_document_is_being_deleted() -> None:
    session = _override_session()
    document = _document()
    session.documents[document.id] = document
    failed_job = _job(document.id, IngestionStatus.FAILED)
    session.jobs[failed_job.id] = failed_job
    session.deletion_jobs["d1"] = DocumentDeletionJob(
        id="d1",
        document_id=document.id,
        status=DocumentDeletionStatus.PENDING,
        vector_cleanup_completed=False,
        storage_cleanup_completed=False,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    response = _retry(document.id)

    assert response.status_code == 409


def test_retry_never_touches_storage_or_vector_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry must only touch Postgres — no FileStorage/embedding/vector-store calls."""

    def _fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("retry must never call this during a Postgres-only operation")

    monkeypatch.setattr("app.rag.providers.provider_factory.get_embedding_provider", _fail_if_called)
    monkeypatch.setattr("app.rag.providers.provider_factory.get_vector_store", _fail_if_called)

    session = _override_session()
    document = _document()
    session.documents[document.id] = document
    failed_job = _job(document.id, IngestionStatus.FAILED)
    session.jobs[failed_job.id] = failed_job

    response = _retry(document.id)

    assert response.status_code == 202
