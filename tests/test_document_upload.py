"""Tests for POST /api/v1/documents with a fake DB session and local temp storage."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.v1.routes.documents import get_local_file_storage
from app.db.session import get_db_session
from app.main import app
from app.models.document import Document
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.services.local_file_storage import LocalFileStorage

client = TestClient(app)


class _FakeAsyncSession:
    """Minimal AsyncSession stand-in: records added rows, no real DB behind it."""

    def __init__(self) -> None:
        self.added: list[object] = []
        self.committed = False

    def add(self, instance: object) -> None:
        self.added.append(instance)

    async def commit(self) -> None:
        self.committed = True


@pytest.fixture(autouse=True)
def _clear_overrides():
    """Reset FastAPI dependency overrides after each test."""
    yield
    app.dependency_overrides.clear()


def _override_dependencies(tmp_path: Path) -> _FakeAsyncSession:
    """Wire a fake DB session and a temp-dir-backed LocalFileStorage into the app."""
    fake_session = _FakeAsyncSession()

    async def _fake_db_session():
        yield fake_session

    app.dependency_overrides[get_db_session] = _fake_db_session
    app.dependency_overrides[get_local_file_storage] = lambda: LocalFileStorage(root=tmp_path)
    return fake_session


def test_upload_creates_document_and_ingestion_job(tmp_path: Path) -> None:
    """A successful upload should create one Document row and one pending IngestionJob row."""
    fake_session = _override_dependencies(tmp_path)

    response = client.post(
        "/api/v1/documents",
        files={"file": ("report.pdf", b"%PDF-1.4 fake content", "application/pdf")},
    )

    assert response.status_code == 202
    assert len(fake_session.added) == 2

    document, job = fake_session.added
    assert isinstance(document, Document)
    assert isinstance(job, IngestionJob)
    assert job.document_id == document.id
    assert job.status == IngestionStatus.PENDING
    assert fake_session.committed is True


def test_upload_response_status_is_202(tmp_path: Path) -> None:
    """The response status code should be 202 Accepted, not 200 or 201."""
    _override_dependencies(tmp_path)

    response = client.post(
        "/api/v1/documents",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )

    assert response.status_code == 202
    body = response.json()
    assert "document_id" in body
    assert "job_id" in body
    assert body["status"] == "pending"


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    """An empty (zero-byte) upload should be rejected with 400, not create any rows."""
    fake_session = _override_dependencies(tmp_path)

    response = client.post(
        "/api/v1/documents",
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 400
    assert fake_session.added == []
    assert fake_session.committed is False


def test_hebrew_filename_upload_works(tmp_path: Path) -> None:
    """A Hebrew filename should upload successfully and be preserved exactly."""
    fake_session = _override_dependencies(tmp_path)

    response = client.post(
        "/api/v1/documents",
        files={"file": ("חוזה_עבודה.pdf", b"%PDF-1.4 fake content", "application/pdf")},
    )

    assert response.status_code == 202
    document, _job = fake_session.added
    assert document.original_filename == "חוזה_עבודה.pdf"


def test_original_filename_is_preserved_exactly(tmp_path: Path) -> None:
    """The original filename should be stored in the DB exactly as received, unmodified."""
    fake_session = _override_dependencies(tmp_path)
    original_name = "Q3 Report (Final) — v2.pdf"

    response = client.post(
        "/api/v1/documents",
        files={"file": (original_name, b"content", "application/pdf")},
    )

    assert response.status_code == 202
    document, _job = fake_session.added
    assert document.original_filename == original_name


def test_stored_filename_is_generated_and_safe(tmp_path: Path) -> None:
    """The stored filename must not be the raw original filename and must be filesystem-safe."""
    fake_session = _override_dependencies(tmp_path)
    original_name = "חוזה עבודה?!*.pdf"

    response = client.post(
        "/api/v1/documents",
        files={"file": (original_name, b"content", "application/pdf")},
    )

    assert response.status_code == 202
    document, _job = fake_session.added
    assert document.stored_filename != original_name
    assert document.original_filename == original_name
    # Safe: ASCII-only, no path separators, no spaces or special characters.
    assert all(c.isascii() and (c.isalnum() or c == ".") for c in document.stored_filename)
    # The uploaded content was actually written under the generated safe name.
    stored_file = tmp_path / document.stored_filename
    assert stored_file.exists()
    assert stored_file.read_bytes() == b"content"


def test_upload_does_not_call_embedding_or_vector_store_or_llm(tmp_path: Path, monkeypatch) -> None:
    """Uploading must not trigger any embedding, vector-store, or LLM provider call."""

    def _fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("upload must not call provider factory functions")

    monkeypatch.setattr("app.rag.providers.provider_factory.get_embedding_provider", _fail)
    monkeypatch.setattr("app.rag.providers.provider_factory.get_llm_provider", _fail)
    monkeypatch.setattr("app.rag.providers.provider_factory.get_vector_store", _fail)

    _override_dependencies(tmp_path)

    response = client.post(
        "/api/v1/documents",
        files={"file": ("report.pdf", b"content", "application/pdf")},
    )

    assert response.status_code == 202
