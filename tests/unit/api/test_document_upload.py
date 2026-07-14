"""Tests for POST /api/v1/documents with a fake DB session and local temp storage.

Outcome/status-mapping and deletion-conflict tests further down monkeypatch
`app.api.v1.routes.documents.upload_document` directly with a canned `UploadResult`/exception —
route-mapping is what's under test there, not the real dedup/concurrency service logic (already
covered by tests/unit/services/documents/test_upload_service.py).
"""

import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.v1.routes import documents as documents_route
from app.api.v1.routes.documents import get_file_storage
from app.db.session import get_db_session
from app.main import app
from app.models.document import Document
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.services.documents.dedup_service import (
    DeletionActiveError,
    DeletionIncompleteError,
    DeletionInvariantViolationError,
    MissingWinnerAfterRaceError,
    UploadOutcome,
)
from app.services.documents.upload_service import UploadResult
from app.storage.local_storage import LocalFileStorage

client = TestClient(app)


class _EmptyScalars:
    """Stand-in for `Result.scalars()` that never matches anything — no document pre-exists."""

    def first(self) -> None:
        return None


class _EmptyResult:
    """Stand-in for a SQLAlchemy `Result` reporting no rows — used for the dedup fast-path SELECT."""

    def scalars(self) -> _EmptyScalars:
        return _EmptyScalars()


class _FakeAsyncSession:
    """Minimal AsyncSession stand-in: records added rows, no real DB behind it.

    `execute()` always reports "no match" — these tests never pre-seed a matching content_hash,
    so every upload here takes the CREATED path, exactly as before content-hash deduplication.
    """

    def __init__(self) -> None:
        self.added: list[object] = []
        self.committed = False

    def add(self, instance: object) -> None:
        self.added.append(instance)

    async def execute(self, stmt: object) -> _EmptyResult:
        return _EmptyResult()

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        return None


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
    app.dependency_overrides[get_file_storage] = lambda: LocalFileStorage(root=tmp_path)
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
    assert document.storage_provider == "local"
    assert document.storage_key is not None
    assert document.storage_key.startswith(f"documents/{document.id}/")


def test_upload_persists_calculated_content_hash(tmp_path: Path) -> None:
    """A new Document's content_hash must be the lowercase hex SHA-256 of the uploaded bytes."""
    fake_session = _override_dependencies(tmp_path)
    content = b"%PDF-1.4 fake content for hashing"

    response = client.post(
        "/api/v1/documents",
        files={"file": ("report.pdf", content, "application/pdf")},
    )

    assert response.status_code == 202
    document, _job = fake_session.added
    assert document.content_hash == hashlib.sha256(content).hexdigest()
    assert len(document.content_hash) == 64
    assert document.content_hash == document.content_hash.lower()


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
    assert body["outcome"] == "CREATED"
    assert body["original_filename"] == "notes.txt"
    assert "content_hash" not in body
    assert "storage_key" not in body
    assert "storage_bucket" not in body


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
    # The uploaded content was actually written under the generated object key.
    stored_file = tmp_path / document.storage_key
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


# --- reuse outcomes: dynamic status + outcome mapping, via a monkeypatched upload_document -------


def _existing_document(*, original_filename: str = "invoice.pdf") -> Document:
    return Document(
        id="existing-doc-id",
        original_filename=original_filename,
        stored_filename="stored.pdf",
        content_type="application/pdf",
        file_size=10,
        stored_path="documents/existing-doc-id/stored.pdf",
        storage_provider="local",
        storage_key="documents/existing-doc-id/stored.pdf",
        content_hash="a" * 64,
    )


def _existing_job(status: IngestionStatus) -> IngestionJob:
    return IngestionJob(id="existing-job-id", document_id="existing-doc-id", status=status)


def _patch_upload_document(monkeypatch: pytest.MonkeyPatch, outcome: object) -> None:
    """Replace the route's `upload_document` with one that returns/raises `outcome` directly."""

    async def _fake_upload_document(**kwargs: object) -> UploadResult:
        if isinstance(outcome, Exception):
            raise outcome
        return outcome  # type: ignore[return-value]

    monkeypatch.setattr(documents_route, "upload_document", _fake_upload_document)


@pytest.mark.parametrize(
    ("job_status", "expected_outcome"),
    [
        (IngestionStatus.PENDING, UploadOutcome.REUSED_ACTIVE),
        (IngestionStatus.COMPLETED, UploadOutcome.REUSED_INDEXED),
        (IngestionStatus.FAILED, UploadOutcome.REUSED_FAILED),
    ],
)
def test_reused_outcomes_return_200_and_the_matching_public_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    job_status: IngestionStatus,
    expected_outcome: UploadOutcome,
) -> None:
    _override_dependencies(tmp_path)
    document = _existing_document()
    job = _existing_job(job_status)
    _patch_upload_document(
        monkeypatch, UploadResult(document=document, ingestion_job=job, outcome=expected_outcome)
    )

    response = client.post(
        "/api/v1/documents",
        files={"file": ("invoice.pdf", b"identical bytes", "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == expected_outcome.name
    assert body["document_id"] == document.id
    assert body["job_id"] == job.id


def test_reused_response_returns_existing_document_and_job_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _override_dependencies(tmp_path)
    document = _existing_document()
    job = _existing_job(IngestionStatus.COMPLETED)
    _patch_upload_document(
        monkeypatch, UploadResult(document=document, ingestion_job=job, outcome=UploadOutcome.REUSED_INDEXED)
    )

    response = client.post(
        "/api/v1/documents",
        files={"file": ("invoice.pdf", b"identical bytes", "application/pdf")},
    )

    body = response.json()
    assert body["document_id"] == "existing-doc-id"
    assert body["job_id"] == "existing-job-id"


def test_reused_response_returns_the_existing_original_filename_not_the_incoming_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same bytes uploaded under a different filename must report the authoritative existing name."""
    _override_dependencies(tmp_path)
    document = _existing_document(original_filename="invoice.pdf")
    job = _existing_job(IngestionStatus.COMPLETED)
    _patch_upload_document(
        monkeypatch, UploadResult(document=document, ingestion_job=job, outcome=UploadOutcome.REUSED_INDEXED)
    )

    response = client.post(
        "/api/v1/documents",
        files={"file": ("invoice-copy.pdf", b"identical bytes", "application/pdf")},
    )

    assert response.json()["original_filename"] == "invoice.pdf"


def test_reused_response_never_exposes_content_hash_or_storage_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _override_dependencies(tmp_path)
    document = _existing_document()
    job = _existing_job(IngestionStatus.COMPLETED)
    _patch_upload_document(
        monkeypatch, UploadResult(document=document, ingestion_job=job, outcome=UploadOutcome.REUSED_INDEXED)
    )

    response = client.post(
        "/api/v1/documents",
        files={"file": ("invoice.pdf", b"identical bytes", "application/pdf")},
    )

    body = response.json()
    assert "content_hash" not in body
    assert "storage_key" not in body
    assert "storage_bucket" not in body
    assert "storage_provider" not in body


# --- deletion-blocking conflicts: mapped to sanitized error responses ----------------------------


def test_active_deletion_maps_to_409(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _override_dependencies(tmp_path)
    _patch_upload_document(monkeypatch, DeletionActiveError("existing-doc-id"))

    response = client.post(
        "/api/v1/documents",
        files={"file": ("invoice.pdf", b"identical bytes", "application/pdf")},
    )

    assert response.status_code == 409


def test_incomplete_deletion_maps_to_409(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _override_dependencies(tmp_path)
    _patch_upload_document(monkeypatch, DeletionIncompleteError("existing-doc-id"))

    response = client.post(
        "/api/v1/documents",
        files={"file": ("invoice.pdf", b"identical bytes", "application/pdf")},
    )

    assert response.status_code == 409
    assert "PARTIALLY_FAILED" in response.json()["detail"]


def test_deletion_invariant_violation_is_sanitized_to_500(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _override_dependencies(tmp_path)
    _patch_upload_document(monkeypatch, DeletionInvariantViolationError("existing-doc-id"))

    response = client.post(
        "/api/v1/documents",
        files={"file": ("invoice.pdf", b"identical bytes", "application/pdf")},
    )

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "existing-doc-id" not in detail
    assert "content_hash" not in detail.lower()


def test_missing_winner_after_race_is_sanitized_to_500(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _override_dependencies(tmp_path)
    _patch_upload_document(monkeypatch, MissingWinnerAfterRaceError("a" * 64))

    response = client.post(
        "/api/v1/documents",
        files={"file": ("invoice.pdf", b"identical bytes", "application/pdf")},
    )

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "a" * 64 not in detail
