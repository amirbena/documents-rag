"""Unit tests for app/services/documents/download_service.py against a fake in-memory session.

Covers storage-key resolution, success/error status mapping, and deletion precedence (410 vs
200). No Postgres, no real filesystem — see test_download_service_local_storage.py for
real-LocalFileStorage-backed coverage and tests/integration/documents/download/test_minio.py for
real-MinIO coverage.
"""

from app.models.document_deletion_job import DocumentDeletionStatus
from app.services.documents.download_service import download_document
from app.storage.errors import StorageObjectNotFoundError, StorageUnavailableError
from tests.support.documents.read.builders import build_deletion_job, build_document
from tests.support.documents.read.fake_session import FakeDocumentQuerySession


class _FakeStorage:
    """Minimal FileStorage double: reads from an in-memory dict, or raises a canned error."""

    def __init__(self, objects: dict[str, bytes] | None = None, error: Exception | None = None) -> None:
        self._objects = objects or {}
        self._error = error
        self.read_calls: list[str] = []

    async def read(self, key: str) -> bytes:
        self.read_calls.append(key)
        if self._error is not None:
            raise self._error
        if key not in self._objects:
            raise StorageObjectNotFoundError(f"missing {key}")
        return self._objects[key]

    async def save(self, *a: object, **k: object) -> None:
        raise AssertionError("download must never write to storage")

    async def delete(self, *a: object, **k: object) -> None:
        raise AssertionError("download must never delete from storage")


async def test_download_document_missing_document_is_404() -> None:
    session = FakeDocumentQuerySession()
    storage = _FakeStorage()

    result = await download_document(session, "does-not-exist", storage)

    assert result.status_code == 404
    assert result.content is None
    assert storage.read_calls == []


async def test_download_document_success_reads_via_resolved_storage_key() -> None:
    session = FakeDocumentQuerySession()
    doc = build_document(0, storage_key="documents/abc/file.pdf")
    session.add(doc)
    storage = _FakeStorage(objects={"documents/abc/file.pdf": b"hello world"})

    result = await download_document(session, doc.id, storage)

    assert result.status_code == 200
    assert result.content == b"hello world"
    assert result.content_type == doc.content_type
    assert result.original_filename == doc.original_filename
    assert storage.read_calls == ["documents/abc/file.pdf"]


async def test_download_document_falls_back_to_stored_path_when_storage_key_is_null() -> None:
    """Pre-migration documents (storage_key IS NULL) resolve via stored_path — see storage.keys."""
    session = FakeDocumentQuerySession()
    doc = build_document(0, storage_key=None, stored_path="legacy/old-name.pdf")
    session.add(doc)
    storage = _FakeStorage(objects={"legacy/old-name.pdf": b"legacy content"})

    result = await download_document(session, doc.id, storage)

    assert result.status_code == 200
    assert result.content == b"legacy content"


async def test_download_document_missing_object_is_409_not_404() -> None:
    session = FakeDocumentQuerySession()
    doc = build_document()
    session.add(doc)
    storage = _FakeStorage(objects={})  # nothing stored -> StorageObjectNotFoundError

    result = await download_document(session, doc.id, storage)

    assert result.status_code == 409
    assert result.content is None


async def test_download_document_storage_unavailable_is_503() -> None:
    session = FakeDocumentQuerySession()
    doc = build_document()
    session.add(doc)
    storage = _FakeStorage(error=StorageUnavailableError("boom"))

    result = await download_document(session, doc.id, storage)

    assert result.status_code == 503
    assert result.content is None


async def test_download_document_completed_deletion_is_410_not_404() -> None:
    session = FakeDocumentQuerySession()
    doc = build_document(0, storage_key="documents/abc/file.pdf")
    session.add(doc)
    session.add(build_deletion_job(doc.id, DocumentDeletionStatus.COMPLETED))
    storage = _FakeStorage(objects={"documents/abc/file.pdf": b"hello world"})

    result = await download_document(session, doc.id, storage)

    assert result.status_code == 410
    assert result.content is None
    assert storage.read_calls == []


async def test_download_document_partially_failed_deletion_still_downloads() -> None:
    """A PARTIALLY_FAILED deletion never blocks download — the object may still be present."""
    session = FakeDocumentQuerySession()
    doc = build_document(0, storage_key="documents/abc/file.pdf")
    session.add(doc)
    session.add(build_deletion_job(doc.id, DocumentDeletionStatus.PARTIALLY_FAILED))
    storage = _FakeStorage(objects={"documents/abc/file.pdf": b"still here"})

    result = await download_document(session, doc.id, storage)

    assert result.status_code == 200
    assert result.content == b"still here"
