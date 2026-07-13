"""Local-storage-backed tests for app.services.documents.download_service.download_document().

Uses a real LocalFileStorage on a temp directory (no Postgres/Docker needed — the fake session
is enough here, since the goal is exercising real filesystem read behavior, not query behavior).
"""

import uuid
from pathlib import Path

from app.models.document import Document
from app.services.documents.download_service import download_document
from app.storage.local_storage import LocalFileStorage
from tests.support.documents.read.fake_session import FakeDocumentQuerySession


def _document(**overrides: object) -> Document:
    defaults: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        original_filename="report.pdf",
        stored_filename="stored.pdf",
        content_type="application/pdf",
        file_size=0,
        stored_path="documents/x/stored.pdf",
        storage_key="documents/x/stored.pdf",
    )
    defaults.update(overrides)
    return Document(**defaults)  # type: ignore[arg-type]


async def test_download_exact_byte_match_against_real_local_storage(tmp_path: Path) -> None:
    storage = LocalFileStorage(root=tmp_path)
    session = FakeDocumentQuerySession()
    doc = _document()
    session.add(doc)
    await storage.save(doc.storage_key, b"\x89PNG\r\n\x1a\nnot really a png but exact bytes")

    result = await download_document(session, doc.id, storage)

    assert result.status_code == 200
    assert result.content == b"\x89PNG\r\n\x1a\nnot really a png but exact bytes"
    assert result.content_type == "application/pdf"


async def test_download_unicode_filename_round_trips(tmp_path: Path) -> None:
    storage = LocalFileStorage(root=tmp_path)
    session = FakeDocumentQuerySession()
    hebrew_name = "מסמך_בדיקה.pdf"
    doc = _document(original_filename=hebrew_name)
    session.add(doc)
    await storage.save(doc.storage_key, b"content")

    result = await download_document(session, doc.id, storage)

    assert result.status_code == 200
    assert result.original_filename == hebrew_name


async def test_download_missing_object_is_409_no_absolute_path_leaked(tmp_path: Path) -> None:
    storage = LocalFileStorage(root=tmp_path)
    session = FakeDocumentQuerySession()
    doc = _document()
    session.add(doc)
    # Nothing saved under doc.storage_key.

    result = await download_document(session, doc.id, storage)

    assert result.status_code == 409
    assert result.content is None
    assert result.detail is not None
    assert str(tmp_path) not in result.detail
