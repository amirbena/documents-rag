"""Integration tests for app.services.documents.download_service.download_document() against a real,
ephemeral MinIO container — proves the download path actually round-trips through the MinIO SDK,
not just a mocked httpx transport (see CLAUDE.md's "Use a real Qdrant/MinIO container" rule).
"""

import uuid

import pytest

from app.core.config import Settings
from app.models.document import Document
from app.services.documents.download_service import download_document
from app.storage.minio_storage import MinioFileStorage
from tests.support.documents.read.fake_session import FakeDocumentQuerySession

pytestmark = pytest.mark.integration


@pytest.fixture
def storage(minio_endpoint: str, minio_credentials: tuple[str, str]) -> MinioFileStorage:
    """A MinioFileStorage pointed at the ephemeral container, with its bucket ensured."""
    access_key, secret_key = minio_credentials
    settings = Settings(
        FILE_STORAGE_PROVIDER="minio",
        MINIO_ENDPOINT=minio_endpoint,
        MINIO_ACCESS_KEY=access_key,
        MINIO_SECRET_KEY=secret_key,
        MINIO_BUCKET="documents-download-test",
        MINIO_SECURE=False,
    )
    return MinioFileStorage(settings=settings)


@pytest.fixture(autouse=True)
async def _ensure_bucket(storage: MinioFileStorage) -> None:
    await storage.ensure_bucket()


def _document(**overrides: object) -> Document:
    defaults: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        original_filename="report.pdf",
        stored_filename="stored.pdf",
        content_type="application/pdf",
        file_size=0,
        stored_path=f"documents/{uuid.uuid4()}/stored.pdf",
        storage_provider="minio",
        storage_key=f"documents/{uuid.uuid4()}/stored.pdf",
    )
    defaults.update(overrides)
    return Document(**defaults)  # type: ignore[arg-type]


async def test_download_exact_byte_match_against_real_minio(storage: MinioFileStorage) -> None:
    session = FakeDocumentQuerySession()
    doc = _document()
    session.add(doc)
    await storage.save(doc.storage_key, b"real minio content, byte for byte")

    result = await download_document(session, doc.id, storage)

    assert result.status_code == 200
    assert result.content == b"real minio content, byte for byte"
    assert result.content_type == "application/pdf"


async def test_download_missing_object_is_409_against_real_minio(storage: MinioFileStorage) -> None:
    session = FakeDocumentQuerySession()
    doc = _document()
    session.add(doc)
    # Nothing saved under doc.storage_key in this bucket.

    result = await download_document(session, doc.id, storage)

    assert result.status_code == 409
    assert result.content is None
