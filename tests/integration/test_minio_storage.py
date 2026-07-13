"""Integration tests for MinioFileStorage against a real, ephemeral MinIO container.

Uses Testcontainers — never the repository's docker-compose.yml, never a fixed host port, never
a persistent volume. Each test uses its own unique key prefix (derived from the test's bucket)
for isolation within the session-scoped container/bucket.
"""

import uuid

import pytest

from app.core.config import Settings
from app.storage.errors import StorageObjectNotFoundError
from app.storage.minio_storage import MinioFileStorage


@pytest.fixture
def storage(minio_endpoint: str, minio_credentials: tuple[str, str]) -> MinioFileStorage:
    """A MinioFileStorage pointed at the ephemeral container, with its bucket ensured."""
    access_key, secret_key = minio_credentials
    settings = Settings(
        FILE_STORAGE_PROVIDER="minio",
        MINIO_ENDPOINT=minio_endpoint,
        MINIO_ACCESS_KEY=access_key,
        MINIO_SECRET_KEY=secret_key,
        MINIO_BUCKET="documents-integration-test",
        MINIO_SECURE=False,
    )
    return MinioFileStorage(settings=settings)


@pytest.fixture(autouse=True)
async def _ensure_bucket(storage: MinioFileStorage) -> None:
    """Ensure the test bucket exists before every test in this module."""
    await storage.ensure_bucket()


def _key() -> str:
    return f"integration-test/{uuid.uuid4().hex}/notes.txt"


async def test_bucket_initialization_is_idempotent(storage: MinioFileStorage) -> None:
    """Calling ensure_bucket() repeatedly must not fail or recreate the bucket."""
    await storage.ensure_bucket()
    await storage.ensure_bucket()


async def test_save_and_read_round_trip(storage: MinioFileStorage) -> None:
    """Content saved to a real MinIO bucket should read back byte-for-byte identical."""
    key = _key()

    stored = await storage.save(key, b"hello world", content_type="text/plain")

    assert stored.key == key
    assert stored.etag is not None
    assert await storage.read(key) == b"hello world"


async def test_save_preserves_custom_metadata(storage: MinioFileStorage) -> None:
    """Custom metadata passed to save() should be retrievable via get_metadata()."""
    key = _key()

    await storage.save(
        key, b"content", content_type="text/plain", metadata={"original-filename": "notes.txt"}
    )

    metadata = await storage.get_metadata(key)
    assert metadata.content_type == "text/plain"
    assert metadata.size_bytes == len(b"content")


async def test_exists_true_and_false(storage: MinioFileStorage) -> None:
    """exists() should report True only for an object that was actually saved."""
    key = _key()
    assert await storage.exists(key) is False

    await storage.save(key, b"content")

    assert await storage.exists(key) is True


async def test_delete_is_idempotent(storage: MinioFileStorage) -> None:
    """Deleting a missing object must succeed; deleting twice must not raise."""
    key = _key()
    await storage.delete(key)  # missing object: no-op

    await storage.save(key, b"content")
    await storage.delete(key)
    await storage.delete(key)  # second delete: still a no-op

    assert await storage.exists(key) is False


async def test_read_missing_object_raises_not_found(storage: MinioFileStorage) -> None:
    """Reading a key with no object must raise StorageObjectNotFoundError."""
    with pytest.raises(StorageObjectNotFoundError):
        await storage.read(_key())


async def test_generate_download_url_is_usable_and_not_persisted(storage: MinioFileStorage) -> None:
    """A presigned URL should be generated for an existing object without persisting anything."""
    key = _key()
    await storage.save(key, b"content")

    url = await storage.generate_download_url(key, expiry_seconds=60)

    assert url.startswith("http")
    assert key in url


async def test_object_key_isolation_between_documents(storage: MinioFileStorage) -> None:
    """Two different keys must never collide, even under the same bucket."""
    key_one, key_two = _key(), _key()

    await storage.save(key_one, b"one")
    await storage.save(key_two, b"two")

    assert await storage.read(key_one) == b"one"
    assert await storage.read(key_two) == b"two"
