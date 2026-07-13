"""Tests for LocalFileStorage: save/read/delete/exists/metadata, path safety, retry semantics."""

from pathlib import Path

import pytest

from app.storage.contract import FileMetadata, StoredFile
from app.storage.errors import StorageKeyError, StorageObjectNotFoundError
from app.storage.local_storage import LocalFileStorage


@pytest.fixture
def storage(tmp_path: Path) -> LocalFileStorage:
    """A LocalFileStorage rooted at a fresh temp directory."""
    return LocalFileStorage(root=tmp_path / "root")


async def test_save_and_read_round_trip(storage: LocalFileStorage) -> None:
    """Content saved under a key should read back byte-for-byte identical."""
    stored = await storage.save("notes.txt", b"hello world", content_type="text/plain")

    assert isinstance(stored, StoredFile)
    assert stored.key == "notes.txt"
    assert stored.size_bytes == 11
    assert await storage.read("notes.txt") == b"hello world"


async def test_save_creates_nested_directories(storage: LocalFileStorage) -> None:
    """A nested key should have its parent directories created automatically."""
    await storage.save("documents/doc-1/file.txt", b"content")

    assert await storage.read("documents/doc-1/file.txt") == b"content"


async def test_save_overwrites_existing_key(storage: LocalFileStorage) -> None:
    """Saving to the same key twice should overwrite, not error — retry-safe."""
    await storage.save("notes.txt", b"first")
    await storage.save("notes.txt", b"second")

    assert await storage.read("notes.txt") == b"second"


async def test_read_missing_key_raises_not_found(storage: LocalFileStorage) -> None:
    """Reading a key with no object should raise StorageObjectNotFoundError."""
    with pytest.raises(StorageObjectNotFoundError):
        await storage.read("does_not_exist.txt")


async def test_delete_is_idempotent(storage: LocalFileStorage) -> None:
    """Deleting a missing object must succeed (no-op), not raise."""
    await storage.delete("does_not_exist.txt")  # must not raise

    await storage.save("notes.txt", b"content")
    await storage.delete("notes.txt")
    await storage.delete("notes.txt")  # second delete is still a no-op

    assert await storage.exists("notes.txt") is False


async def test_exists_true_and_false(storage: LocalFileStorage) -> None:
    """exists() should report True only for an object that was actually saved."""
    assert await storage.exists("notes.txt") is False

    await storage.save("notes.txt", b"content")

    assert await storage.exists("notes.txt") is True


async def test_get_metadata_returns_size_and_last_modified(storage: LocalFileStorage) -> None:
    """get_metadata() should report the object's size without needing to read its content."""
    await storage.save("notes.txt", b"hello world")

    metadata = await storage.get_metadata("notes.txt")

    assert isinstance(metadata, FileMetadata)
    assert metadata.key == "notes.txt"
    assert metadata.size_bytes == 11
    assert metadata.last_modified is not None


async def test_get_metadata_missing_key_raises_not_found(storage: LocalFileStorage) -> None:
    """get_metadata() on a missing key should raise StorageObjectNotFoundError."""
    with pytest.raises(StorageObjectNotFoundError):
        await storage.get_metadata("does_not_exist.txt")


async def test_generate_download_url_returns_a_uri(storage: LocalFileStorage) -> None:
    """generate_download_url() should return a file:// URI for an existing object."""
    await storage.save("notes.txt", b"content")

    url = await storage.generate_download_url("notes.txt")

    assert url.startswith("file://")


@pytest.mark.parametrize(
    "key", ["../escape.txt", "/etc/passwd", "documents/../../escape.txt", ""]
)
async def test_save_rejects_unsafe_keys(storage: LocalFileStorage, key: str) -> None:
    """save() must reject a key that would escape the configured root."""
    with pytest.raises(StorageKeyError):
        await storage.save(key, b"content")


async def test_nested_keys_are_isolated_from_each_other(storage: LocalFileStorage) -> None:
    """Two different nested keys must not collide."""
    await storage.save("documents/doc-1/file.txt", b"one")
    await storage.save("documents/doc-2/file.txt", b"two")

    assert await storage.read("documents/doc-1/file.txt") == b"one"
    assert await storage.read("documents/doc-2/file.txt") == b"two"
