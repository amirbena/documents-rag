"""Unit tests for MinioFileStorage against a fake Minio SDK client double — no real MinIO.

Real-MinIO behavior is covered separately by tests/integration/test_minio_storage.py (Testcontainers).
These tests only verify: SDK exception translation into app.storage.errors.StorageError subclasses,
and that this module never returns a raw SDK/urllib3 type to a caller.
"""

from datetime import UTC, datetime

import pytest
from minio.error import S3Error

from app.core.config import Settings
from app.storage.contract import FileMetadata, StoredFile
from app.storage.errors import (
    StorageConfigurationError,
    StorageDeleteError,
    StorageMetadataError,
    StorageObjectNotFoundError,
    StorageReadError,
    StorageUnavailableError,
    StorageUrlGenerationError,
    StorageWriteError,
)
from app.storage.minio_storage import MinioFileStorage


def _settings(**overrides: object) -> Settings:
    fields = {
        "FILE_STORAGE_PROVIDER": "minio",
        "MINIO_ENDPOINT": "localhost:9000",
        "MINIO_ACCESS_KEY": "key",
        "MINIO_SECRET_KEY": "secret",
        "MINIO_BUCKET": "documents",
    }
    fields.update(overrides)
    return Settings(**fields)  # type: ignore[arg-type]


def _s3_error(code: str) -> S3Error:
    return S3Error(
        response=None,
        code=code,
        message="message",
        resource="resource",
        request_id="request_id",
        host_id="host_id",
    )


class _FakeStat:
    def __init__(self) -> None:
        self.size = 11
        self.etag = "abc123"
        self.content_type = "text/plain"
        self.last_modified = datetime(2026, 1, 1, tzinfo=UTC)
        self.metadata = {"x-amz-meta-original-filename": "notes.txt"}


class _FakePutResult:
    def __init__(self) -> None:
        self.etag = "abc123"


def test_missing_endpoint_or_bucket_raises_configuration_error() -> None:
    """Constructing MinioFileStorage without endpoint/bucket must fail clearly, no SDK client built.

    As of Phase 2.10, `Settings` itself rejects an incomplete MinIO configuration at construction
    time (see test_storage_factory.py / test_config.py), so an incomplete `Settings` instance can
    no longer be built through normal construction. This check in `MinioFileStorage.__init__`
    remains as defense in depth — exercised here via `model_construct()`, which bypasses Settings'
    own validators, to simulate the only way an incomplete instance could still reach this code.
    """
    base_fields = _settings().model_dump()
    with pytest.raises(StorageConfigurationError):
        MinioFileStorage(settings=Settings.model_construct(**{**base_fields, "minio_endpoint": None}))

    with pytest.raises(StorageConfigurationError):
        MinioFileStorage(settings=Settings.model_construct(**{**base_fields, "minio_bucket": None}))


async def test_save_translates_s3_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A put_object S3Error must translate to StorageWriteError, preserving the cause."""
    storage = MinioFileStorage(settings=_settings())

    def _raise(*args: object, **kwargs: object) -> None:
        raise _s3_error("InternalError")

    monkeypatch.setattr(storage._client, "put_object", _raise)

    with pytest.raises(StorageWriteError) as exc_info:
        await storage.save("notes.txt", b"content")
    assert isinstance(exc_info.value.__cause__, S3Error)


async def test_save_returns_typed_stored_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful save() must return a StoredFile, never the raw SDK result object."""
    storage = MinioFileStorage(settings=_settings())
    monkeypatch.setattr(storage._client, "put_object", lambda *a, **k: _FakePutResult())

    result = await storage.save("notes.txt", b"hello world", content_type="text/plain")

    assert isinstance(result, StoredFile)
    assert result.key == "notes.txt"
    assert result.etag == "abc123"


async def test_read_missing_object_raises_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """A get_object NoSuchKey S3Error must translate to StorageObjectNotFoundError."""
    storage = MinioFileStorage(settings=_settings())

    def _raise(*args: object, **kwargs: object) -> None:
        raise _s3_error("NoSuchKey")

    monkeypatch.setattr(storage._client, "get_object", _raise)

    with pytest.raises(StorageObjectNotFoundError):
        await storage.read("missing.txt")


async def test_read_other_failure_raises_read_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A get_object failure that isn't not-found must translate to StorageReadError."""
    storage = MinioFileStorage(settings=_settings())

    def _raise(*args: object, **kwargs: object) -> None:
        raise _s3_error("InternalError")

    monkeypatch.setattr(storage._client, "get_object", _raise)

    with pytest.raises(StorageReadError):
        await storage.read("notes.txt")


async def test_delete_is_idempotent_on_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """delete() on a missing object (NoSuchKey) must succeed silently, not raise."""
    storage = MinioFileStorage(settings=_settings())

    def _raise(*args: object, **kwargs: object) -> None:
        raise _s3_error("NoSuchKey")

    monkeypatch.setattr(storage._client, "remove_object", _raise)

    await storage.delete("missing.txt")  # must not raise


async def test_delete_other_failure_raises_delete_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A remove_object failure that isn't not-found must translate to StorageDeleteError."""
    storage = MinioFileStorage(settings=_settings())

    def _raise(*args: object, **kwargs: object) -> None:
        raise _s3_error("InternalError")

    monkeypatch.setattr(storage._client, "remove_object", _raise)

    with pytest.raises(StorageDeleteError):
        await storage.delete("notes.txt")


async def test_exists_false_on_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """exists() must return False for a NoSuchKey stat_object failure."""
    storage = MinioFileStorage(settings=_settings())

    def _raise(*args: object, **kwargs: object) -> None:
        raise _s3_error("NoSuchKey")

    monkeypatch.setattr(storage._client, "stat_object", _raise)

    assert await storage.exists("missing.txt") is False


async def test_exists_does_not_hide_a_provider_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """exists() must raise (not return False) when stat_object fails for a non-not-found reason."""
    storage = MinioFileStorage(settings=_settings())

    def _raise(*args: object, **kwargs: object) -> None:
        raise _s3_error("InternalError")

    monkeypatch.setattr(storage._client, "stat_object", _raise)

    with pytest.raises(StorageMetadataError):
        await storage.exists("notes.txt")


async def test_get_metadata_returns_typed_file_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_metadata() must return a FileMetadata, mapping every field from the SDK stat result."""
    storage = MinioFileStorage(settings=_settings())
    monkeypatch.setattr(storage._client, "stat_object", lambda *a, **k: _FakeStat())

    metadata = await storage.get_metadata("notes.txt")

    assert isinstance(metadata, FileMetadata)
    assert metadata.size_bytes == 11
    assert metadata.etag == "abc123"
    assert metadata.content_type == "text/plain"


async def test_generate_download_url_translates_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A presigned_get_object failure must translate to StorageUrlGenerationError."""
    storage = MinioFileStorage(settings=_settings())

    def _raise(*args: object, **kwargs: object) -> None:
        raise _s3_error("InternalError")

    monkeypatch.setattr(storage._client, "presigned_get_object", _raise)

    with pytest.raises(StorageUrlGenerationError):
        await storage.generate_download_url("notes.txt")


async def test_ensure_bucket_translates_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bucket_exists failure must translate to StorageUnavailableError, not an SDK exception."""
    storage = MinioFileStorage(settings=_settings())

    def _raise(*args: object, **kwargs: object) -> None:
        raise _s3_error("InternalError")

    monkeypatch.setattr(storage._client, "bucket_exists", _raise)

    with pytest.raises(StorageUnavailableError):
        await storage.ensure_bucket()


async def test_ensure_bucket_creates_missing_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    """ensure_bucket() should create the bucket when it doesn't exist and creation is enabled."""
    storage = MinioFileStorage(settings=_settings())
    created: list[str] = []
    monkeypatch.setattr(storage._client, "bucket_exists", lambda *a, **k: False)
    monkeypatch.setattr(storage._client, "make_bucket", lambda bucket: created.append(bucket))

    await storage.ensure_bucket()

    assert created == ["documents"]


async def test_ensure_bucket_does_not_recreate_existing_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    """ensure_bucket() must not call make_bucket when the bucket already exists."""
    storage = MinioFileStorage(settings=_settings())
    monkeypatch.setattr(storage._client, "bucket_exists", lambda *a, **k: True)

    def _fail_if_called(*a: object, **k: object) -> None:
        raise AssertionError("make_bucket must not be called for an existing bucket")

    monkeypatch.setattr(storage._client, "make_bucket", _fail_if_called)

    await storage.ensure_bucket()  # must not raise


async def test_ensure_bucket_respects_create_bucket_if_missing_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """When MINIO_CREATE_BUCKET_IF_MISSING is false, a missing bucket must fail explicitly."""
    storage = MinioFileStorage(settings=_settings(MINIO_CREATE_BUCKET_IF_MISSING=False))
    monkeypatch.setattr(storage._client, "bucket_exists", lambda *a, **k: False)

    with pytest.raises(StorageConfigurationError):
        await storage.ensure_bucket()
