"""MinIO/S3-compatible `FileStorage` implementation.

Uses the official `minio` Python SDK rather than raw `httpx` calls — unlike Ollama/Qdrant
elsewhere in this codebase, S3-compatible object storage requires request signing (SigV4) and a
non-trivial multi-part/streaming protocol; reimplementing that over raw HTTP would be a much
larger, riskier undertaking than this phase calls for, and the `minio` SDK is the idiomatic,
well-tested client for both MinIO and other S3-compatible backends. All SDK types/exceptions
(`minio.error.S3Error`, `urllib3` response objects, etc.) are translated to
`app.storage.errors.StorageError` subclasses before leaving this module — no SDK type is
returned to a caller.
"""

import asyncio
import io
from collections.abc import Mapping
from datetime import timedelta

from minio import Minio
from minio.deleteobjects import DeleteObject
from minio.error import S3Error
from urllib3.exceptions import MaxRetryError

from app.core.config import Settings
from app.storage.contract import FileMetadata, FileStorage, StoredFile
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
from app.storage.keys import validate_object_key

_NOT_FOUND_CODES = {"NoSuchKey", "NoSuchObject"}


class MinioFileStorage(FileStorage):
    """Saves/reads uploaded file content in a MinIO (or other S3-compatible) bucket."""

    def __init__(self, settings: Settings) -> None:
        if not settings.minio_endpoint or not settings.minio_bucket:
            raise StorageConfigurationError(
                "MINIO_ENDPOINT and MINIO_BUCKET are required when FILE_STORAGE_PROVIDER=minio."
            )
        self._bucket = settings.minio_bucket
        self._expiry_seconds = settings.minio_presigned_url_expiry_seconds
        self._create_bucket_if_missing = settings.minio_create_bucket_if_missing
        self._client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
            region=settings.minio_region,
        )

    async def ensure_bucket(self) -> None:
        """Create the configured bucket if it is missing and creation is enabled; else validate it exists.

        Safe under concurrent startup: `make_bucket` racing another process's creation surfaces as
        a `BucketAlreadyOwnedByYou`/`BucketAlreadyExists` S3Error, which is treated as success.
        """
        try:
            exists = await asyncio.to_thread(self._client.bucket_exists, self._bucket)
            if exists:
                return
            if not self._create_bucket_if_missing:
                raise StorageConfigurationError(
                    f"MinIO bucket {self._bucket!r} does not exist and "
                    "MINIO_CREATE_BUCKET_IF_MISSING is false."
                )
            await asyncio.to_thread(self._client.make_bucket, self._bucket)
        except S3Error as exc:
            if exc.code in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
                return
            raise StorageUnavailableError("MinIO bucket initialization failed.") from exc
        except MaxRetryError as exc:
            raise StorageUnavailableError("MinIO is unreachable.") from exc

    async def save(
        self,
        key: str,
        content: bytes,
        *,
        content_type: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> StoredFile:
        """Upload `content` to `key`, overwriting any existing object at that key."""
        validate_object_key(key)
        try:
            result = await asyncio.to_thread(
                self._client.put_object,
                self._bucket,
                key,
                io.BytesIO(content),
                length=len(content),
                content_type=content_type or "application/octet-stream",
                metadata=dict(metadata) if metadata else None,
            )
        except (S3Error, MaxRetryError) as exc:
            raise StorageWriteError(f"Failed to write object at key {key!r}") from exc
        return StoredFile(
            key=key,
            size_bytes=len(content),
            content_type=content_type,
            etag=result.etag,
            metadata=metadata or {},
        )

    async def read(self, key: str) -> bytes:
        """Return the bytes stored at `key`; raises `StorageObjectNotFoundError` if missing."""
        validate_object_key(key)
        response = None
        try:
            response = await asyncio.to_thread(self._client.get_object, self._bucket, key)
            return await asyncio.to_thread(response.read)
        except S3Error as exc:
            if exc.code in _NOT_FOUND_CODES:
                raise StorageObjectNotFoundError(f"No object found at key {key!r}") from exc
            raise StorageReadError(f"Failed to read object at key {key!r}") from exc
        except MaxRetryError as exc:
            raise StorageUnavailableError("MinIO is unreachable.") from exc
        finally:
            if response is not None:
                response.close()
                response.release_conn()

    async def delete(self, key: str) -> None:
        """Delete the object at `key`; idempotent — a missing object is a successful no-op."""
        validate_object_key(key)
        try:
            await asyncio.to_thread(self._client.remove_object, self._bucket, key)
        except S3Error as exc:
            if exc.code in _NOT_FOUND_CODES:
                return
            raise StorageDeleteError(f"Failed to delete object at key {key!r}") from exc
        except MaxRetryError as exc:
            raise StorageUnavailableError("MinIO is unreachable.") from exc

    async def exists(self, key: str) -> bool:
        """Return whether an object exists at `key`, using a stat (HEAD) call — never guesses."""
        validate_object_key(key)
        try:
            await asyncio.to_thread(self._client.stat_object, self._bucket, key)
            return True
        except S3Error as exc:
            if exc.code in _NOT_FOUND_CODES:
                return False
            raise StorageMetadataError(f"Failed to check existence of key {key!r}") from exc
        except MaxRetryError as exc:
            raise StorageUnavailableError("MinIO is unreachable.") from exc

    async def get_metadata(self, key: str) -> FileMetadata:
        """Return `key`'s object metadata via a stat (HEAD) call — no content is downloaded."""
        validate_object_key(key)
        try:
            stat = await asyncio.to_thread(self._client.stat_object, self._bucket, key)
        except S3Error as exc:
            if exc.code in _NOT_FOUND_CODES:
                raise StorageObjectNotFoundError(f"No object found at key {key!r}") from exc
            raise StorageMetadataError(f"Failed to stat object at key {key!r}") from exc
        except MaxRetryError as exc:
            raise StorageUnavailableError("MinIO is unreachable.") from exc
        return FileMetadata(
            key=key,
            size_bytes=stat.size or 0,
            content_type=stat.content_type,
            etag=stat.etag,
            last_modified=stat.last_modified,
            metadata=dict(stat.metadata) if stat.metadata else {},
        )

    async def generate_download_url(self, key: str, *, expiry_seconds: int | None = None) -> str:
        """Return a time-limited presigned GET URL; never persisted, never logged by this method."""
        validate_object_key(key)
        expiry = expiry_seconds or self._expiry_seconds
        try:
            return await asyncio.to_thread(
                self._client.presigned_get_object,
                self._bucket,
                key,
                expires=timedelta(seconds=expiry),
            )
        except (S3Error, MaxRetryError) as exc:
            raise StorageUrlGenerationError(f"Failed to generate a download URL for key {key!r}") from exc

    async def _cleanup_test_objects(self, keys: list[str]) -> None:
        """Best-effort batch delete — used only by integration tests to keep buckets clean."""
        delete_objects = [DeleteObject(key) for key in keys]
        await asyncio.to_thread(lambda: list(self._client.remove_objects(self._bucket, delete_objects)))
