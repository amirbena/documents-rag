"""Local filesystem-backed `FileStorage` implementation.

Object keys are relative POSIX-style paths resolved safely under a configured root directory —
`validate_object_key` rejects absolute paths and `..` traversal before any filesystem call, so a
caller-supplied key can never escape the root. Blocking filesystem I/O runs via
`asyncio.to_thread` so this stays a well-behaved async dependency.
"""

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from app.storage.contract import FileMetadata, FileStorage, StoredFile
from app.storage.errors import (
    StorageDeleteError,
    StorageKeyError,
    StorageMetadataError,
    StorageObjectNotFoundError,
    StorageReadError,
    StorageUrlGenerationError,
    StorageWriteError,
)
from app.storage.keys import validate_object_key

DEFAULT_STORAGE_ROOT = Path("storage/documents")


class LocalFileStorage(FileStorage):
    """Saves/reads uploaded file content on local disk under a configured root directory."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = (root or DEFAULT_STORAGE_ROOT).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        """Validate `key` and resolve it to an absolute path guaranteed to live under the root."""
        validate_object_key(key)
        resolved = (self._root / key).resolve()
        try:
            resolved.relative_to(self._root)
        except ValueError as exc:
            raise StorageKeyError(f"Object key escapes the configured storage root: {key!r}") from exc
        return resolved

    async def save(
        self,
        key: str,
        content: bytes,
        *,
        content_type: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> StoredFile:
        """Write `content` to the local path resolved from `key`, creating parent directories."""
        path = self._resolve(key)
        try:
            await asyncio.to_thread(self._write_bytes, path, content)
        except OSError as exc:
            raise StorageWriteError(f"Failed to write object at key {key!r}") from exc
        return StoredFile(
            key=key, size_bytes=len(content), content_type=content_type, etag=None, metadata=metadata or {}
        )

    @staticmethod
    def _write_bytes(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    async def read(self, key: str) -> bytes:
        """Return the bytes stored at `key`; raises `StorageObjectNotFoundError` if missing."""
        path = self._resolve(key)
        if not path.exists():
            raise StorageObjectNotFoundError(f"No object found at key {key!r}")
        try:
            return await asyncio.to_thread(path.read_bytes)
        except OSError as exc:
            raise StorageReadError(f"Failed to read object at key {key!r}") from exc

    async def delete(self, key: str) -> None:
        """Delete the file at `key`; idempotent — a missing file is a successful no-op."""
        path = self._resolve(key)
        try:
            await asyncio.to_thread(path.unlink, missing_ok=True)
        except OSError as exc:
            raise StorageDeleteError(f"Failed to delete object at key {key!r}") from exc

    async def exists(self, key: str) -> bool:
        """Return whether a file exists at `key`."""
        path = self._resolve(key)
        try:
            return await asyncio.to_thread(path.exists)
        except OSError as exc:
            raise StorageMetadataError(f"Failed to check existence of key {key!r}") from exc

    async def get_metadata(self, key: str) -> FileMetadata:
        """Return `key`'s filesystem metadata (size, last-modified); no content-type/etag stored."""
        path = self._resolve(key)
        if not path.exists():
            raise StorageObjectNotFoundError(f"No object found at key {key!r}")
        try:
            stat = await asyncio.to_thread(path.stat)
        except OSError as exc:
            raise StorageMetadataError(f"Failed to stat object at key {key!r}") from exc
        return FileMetadata(
            key=key,
            size_bytes=stat.st_size,
            content_type=None,
            etag=None,
            last_modified=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
            metadata={},
        )

    async def generate_download_url(self, key: str, *, expiry_seconds: int | None = None) -> str:
        """Return a `file://` URI identifying the local path — an internal representation only.

        No browser-facing download route exists yet (out of scope for Phase 2.6/2.7 — see
        ARCHITECTURE.md's "Storage Abstraction" section), so this is not a usable HTTP URL. It
        exists to satisfy the `FileStorage` contract and is never persisted.
        """
        path = self._resolve(key)
        if not path.exists():
            raise StorageObjectNotFoundError(f"No object found at key {key!r}")
        try:
            return path.as_uri()
        except ValueError as exc:
            raise StorageUrlGenerationError(f"Failed to generate a URL for key {key!r}") from exc
