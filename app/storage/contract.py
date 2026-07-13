"""The `FileStorage` contract every storage provider implements, plus its shared result types.

Application code (upload route, ingestion worker, extractor, reindex service) depends only on
this module's `FileStorage` — never on `LocalFileStorage`/`MinioFileStorage` concretely, and
never on a filesystem path or a provider SDK response type. `StoredFile`/`FileMetadata` are the
only shapes a caller ever sees back from a storage operation.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class StoredFile:
    """Provider-neutral result of a successful `FileStorage.save()` call."""

    key: str
    size_bytes: int
    content_type: str | None
    etag: str | None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FileMetadata:
    """Provider-neutral result of a successful `FileStorage.get_metadata()` call."""

    key: str
    size_bytes: int
    content_type: str | None
    etag: str | None
    last_modified: datetime | None
    metadata: Mapping[str, str] = field(default_factory=dict)


class FileStorage(ABC):
    """Provider-neutral contract for saving, reading, and managing uploaded document content.

    All object keys are provider-neutral strings (never a local absolute path or a MinIO
    endpoint URL) — see `app.storage.keys` for the shared key-generation helper. Implementations
    must translate every provider-specific failure into the `app.storage.errors.StorageError`
    hierarchy before it leaves the adapter.
    """

    @abstractmethod
    async def save(
        self,
        key: str,
        content: bytes,
        *,
        content_type: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> StoredFile:
        """Store `content` under `key`, overwriting any existing object at that key."""

    @abstractmethod
    async def read(self, key: str) -> bytes:
        """Return the exact bytes stored under `key`."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete the object at `key`; a no-op (not an error) if it does not exist."""

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Return whether an object exists at `key` — never swallows a provider failure as False."""

    @abstractmethod
    async def get_metadata(self, key: str) -> FileMetadata:
        """Return `key`'s metadata without downloading its full content."""

    @abstractmethod
    async def generate_download_url(self, key: str, *, expiry_seconds: int | None = None) -> str:
        """Return a URL to retrieve `key`'s content; never persisted as the object's identity."""
