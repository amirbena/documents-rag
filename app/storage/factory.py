"""Resolves the configured `FileStorage` implementation from settings.

Mirrors `app/rag/providers/provider_factory.py`'s pattern exactly: one function, one `if` on the
settings string, one dedicated error. This is the *only* place any code constructs
`LocalFileStorage`/`MinioFileStorage` — no route, worker, or service may instantiate a concrete
storage class directly or branch on `settings.file_storage_provider` itself.
"""

from pathlib import Path

from app.core.config import Settings, get_settings
from app.storage.contract import FileStorage
from app.storage.errors import StorageConfigurationError
from app.storage.local_storage import LocalFileStorage
from app.storage.minio_storage import MinioFileStorage


def create_file_storage(settings: Settings | None = None) -> FileStorage:
    """Return the FileStorage implementation configured via FILE_STORAGE_PROVIDER."""
    settings = settings or get_settings()
    provider = settings.file_storage_provider

    if provider == "local":
        return LocalFileStorage(root=Path(settings.local_storage_root))

    if provider == "minio":
        return MinioFileStorage(settings=settings)

    raise StorageConfigurationError(
        f"Unsupported FILE_STORAGE_PROVIDER: {provider!r}. Supported providers: 'local', 'minio'."
    )
