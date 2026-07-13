"""Provider-neutral storage error hierarchy.

Every `FileStorage` implementation (`LocalFileStorage`, `MinioFileStorage`) must translate its
own failures (`OSError`, MinIO SDK exceptions, etc.) into one of these before the exception
leaves the adapter — no provider-specific exception type may reach service or route code. The
original exception is always preserved as `__cause__` (via `raise ... from exc`). Messages
include the operation and object key where safe, and never a credential, secret, or signed URL.
"""


class StorageError(Exception):
    """Base class for every provider-neutral storage failure."""


class StorageUnavailableError(StorageError):
    """The storage backend itself could not be reached (connection/auth/DNS failure)."""


class StorageObjectNotFoundError(StorageError):
    """The requested object key does not exist in the configured storage backend."""


class StorageWriteError(StorageError):
    """Writing (saving) an object failed."""


class StorageReadError(StorageError):
    """Reading an object's content failed (for a reason other than not-found)."""


class StorageDeleteError(StorageError):
    """Deleting an object failed (for a reason other than not-found, which is a no-op)."""


class StorageMetadataError(StorageError):
    """Retrieving or checking existence of an object's metadata failed."""


class StorageConfigurationError(StorageError):
    """The configured storage provider/settings are invalid or incomplete."""


class StorageUrlGenerationError(StorageError):
    """Generating a download URL for an object failed."""


class StorageKeyError(StorageError):
    """An object key is unsafe (path traversal, absolute path, or otherwise invalid)."""
