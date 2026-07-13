"""Provider-neutral object-key generation and safety validation.

Storage providers never invent their own keys — key generation is shared application logic, so
the same deterministic key addresses the same content regardless of which `FileStorage`
implementation is configured. `validate_object_key` is the one place path-traversal/absolute-path
rejection is enforced; `LocalFileStorage` calls it before resolving a key to a filesystem path.
"""

import re
import uuid
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from app.storage.errors import StorageKeyError

if TYPE_CHECKING:
    from app.models.document import Document

_MAX_SAFE_SUFFIX_LENGTH = 10
_SAFE_SUFFIX_PATTERN = re.compile(r"^[A-Za-z0-9]+$")


def generate_object_key(document_id: str, original_filename: str) -> str:
    """Build a deterministic, provider-neutral key: documents/{document_id}/{uuid-hex}{suffix}.

    The random component keeps two uploads with the same original filename from colliding; the
    document-id prefix keeps every document's content isolated under its own key namespace.
    """
    suffix = Path(original_filename).suffix
    is_safe = 1 < len(suffix) <= _MAX_SAFE_SUFFIX_LENGTH and bool(_SAFE_SUFFIX_PATTERN.match(suffix[1:]))
    safe_suffix = suffix if is_safe else ""
    return f"documents/{document_id}/{uuid.uuid4().hex}{safe_suffix}"


def validate_object_key(key: str) -> str:
    """Reject an unsafe object key (empty, absolute, or path-traversing); return it unchanged."""
    if not key or not key.strip():
        raise StorageKeyError("Object key must not be empty.")

    posix_key = PurePosixPath(key)
    if posix_key.is_absolute():
        raise StorageKeyError(f"Object key must not be an absolute path: {key!r}")

    if ".." in posix_key.parts:
        raise StorageKeyError(f"Object key must not contain '..': {key!r}")

    return key


def resolve_document_storage_key(document: "Document") -> str:
    """Return the key to address `document`'s content in `FileStorage`.

    Backward compatibility: a document written before Phase 2.6/2.7 has `storage_key IS NULL` —
    its `stored_path` value (its pre-migration local filename) is treated as the storage key
    instead. See `app/models/document.py`'s module docstring and the Alembic migration that
    backfills `storage_provider='local'`/`storage_key=stored_filename` for existing rows.
    """
    return document.storage_key or document.stored_path
