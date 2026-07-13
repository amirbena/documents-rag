"""Tests for app.storage.keys: object-key generation and safety validation."""

import pytest

from app.storage.errors import StorageKeyError
from app.storage.keys import generate_object_key, resolve_document_storage_key, validate_object_key


def test_generate_object_key_is_namespaced_under_document_id() -> None:
    """A generated key must live under documents/{document_id}/."""
    key = generate_object_key("doc-123", "report.pdf")

    assert key.startswith("documents/doc-123/")
    assert key.endswith(".pdf")


def test_generate_object_key_is_unique_for_the_same_filename() -> None:
    """Two uploads of the same original filename must not collide on the same key."""
    key_one = generate_object_key("doc-123", "report.pdf")
    key_two = generate_object_key("doc-123", "report.pdf")

    assert key_one != key_two


def test_generate_object_key_drops_unsafe_suffix() -> None:
    """A filename with no extension or an unsafe suffix should not propagate an unsafe suffix."""
    key = generate_object_key("doc-123", "no_extension_file")

    assert not key.endswith("_file")


@pytest.mark.parametrize(
    "key",
    ["", "   ", "/absolute/path", "documents/../../../etc/passwd", "documents/..", ".."],
)
def test_validate_object_key_rejects_unsafe_keys(key: str) -> None:
    """Empty, absolute, and path-traversing keys must be rejected."""
    with pytest.raises(StorageKeyError):
        validate_object_key(key)


@pytest.mark.parametrize("key", ["notes.txt", "documents/doc-1/notes.txt", "a/b/c.pdf"])
def test_validate_object_key_accepts_safe_relative_keys(key: str) -> None:
    """A safe relative key should be returned unchanged."""
    assert validate_object_key(key) == key


class _FakeDocument:
    """Minimal Document stand-in exposing only the two fields resolve_document_storage_key reads."""

    def __init__(self, storage_key: str | None, stored_path: str) -> None:
        self.storage_key = storage_key
        self.stored_path = stored_path


def test_resolve_document_storage_key_prefers_storage_key() -> None:
    """A document with storage_key set should resolve to that key, not stored_path."""
    document = _FakeDocument(storage_key="documents/doc-1/file.txt", stored_path="legacy.txt")

    assert resolve_document_storage_key(document) == "documents/doc-1/file.txt"  # type: ignore[arg-type]


def test_resolve_document_storage_key_falls_back_to_stored_path() -> None:
    """A pre-migration document (storage_key IS NULL) should fall back to stored_path."""
    document = _FakeDocument(storage_key=None, stored_path="legacy_filename.txt")

    assert resolve_document_storage_key(document) == "legacy_filename.txt"  # type: ignore[arg-type]
