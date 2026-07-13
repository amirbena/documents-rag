"""Unit tests for app/services/indexing/collection_registry.py — collection safety and staleness,
no real database (a minimal fake session/vector-store double is enough for these unit-level
checks).
"""

from datetime import UTC, datetime

import pytest

from app.models.index_collection import IndexCollection
from app.services.indexing.collection_registry import (
    IncompatibleIndexConfigurationError,
    ensure_active_collection,
    is_document_stale,
    mark_document_indexed,
    retire_collection,
)
from tests.support.indexing.builders import build_document, build_embedding_config
from tests.support.indexing.fakes import FakeIndexSession, FakeVectorStore


async def test_ensure_active_collection_creates_a_compatible_collection() -> None:
    """A never-before-seen collection should be created with the config's dimension."""
    vector_store = FakeVectorStore(existing_dimension=None)
    session = FakeIndexSession()
    config = build_embedding_config()

    await ensure_active_collection(vector_store, session, config)

    assert vector_store.created_collections == [(config.collection_name, config.dimension)]
    tracked = await session.get(IndexCollection, config.collection_name)
    assert tracked is not None
    assert tracked.embedding_dimension == config.dimension


async def test_ensure_active_collection_reuses_a_compatible_collection() -> None:
    """An existing collection with a matching dimension should not raise, and stays tracked once."""
    config = build_embedding_config()
    vector_store = FakeVectorStore(existing_dimension=config.dimension)
    session = FakeIndexSession()

    await ensure_active_collection(vector_store, session, config)
    await ensure_active_collection(vector_store, session, config)

    assert len(session.index_collections) == 1


async def test_ensure_active_collection_rejects_incompatible_dimension() -> None:
    """An existing collection with a different dimension must raise, never be silently reused."""
    config = build_embedding_config(dimension=768)
    vector_store = FakeVectorStore(existing_dimension=1024)
    session = FakeIndexSession()

    with pytest.raises(IncompatibleIndexConfigurationError, match="768"):
        await ensure_active_collection(vector_store, session, config)


async def test_ensure_active_collection_never_deletes_a_mismatched_collection() -> None:
    """A dimension mismatch must fail loudly — no delete/recreate call is ever made."""

    class _NoDeleteVectorStore(FakeVectorStore):
        async def delete_by_document_id(self, collection_name: str, document_id: str) -> None:
            raise AssertionError("must never delete a mismatched collection automatically")

    vector_store = _NoDeleteVectorStore(existing_dimension=1024)
    session = FakeIndexSession()
    config = build_embedding_config(dimension=768)

    with pytest.raises(IncompatibleIndexConfigurationError):
        await ensure_active_collection(vector_store, session, config)

    assert vector_store.created_collections == []


def test_mark_document_indexed_sets_all_indexing_fields() -> None:
    """A successful index must persist provider/model/dimension/version/collection/indexed_at."""
    document = build_document()
    config = build_embedding_config()

    mark_document_indexed(document, config)

    assert document.embedding_provider == config.provider
    assert document.embedding_model == config.model
    assert document.embedding_dimension == config.dimension
    assert document.embedding_version == config.embedding_version
    assert document.chunking_version == config.chunking_version
    assert document.collection_name == config.collection_name
    assert document.indexed_at is not None
    assert document.indexed_at.tzinfo is not None


def test_never_indexed_document_is_stale() -> None:
    """A document with no collection_name at all must be reported stale."""
    document = build_document()

    assert is_document_stale(document, build_embedding_config()) is True


def test_document_indexed_under_active_config_is_not_stale() -> None:
    """A document whose stored collection_name matches the active config is current."""
    document = build_document()
    config = build_embedding_config()
    mark_document_indexed(document, config)

    assert is_document_stale(document, config) is False


def test_document_indexed_under_a_different_config_is_stale() -> None:
    """A document indexed under an old embedding_version must be reported stale after a bump."""
    document = build_document()
    old_config = build_embedding_config(embedding_version="v1")
    mark_document_indexed(document, old_config)

    new_config = build_embedding_config(embedding_version="v2")

    assert is_document_stale(document, new_config) is True


async def test_retire_collection_marks_status_without_deleting_anything() -> None:
    """retire_collection() must only flip the tracked status — no Qdrant call at all."""
    config = build_embedding_config()
    vector_store = FakeVectorStore(existing_dimension=None)
    session = FakeIndexSession()
    await ensure_active_collection(vector_store, session, config)

    await retire_collection(session, config.collection_name)

    tracked = await session.get(IndexCollection, config.collection_name)
    assert tracked is not None
    assert tracked.status.value == "retired"
    assert vector_store.deleted == []


def test_mark_document_indexed_uses_timezone_aware_utc() -> None:
    """indexed_at must be timezone-aware UTC, comparable to datetime.now(UTC) safely."""
    document = build_document()
    before = datetime.now(UTC)

    mark_document_indexed(document, build_embedding_config())

    after = datetime.now(UTC)
    assert document.indexed_at is not None
    assert before <= document.indexed_at <= after
