"""Tests for app/services/index_registry.py — collection safety and staleness, no real database
(a minimal fake session/vector-store double is enough for these unit-level checks)."""

import uuid
from datetime import UTC, datetime

import pytest

from app.models.document import Document
from app.models.index_collection import IndexCollection
from app.rag.embedding_config import EmbeddingIndexConfig
from app.services.index_registry import (
    IncompatibleIndexConfigurationError,
    delete_document_vectors,
    ensure_active_collection,
    is_document_stale,
    mark_document_indexed,
    retire_collection,
)


class _FakeVectorStore:
    def __init__(self, existing_dimension: int | None = None) -> None:
        self.existing_dimension = existing_dimension
        self.created_collections: list[tuple[str, int]] = []
        self.deleted: list[tuple[str, str]] = []

    async def get_collection_vector_size(self, collection_name: str) -> int | None:
        return self.existing_dimension

    async def create_collection_if_not_exists(self, collection_name: str, vector_size: int) -> None:
        self.created_collections.append((collection_name, vector_size))

    async def delete_by_document_id(self, collection_name: str, document_id: str) -> None:
        self.deleted.append((collection_name, document_id))


class _FakeSession:
    def __init__(self) -> None:
        self._index_collections: dict[str, IndexCollection] = {}
        self.commit_count = 0

    def add(self, instance: object) -> None:
        if isinstance(instance, IndexCollection):
            self._index_collections[instance.collection_name] = instance

    async def get(self, model: type, instance_id: str) -> object | None:
        if model is IndexCollection:
            return self._index_collections.get(instance_id)
        return None

    async def commit(self) -> None:
        self.commit_count += 1


def _config(**overrides: object) -> EmbeddingIndexConfig:
    fields: dict[str, object] = {
        "collection_prefix": "documents",
        "provider": "ollama",
        "model": "nomic-embed-text",
        "dimension": 768,
        "embedding_version": "v1",
        "chunking_version": "v1",
    }
    fields.update(overrides)
    return EmbeddingIndexConfig(**fields)  # type: ignore[arg-type]


def _document(**overrides: object) -> Document:
    fields: dict[str, object] = {
        "id": str(uuid.uuid4()),
        "original_filename": "notes.txt",
        "stored_filename": f"{uuid.uuid4().hex}.txt",
        "content_type": "text/plain",
        "file_size": 100,
        "stored_path": "unset",
    }
    fields.update(overrides)
    return Document(**fields)  # type: ignore[arg-type]


async def test_ensure_active_collection_creates_a_compatible_collection() -> None:
    """A never-before-seen collection should be created with the config's dimension."""
    vector_store = _FakeVectorStore(existing_dimension=None)
    session = _FakeSession()
    config = _config()

    await ensure_active_collection(vector_store, session, config)

    assert vector_store.created_collections == [(config.collection_name, config.dimension)]
    tracked = await session.get(IndexCollection, config.collection_name)
    assert tracked is not None
    assert tracked.embedding_dimension == config.dimension


async def test_ensure_active_collection_reuses_a_compatible_collection() -> None:
    """An existing collection with a matching dimension should not raise, and stays tracked once."""
    config = _config()
    vector_store = _FakeVectorStore(existing_dimension=config.dimension)
    session = _FakeSession()

    await ensure_active_collection(vector_store, session, config)
    await ensure_active_collection(vector_store, session, config)

    assert len(session._index_collections) == 1


async def test_ensure_active_collection_rejects_incompatible_dimension() -> None:
    """An existing collection with a different dimension must raise, never be silently reused."""
    config = _config(dimension=768)
    vector_store = _FakeVectorStore(existing_dimension=1024)
    session = _FakeSession()

    with pytest.raises(IncompatibleIndexConfigurationError, match="768"):
        await ensure_active_collection(vector_store, session, config)


async def test_ensure_active_collection_never_deletes_a_mismatched_collection() -> None:
    """A dimension mismatch must fail loudly — no delete/recreate call is ever made."""

    class _NoDeleteVectorStore(_FakeVectorStore):
        async def delete_by_document_id(self, collection_name: str, document_id: str) -> None:
            raise AssertionError("must never delete a mismatched collection automatically")

    vector_store = _NoDeleteVectorStore(existing_dimension=1024)
    session = _FakeSession()
    config = _config(dimension=768)

    with pytest.raises(IncompatibleIndexConfigurationError):
        await ensure_active_collection(vector_store, session, config)

    assert vector_store.created_collections == []


def test_mark_document_indexed_sets_all_indexing_fields() -> None:
    """A successful index must persist provider/model/dimension/version/collection/indexed_at."""
    document = _document()
    config = _config()

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
    document = _document()

    assert is_document_stale(document, _config()) is True


def test_document_indexed_under_active_config_is_not_stale() -> None:
    """A document whose stored collection_name matches the active config is current."""
    document = _document()
    config = _config()
    mark_document_indexed(document, config)

    assert is_document_stale(document, config) is False


def test_document_indexed_under_a_different_config_is_stale() -> None:
    """A document indexed under an old embedding_version must be reported stale after a bump."""
    document = _document()
    old_config = _config(embedding_version="v1")
    mark_document_indexed(document, old_config)

    new_config = _config(embedding_version="v2")

    assert is_document_stale(document, new_config) is True


async def test_retire_collection_marks_status_without_deleting_anything() -> None:
    """retire_collection() must only flip the tracked status — no Qdrant call at all."""
    config = _config()
    vector_store = _FakeVectorStore(existing_dimension=None)
    session = _FakeSession()
    await ensure_active_collection(vector_store, session, config)

    await retire_collection(session, config.collection_name)

    tracked = await session.get(IndexCollection, config.collection_name)
    assert tracked is not None
    assert tracked.status.value == "retired"
    assert vector_store.deleted == []


async def test_delete_document_vectors_targets_only_the_documents_tracked_collection() -> None:
    """delete_document_vectors() must delete from the document's own collection_name only."""
    vector_store = _FakeVectorStore()
    document = _document(collection_name="documents__ollama__m__ev1__cv1__d768")

    await delete_document_vectors(document, vector_store)

    assert vector_store.deleted == [("documents__ollama__m__ev1__cv1__d768", document.id)]


async def test_delete_document_vectors_is_a_noop_for_a_never_indexed_document() -> None:
    """A document with no collection_name has nothing to delete — no vector-store call at all."""

    class _AssertNoDeleteVectorStore(_FakeVectorStore):
        async def delete_by_document_id(self, collection_name: str, document_id: str) -> None:
            raise AssertionError("must not attempt to delete a never-indexed document's vectors")

    document = _document(collection_name=None)

    await delete_document_vectors(document, _AssertNoDeleteVectorStore())


def test_mark_document_indexed_uses_timezone_aware_utc() -> None:
    """indexed_at must be timezone-aware UTC, comparable to datetime.now(UTC) safely."""
    document = _document()
    before = datetime.now(UTC)

    mark_document_indexed(document, _config())

    after = datetime.now(UTC)
    assert document.indexed_at is not None
    assert before <= document.indexed_at <= after
