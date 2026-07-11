"""Tests for reindex_document() against fake embedding/vector-store providers — no real network,
no real database (a minimal fake AsyncSession is enough since only .commit()/.get() are used).
"""

import uuid
from pathlib import Path

import pytest

import app.services.reindex_service as reindex_service_module
from app.core.config import get_settings
from app.models.document import Document
from app.rag.embedding_config import get_active_embedding_config
from app.rag.providers.vector_store import VectorPoint
from app.services.reindex_service import reindex_document


@pytest.fixture(autouse=True)
def _fake_embedding_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    """Match the active config's dimension to _FakeEmbeddingProvider's 3-dim output."""
    monkeypatch.setattr(get_settings(), "vector_size", 3)


class _FakeEmbeddingProvider:
    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [0.1, 0.2, 0.3]
        self.embed_calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(texts)
        return [self.vector for _ in texts]


class _FakeVectorStore:
    def __init__(self) -> None:
        self.created_collections: list[tuple[str, int]] = []
        self.upserted: dict[str, list[VectorPoint]] = {}
        self.deleted: list[tuple[str, str]] = []

    async def create_collection_if_not_exists(self, collection_name: str, vector_size: int) -> None:
        self.created_collections.append((collection_name, vector_size))

    async def upsert_vectors(self, collection_name: str, points: list[VectorPoint]) -> None:
        self.upserted.setdefault(collection_name, []).extend(points)

    async def get_collection_vector_size(self, collection_name: str) -> int | None:
        return None

    async def delete_by_document_id(self, collection_name: str, document_id: str) -> None:
        self.deleted.append((collection_name, document_id))


class _FakeSession:
    def __init__(self) -> None:
        self.commit_count = 0
        self._index_collections: dict[str, object] = {}

    def add(self, instance: object) -> None:
        pass

    async def get(self, model: type, instance_id: str) -> object | None:
        return None

    async def commit(self) -> None:
        self.commit_count += 1


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


async def test_reindex_of_a_never_indexed_document_writes_vectors_and_marks_indexed(
    tmp_path: Path, monkeypatch
) -> None:
    """A document with no prior collection_name should be embedded and upserted, then marked."""
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world " * 50, encoding="utf-8")

    embedding_provider = _FakeEmbeddingProvider()
    vector_store = _FakeVectorStore()
    monkeypatch.setattr(reindex_service_module, "get_embedding_provider", lambda settings: embedding_provider)
    monkeypatch.setattr(reindex_service_module, "get_vector_store", lambda settings: vector_store)

    document = _document(stored_path=str(file_path))
    session = _FakeSession()

    result = await reindex_document(document, session)

    assert result is True
    active_config = get_active_embedding_config()
    assert document.collection_name == active_config.collection_name
    assert document.indexed_at is not None
    assert vector_store.upserted[active_config.collection_name]
    assert vector_store.deleted == []  # nothing to clean up — no previous collection


async def test_reindex_of_an_already_current_document_is_a_noop(tmp_path: Path, monkeypatch) -> None:
    """A document already indexed under the active config should not be re-embedded at all."""
    embedding_provider = _FakeEmbeddingProvider()
    vector_store = _FakeVectorStore()
    monkeypatch.setattr(reindex_service_module, "get_embedding_provider", lambda settings: embedding_provider)
    monkeypatch.setattr(reindex_service_module, "get_vector_store", lambda settings: vector_store)

    active_config = get_active_embedding_config()
    document = _document(
        collection_name=active_config.collection_name,
        embedding_provider=active_config.provider,
        embedding_model=active_config.model,
        embedding_dimension=active_config.dimension,
        embedding_version=active_config.embedding_version,
        chunking_version=active_config.chunking_version,
    )
    session = _FakeSession()

    result = await reindex_document(document, session)

    assert result is True
    assert embedding_provider.embed_calls == []
    assert vector_store.upserted == {}


async def test_reindex_into_a_new_version_deletes_the_previous_collections_vectors(
    tmp_path: Path, monkeypatch
) -> None:
    """After a successful re-index into a new collection, the old collection is cleaned up."""
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world " * 50, encoding="utf-8")

    embedding_provider = _FakeEmbeddingProvider()
    vector_store = _FakeVectorStore()
    monkeypatch.setattr(reindex_service_module, "get_embedding_provider", lambda settings: embedding_provider)
    monkeypatch.setattr(reindex_service_module, "get_vector_store", lambda settings: vector_store)

    document = _document(
        stored_path=str(file_path),
        collection_name="documents__ollama__old-model__ev0__cv0__d3",
        embedding_version="v0",
        chunking_version="v0",
    )
    session = _FakeSession()

    active_config = get_active_embedding_config()
    result = await reindex_document(document, session)

    assert result is True
    assert document.collection_name == active_config.collection_name
    assert vector_store.deleted == [("documents__ollama__old-model__ev0__cv0__d3", document.id)]


async def test_reindex_failure_does_not_mark_document_current(tmp_path: Path, monkeypatch) -> None:
    """A failure during extraction/embedding must leave the document's indexing metadata untouched."""

    class _FailingEmbeddingProvider:
        async def embed(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("embedding unavailable")

    vector_store = _FakeVectorStore()
    monkeypatch.setattr(
        reindex_service_module, "get_embedding_provider", lambda settings: _FailingEmbeddingProvider()
    )
    monkeypatch.setattr(reindex_service_module, "get_vector_store", lambda settings: vector_store)

    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world " * 50, encoding="utf-8")
    document = _document(stored_path=str(file_path))
    session = _FakeSession()

    try:
        await reindex_document(document, session)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert str(exc) == "embedding unavailable"

    assert document.collection_name is None
    assert document.indexed_at is None
    assert vector_store.upserted == {}


async def test_reindex_is_idempotent_for_the_same_active_collection(tmp_path: Path, monkeypatch) -> None:
    """Re-running reindex twice against the same active config should not duplicate points."""
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world", encoding="utf-8")

    embedding_provider = _FakeEmbeddingProvider()
    vector_store = _FakeVectorStore()
    monkeypatch.setattr(reindex_service_module, "get_embedding_provider", lambda settings: embedding_provider)
    monkeypatch.setattr(reindex_service_module, "get_vector_store", lambda settings: vector_store)

    document = _document(stored_path=str(file_path))
    session = _FakeSession()

    await reindex_document(document, session)
    active_config = get_active_embedding_config()
    first_point_ids = [point.id for point in vector_store.upserted[active_config.collection_name]]

    # Force staleness again to exercise a second real re-index run (same active config).
    document.collection_name = None
    await reindex_document(document, session)
    all_points = vector_store.upserted[active_config.collection_name]
    second_point_ids = [point.id for point in all_points[len(first_point_ids) :]]

    assert first_point_ids == second_point_ids, "re-indexing must produce identical point IDs"


def test_reindex_service_uses_existing_provider_factory() -> None:
    """reindex_service must resolve providers via the existing factory, never construct clients."""
    import inspect

    source = inspect.getsource(reindex_service_module)
    assert "from app.rag.providers.provider_factory import get_embedding_provider, get_vector_store" in source
