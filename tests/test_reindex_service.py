"""Tests for reindex_document() against fake embedding/vector-store providers — no real network,
no real database (a minimal fake AsyncSession is enough since only
.add()/.get()/.commit()/.rollback()/.expire() are used).
"""

import uuid
from pathlib import Path

import pytest

import app.services.reindex_service as reindex_service_module
from app.core.config import get_settings
from app.models.document import Document
from app.models.vector_cleanup_job import VectorCleanupJob, VectorCleanupStatus
from app.rag.embedding_config import get_active_embedding_config
from app.rag.providers.vector_store import VectorPoint
from app.services.document_chunker import DocumentChunker
from app.services.index_registry import get_pending_cleanup_jobs
from app.services.reindex_service import ReindexOutcome, reindex_document
from app.storage.local_storage import LocalFileStorage


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
    def __init__(self, fail_delete_for: set[str] | None = None) -> None:
        self.created_collections: list[tuple[str, int]] = []
        self.upserted: dict[str, list[VectorPoint]] = {}
        self.deleted: list[tuple[str, str]] = []
        self._fail_delete_for = fail_delete_for or set()

    async def create_collection_if_not_exists(self, collection_name: str, vector_size: int) -> None:
        self.created_collections.append((collection_name, vector_size))

    async def upsert_vectors(self, collection_name: str, points: list[VectorPoint]) -> None:
        self.upserted.setdefault(collection_name, []).extend(points)

    async def get_collection_vector_size(self, collection_name: str) -> int | None:
        return None

    async def delete_by_document_id(self, collection_name: str, document_id: str) -> None:
        if collection_name in self._fail_delete_for:
            raise RuntimeError(f"could not delete from {collection_name}")
        self.deleted.append((collection_name, document_id))


class _FakeSession:
    """Minimal AsyncSession double: tracks commit/rollback/expire calls and stored rows."""

    def __init__(self, fail_commit: bool = False) -> None:
        self.commit_count = 0
        self.rollback_count = 0
        self.expired: list[object] = []
        self.added: list[object] = []
        self._cleanup_jobs: dict[str, VectorCleanupJob] = {}
        self._fail_commit = fail_commit

    def add(self, instance: object) -> None:
        self.added.append(instance)
        if isinstance(instance, VectorCleanupJob):
            self._cleanup_jobs[instance.id] = instance

    async def get(self, model: type, instance_id: str) -> object | None:
        # Pretend the active collection is already tracked, so ensure_active_collection() never
        # issues its own internal commit — tests here only care about the metadata commit that
        # reindex_document() itself performs.
        if model.__name__ == "IndexCollection":
            return object()
        return None

    async def execute(self, stmt: object):
        """Simulate: SELECT * FROM vector_cleanup_jobs WHERE status IN (pending, failed)."""
        matching = [
            job
            for job in self._cleanup_jobs.values()
            if job.status in (VectorCleanupStatus.PENDING, VectorCleanupStatus.FAILED)
        ]

        class _Scalars:
            def all(_self) -> list[VectorCleanupJob]:
                return matching

        class _Result:
            def scalars(_self) -> _Scalars:
                return _Scalars()

        return _Result()

    async def commit(self) -> None:
        if self._fail_commit:
            raise RuntimeError("db unavailable")
        self.commit_count += 1

    async def rollback(self) -> None:
        self.rollback_count += 1

    def expire(self, instance: object) -> None:
        self.expired.append(instance)


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

    document = _document(storage_provider="local", storage_key=file_path.name)
    session = _FakeSession()

    result = await reindex_document(document, session, file_storage=LocalFileStorage(root=tmp_path))

    assert result.outcome == ReindexOutcome.REINDEXED
    assert result.document is document
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

    result = await reindex_document(document, session, file_storage=LocalFileStorage(root=tmp_path))

    assert result.outcome == ReindexOutcome.ALREADY_CURRENT
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
        storage_provider="local",
        storage_key=file_path.name,
        collection_name="documents__ollama__old-model__ev0__cv0__d3",
        embedding_version="v0",
        chunking_version="v0",
    )
    session = _FakeSession()

    active_config = get_active_embedding_config()
    result = await reindex_document(document, session, file_storage=LocalFileStorage(root=tmp_path))

    assert result.outcome == ReindexOutcome.REINDEXED
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
    document = _document(storage_provider="local", storage_key=file_path.name)
    session = _FakeSession()

    try:
        await reindex_document(document, session, file_storage=LocalFileStorage(root=tmp_path))
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert str(exc) == "embedding unavailable"

    assert document.collection_name is None
    assert document.indexed_at is None
    assert vector_store.upserted == {}
    assert session.commit_count == 0
    assert session.rollback_count == 0  # failure happened before the metadata commit was attempted


async def test_reindex_is_idempotent_for_the_same_active_collection(tmp_path: Path, monkeypatch) -> None:
    """Re-running reindex twice against the same active config should not duplicate points."""
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world", encoding="utf-8")

    embedding_provider = _FakeEmbeddingProvider()
    vector_store = _FakeVectorStore()
    monkeypatch.setattr(reindex_service_module, "get_embedding_provider", lambda settings: embedding_provider)
    monkeypatch.setattr(reindex_service_module, "get_vector_store", lambda settings: vector_store)

    document = _document(storage_provider="local", storage_key=file_path.name)
    session = _FakeSession()

    file_storage = LocalFileStorage(root=tmp_path)
    await reindex_document(document, session, file_storage=file_storage)
    active_config = get_active_embedding_config()
    first_point_ids = [point.id for point in vector_store.upserted[active_config.collection_name]]

    # Force staleness again to exercise a second real re-index run (same active config).
    document.collection_name = None
    await reindex_document(document, session, file_storage=file_storage)
    all_points = vector_store.upserted[active_config.collection_name]
    second_point_ids = [point.id for point in all_points[len(first_point_ids) :]]

    assert first_point_ids == second_point_ids, "re-indexing must produce identical point IDs"


async def test_reindex_service_uses_existing_provider_factory() -> None:
    """reindex_service must resolve providers via the existing factory, never construct clients."""
    import inspect

    source = inspect.getsource(reindex_service_module)
    assert "from app.rag.providers.provider_factory import get_embedding_provider, get_vector_store" in source


# --- Transaction/failure semantics ---------------------------------------------------------------


async def test_commit_failure_rolls_back_and_expires_the_document(tmp_path: Path, monkeypatch) -> None:
    """A Postgres commit failure after a successful Qdrant upsert must roll back and expire."""
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world " * 50, encoding="utf-8")

    embedding_provider = _FakeEmbeddingProvider()
    vector_store = _FakeVectorStore()
    monkeypatch.setattr(reindex_service_module, "get_embedding_provider", lambda settings: embedding_provider)
    monkeypatch.setattr(reindex_service_module, "get_vector_store", lambda settings: vector_store)

    document = _document(storage_provider="local", storage_key=file_path.name)
    session = _FakeSession(fail_commit=True)

    with pytest.raises(RuntimeError, match="db unavailable"):
        await reindex_document(document, session, file_storage=LocalFileStorage(root=tmp_path))

    assert session.rollback_count == 1
    assert document in session.expired
    active_config = get_active_embedding_config()
    # The Qdrant write already happened (retry-safe via deterministic point IDs) — not undone.
    assert vector_store.upserted[active_config.collection_name]
    # No previous-collection cleanup is attempted when the metadata commit itself failed.
    assert vector_store.deleted == []


async def test_cleanup_failure_after_successful_commit_is_reindexed_with_cleanup_pending(
    tmp_path: Path, monkeypatch
) -> None:
    """A legacy-collection delete failure must not fail the re-index — it's tracked separately."""
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world " * 50, encoding="utf-8")

    previous_collection = "documents__ollama__old-model__ev0__cv0__d3"
    embedding_provider = _FakeEmbeddingProvider()
    vector_store = _FakeVectorStore(fail_delete_for={previous_collection})
    monkeypatch.setattr(reindex_service_module, "get_embedding_provider", lambda settings: embedding_provider)
    monkeypatch.setattr(reindex_service_module, "get_vector_store", lambda settings: vector_store)

    document = _document(
        storage_provider="local",
        storage_key=file_path.name,
        collection_name=previous_collection,
        embedding_version="v0",
        chunking_version="v0",
    )
    session = _FakeSession()

    result = await reindex_document(document, session, file_storage=LocalFileStorage(root=tmp_path))

    assert result.outcome == ReindexOutcome.REINDEXED_WITH_CLEANUP_PENDING
    # The document itself IS current — the re-index is not reported as a failure.
    active_config = get_active_embedding_config()
    assert document.collection_name == active_config.collection_name
    assert vector_store.deleted == []  # the delete attempt failed, so nothing was actually removed

    jobs = await get_pending_cleanup_jobs(session, document_id=document.id)
    assert len(jobs) == 1
    assert jobs[0].collection_name == previous_collection
    assert jobs[0].attempts == 1
    assert jobs[0].last_error is not None


async def test_zero_chunk_document_is_reindexed_empty(tmp_path: Path, monkeypatch) -> None:
    """A document producing zero chunks is marked current with REINDEXED_EMPTY, no vectors written."""
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world", encoding="utf-8")

    embedding_provider = _FakeEmbeddingProvider()
    vector_store = _FakeVectorStore()
    monkeypatch.setattr(reindex_service_module, "get_embedding_provider", lambda settings: embedding_provider)
    monkeypatch.setattr(reindex_service_module, "get_vector_store", lambda settings: vector_store)
    monkeypatch.setattr(DocumentChunker, "chunk", lambda self, extracted: [])

    document = _document(storage_provider="local", storage_key=file_path.name)
    session = _FakeSession()

    result = await reindex_document(document, session, file_storage=LocalFileStorage(root=tmp_path))

    assert result.outcome == ReindexOutcome.REINDEXED_EMPTY
    active_config = get_active_embedding_config()
    assert document.collection_name == active_config.collection_name
    assert document.indexed_at is not None
    assert embedding_provider.embed_calls == []
    assert vector_store.upserted == {}
