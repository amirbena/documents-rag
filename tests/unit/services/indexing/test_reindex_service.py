"""Tests for build_reindex_target()/activate_reindexed_document() against fake embedding/
vector-store providers — no real network, no real database (a minimal fake AsyncSession is
enough since only .add()/.get()/.commit()/.rollback()/.expire()/.execute() are used).

Split to mirror the production split (Phase 2.8.6, subtask 1): build-target tests prove a
successful build never touches the document's current serving metadata or previous vectors;
activation tests prove the metadata switch and deferred-cleanup-job persistence are atomic.
"""

import inspect
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

import app.services.indexing.reindex_service as reindex_service_module
from app.core.config import Settings
from app.models.document import Document
from app.models.vector_cleanup_job import VectorCleanupJob, VectorCleanupStatus
from app.rag.embedding_config import EmbeddingIndexConfig
from app.rag.embedding_validation import EmbeddingDimensionMismatchError
from app.services.documents.chunker import DocumentChunker
from app.services.documents.text_extractor import DocumentTextExtractionError
from app.services.indexing.reindex_service import (
    ReindexBuildOutcome,
    TargetConfigurationMismatchError,
    activate_reindexed_document,
    build_reindex_target,
    build_settings_for_target,
)
from app.storage.local_storage import LocalFileStorage

# --- shared fixtures/helpers ----------------------------------------------------------------


def _base_settings(**overrides: object) -> Settings:
    """A 'live process' Settings instance, deliberately unlike any target used in these tests."""
    fields: dict[str, object] = dict(
        EMBEDDING_PROVIDER="ollama",
        EMBEDDING_MODEL=None,
        OLLAMA_EMBEDDING_MODEL="live-model",
        VECTOR_SIZE=999,
        EMBEDDING_VERSION="v-live",
        CHUNKING_VERSION="v-live",
        QDRANT_COLLECTION_NAME="live-prefix",
        CHUNK_SIZE=111,
        CHUNK_OVERLAP=22,
    )
    fields.update(overrides)
    return Settings(**fields)  # type: ignore[arg-type]


def _target_config(**overrides: object) -> EmbeddingIndexConfig:
    fields: dict[str, object] = dict(
        collection_prefix="documents",
        provider="ollama",
        model="target-model",
        dimension=3,
        embedding_version="v9",
        chunking_version="v9",
    )
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


class _CapturingEmbeddingProvider:
    """Records every settings object it's constructed for and every embed() call's texts."""

    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [0.1, 0.2, 0.3]
        self.embed_calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(texts)
        return [self.vector for _ in texts]


class _FailingEmbeddingProvider:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding unavailable")


class _FakeVectorStore:
    def __init__(self) -> None:
        self.created_collections: list[tuple[str, int]] = []
        self.upserted: dict[str, list] = {}
        self.deleted: list[tuple[str, str]] = []
        self.raise_on_upsert = False

    async def create_collection_if_not_exists(self, collection_name: str, vector_size: int) -> None:
        self.created_collections.append((collection_name, vector_size))

    async def upsert_vectors(self, collection_name: str, points: list) -> None:
        if self.raise_on_upsert:
            raise RuntimeError("Qdrant unreachable")
        self.upserted.setdefault(collection_name, []).extend(points)

    async def get_collection_vector_size(self, collection_name: str) -> int | None:
        return None

    async def delete_by_document_id(self, collection_name: str, document_id: str) -> None:
        self.deleted.append((collection_name, document_id))


class _FakeSession:
    """Minimal AsyncSession double: tracks commit/rollback/expire calls and stored rows."""

    def __init__(self, fail_commit: bool = False) -> None:
        self.commit_count = 0
        self.rollback_count = 0
        self.expired: list[object] = []
        self.added: list[object] = []
        self._fail_commit = fail_commit
        self.commit_snapshots: list[str | None] = []
        self._document_for_snapshot: Document | None = None

    def watch(self, document: Document) -> None:
        """Record document.collection_name at the moment commit() is called."""
        self._document_for_snapshot = document

    def add(self, instance: object) -> None:
        self.added.append(instance)

    async def get(self, model: type, instance_id: str) -> object | None:
        # Pretend the target collection is already tracked, so ensure_active_collection() never
        # issues its own internal commit — these tests only care about commits build/activate
        # themselves perform.
        if model.__name__ == "IndexCollection":
            return object()
        return None

    async def commit(self) -> None:
        if self._document_for_snapshot is not None:
            self.commit_snapshots.append(self._document_for_snapshot.collection_name)
        if self._fail_commit:
            raise RuntimeError("db unavailable")
        self.commit_count += 1

    async def rollback(self) -> None:
        self.rollback_count += 1

    def expire(self, instance: object) -> None:
        self.expired.append(instance)


# --- build_settings_for_target --------------------------------------------------------------


def test_build_settings_for_target_overrides_every_build_relevant_field() -> None:
    base = _base_settings()
    target = _target_config()

    derived = build_settings_for_target(base, target, chunk_size=500, chunk_overlap=50)

    assert derived.embedding_provider == target.provider
    assert derived.embedding_model == target.model
    assert derived.ollama_embedding_model == target.model
    assert derived.vector_size == target.dimension
    assert derived.embedding_version == target.embedding_version
    assert derived.chunking_version == target.chunking_version
    assert derived.qdrant_collection_name == target.collection_prefix
    assert derived.chunk_size == 500
    assert derived.chunk_overlap == 50


def test_build_settings_for_target_does_not_mutate_the_base_settings() -> None:
    base = _base_settings()
    target = _target_config()

    build_settings_for_target(base, target, chunk_size=500, chunk_overlap=50)

    assert base.embedding_provider == "ollama"
    assert base.ollama_embedding_model == "live-model"
    assert base.embedding_model is None
    assert base.vector_size == 999
    assert base.embedding_version == "v-live"
    assert base.chunking_version == "v-live"
    assert base.qdrant_collection_name == "live-prefix"
    assert base.chunk_size == 111
    assert base.chunk_overlap == 22


def test_build_settings_for_target_reproduces_the_exact_target_collection_name() -> None:
    from app.rag.embedding_config import get_active_embedding_config

    base = _base_settings()
    target = _target_config()

    derived = build_settings_for_target(base, target, chunk_size=500, chunk_overlap=50)

    assert get_active_embedding_config(derived).collection_name == target.collection_name


# --- build_reindex_target: happy path ---------------------------------------------------------


async def test_build_uses_the_explicit_target_provider_and_model(tmp_path: Path, monkeypatch) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world " * 50, encoding="utf-8")

    embedding_provider = _CapturingEmbeddingProvider()
    vector_store = _FakeVectorStore()
    captured_settings: list[Settings] = []
    monkeypatch.setattr(
        reindex_service_module,
        "get_embedding_provider",
        lambda settings: (captured_settings.append(settings), embedding_provider)[1],
    )
    monkeypatch.setattr(reindex_service_module, "get_vector_store", lambda settings: vector_store)

    document = _document(storage_provider="local", storage_key=file_path.name)
    session = _FakeSession()
    base = _base_settings()
    target = _target_config()

    result = await build_reindex_target(
        document,
        session,
        base,
        LocalFileStorage(root=tmp_path),
        target,
        target_chunk_size=500,
        target_chunk_overlap=50,
    )

    assert result.outcome == ReindexBuildOutcome.BUILT
    assert result.target_collection_name == target.collection_name
    used_settings = captured_settings[0]
    assert used_settings.embedding_provider == target.provider  # test 1
    assert used_settings.resolved_embedding_model == target.model  # test 2
    assert used_settings.embedding_model == used_settings.ollama_embedding_model == target.model  # test 3
    assert used_settings.vector_size == target.dimension  # test 4
    assert used_settings.embedding_version == target.embedding_version  # test 5
    assert used_settings.chunking_version == target.chunking_version  # test 6
    assert used_settings.chunk_size == 500  # test 7
    assert used_settings.chunk_overlap == 50  # test 8


async def test_successful_build_writes_vectors_only_to_the_target_collection(
    tmp_path: Path, monkeypatch
) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world " * 50, encoding="utf-8")

    embedding_provider = _CapturingEmbeddingProvider()
    vector_store = _FakeVectorStore()
    monkeypatch.setattr(reindex_service_module, "get_embedding_provider", lambda settings: embedding_provider)
    monkeypatch.setattr(reindex_service_module, "get_vector_store", lambda settings: vector_store)

    document = _document(storage_provider="local", storage_key=file_path.name)
    session = _FakeSession()
    target = _target_config()

    await build_reindex_target(
        document,
        session,
        _base_settings(),
        LocalFileStorage(root=tmp_path),
        target,
        target_chunk_size=500,
        target_chunk_overlap=50,
    )

    assert set(vector_store.upserted.keys()) == {target.collection_name}
    assert vector_store.upserted[target.collection_name]


async def test_successful_build_never_touches_document_serving_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world " * 50, encoding="utf-8")

    embedding_provider = _CapturingEmbeddingProvider()
    vector_store = _FakeVectorStore()
    monkeypatch.setattr(reindex_service_module, "get_embedding_provider", lambda settings: embedding_provider)
    monkeypatch.setattr(reindex_service_module, "get_vector_store", lambda settings: vector_store)

    fixed_indexed_at = datetime(2026, 1, 1, tzinfo=UTC)
    document = _document(
        storage_provider="local",
        storage_key=file_path.name,
        collection_name="serving-collection-a",
        embedding_provider="ollama",
        embedding_model="serving-model",
        embedding_dimension=3,
        embedding_version="v-serving",
        chunking_version="v-serving",
        indexed_at=fixed_indexed_at,
    )
    session = _FakeSession()
    target = _target_config()

    result = await build_reindex_target(
        document,
        session,
        _base_settings(),
        LocalFileStorage(root=tmp_path),
        target,
        target_chunk_size=500,
        target_chunk_overlap=50,
    )

    assert result.outcome == ReindexBuildOutcome.BUILT
    assert document.collection_name == "serving-collection-a"  # test 14
    assert document.embedding_provider == "ollama"  # test 15
    assert document.embedding_model == "serving-model"  # test 15
    assert document.embedding_dimension == 3  # test 15
    assert document.embedding_version == "v-serving"  # test 15
    assert document.chunking_version == "v-serving"  # test 15
    assert document.indexed_at == fixed_indexed_at  # test 16
    assert vector_store.deleted == []  # test 17 — nothing deleted from the serving collection
    assert session.added == []  # test 18 — no VectorCleanupJob created


async def test_successful_build_does_not_call_mark_document_indexed(tmp_path: Path, monkeypatch) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world " * 50, encoding="utf-8")

    embedding_provider = _CapturingEmbeddingProvider()
    vector_store = _FakeVectorStore()
    monkeypatch.setattr(reindex_service_module, "get_embedding_provider", lambda settings: embedding_provider)
    monkeypatch.setattr(reindex_service_module, "get_vector_store", lambda settings: vector_store)

    called = []
    monkeypatch.setattr(
        reindex_service_module, "mark_document_indexed", lambda *a, **k: called.append((a, k))
    )

    document = _document(storage_provider="local", storage_key=file_path.name)
    session = _FakeSession()

    await build_reindex_target(
        document,
        session,
        _base_settings(),
        LocalFileStorage(root=tmp_path),
        _target_config(),
        target_chunk_size=500,
        target_chunk_overlap=50,
    )

    assert called == []  # test 13


async def test_build_is_idempotent_for_deterministic_point_ids(tmp_path: Path, monkeypatch) -> None:
    """Re-running the build against the same target must produce identical point IDs."""
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world", encoding="utf-8")

    embedding_provider = _CapturingEmbeddingProvider()
    vector_store = _FakeVectorStore()
    monkeypatch.setattr(reindex_service_module, "get_embedding_provider", lambda settings: embedding_provider)
    monkeypatch.setattr(reindex_service_module, "get_vector_store", lambda settings: vector_store)

    document = _document(storage_provider="local", storage_key=file_path.name)
    session = _FakeSession()
    target = _target_config()
    file_storage = LocalFileStorage(root=tmp_path)

    await build_reindex_target(
        document, session, _base_settings(), file_storage, target,
        target_chunk_size=500, target_chunk_overlap=50,
    )
    first_ids = [point.id for point in vector_store.upserted[target.collection_name]]

    await build_reindex_target(
        document, session, _base_settings(), file_storage, target,
        target_chunk_size=500, target_chunk_overlap=50,
    )
    all_points = vector_store.upserted[target.collection_name]
    second_ids = [point.id for point in all_points[len(first_ids) :]]

    assert first_ids == second_ids  # test 24


async def test_build_does_not_touch_an_unrelated_documents_vectors(tmp_path: Path, monkeypatch) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world " * 50, encoding="utf-8")

    embedding_provider = _CapturingEmbeddingProvider()
    vector_store = _FakeVectorStore()
    monkeypatch.setattr(reindex_service_module, "get_embedding_provider", lambda settings: embedding_provider)
    monkeypatch.setattr(reindex_service_module, "get_vector_store", lambda settings: vector_store)

    document = _document(storage_provider="local", storage_key=file_path.name)
    unrelated_document_id = str(uuid.uuid4())
    session = _FakeSession()
    target = _target_config()

    await build_reindex_target(
        document,
        session,
        _base_settings(),
        LocalFileStorage(root=tmp_path),
        target,
        target_chunk_size=500,
        target_chunk_overlap=50,
    )

    points = vector_store.upserted[target.collection_name]
    assert all(point.document_id == document.id for point in points)  # test 25
    assert all(point.document_id != unrelated_document_id for point in points)


# --- build_reindex_target: target/settings mismatch guard --------------------------------------


async def test_target_settings_mismatch_fails_before_any_write(tmp_path: Path, monkeypatch) -> None:
    embedding_provider = _CapturingEmbeddingProvider()
    vector_store = _FakeVectorStore()
    monkeypatch.setattr(reindex_service_module, "get_embedding_provider", lambda settings: embedding_provider)
    monkeypatch.setattr(reindex_service_module, "get_vector_store", lambda settings: vector_store)

    different_config = _target_config(provider="ollama", model="a-different-model")
    monkeypatch.setattr(
        reindex_service_module, "get_active_embedding_config", lambda settings: different_config
    )

    document = _document(storage_provider="local", storage_key="nonexistent.txt")
    session = _FakeSession()
    target = _target_config()

    with pytest.raises(TargetConfigurationMismatchError):
        await build_reindex_target(
            document,
            session,
            _base_settings(),
            LocalFileStorage(root=tmp_path),
            target,
            target_chunk_size=500,
            target_chunk_overlap=50,
        )

    assert embedding_provider.embed_calls == []  # test 11 — never reached embedding
    assert vector_store.upserted == {}  # test 11 — never reached the vector store


# --- build_reindex_target: failure paths leave document metadata untouched ----------------------


def _serving_document(**overrides: object) -> Document:
    fixed_indexed_at = datetime(2026, 1, 1, tzinfo=UTC)
    fields: dict[str, object] = dict(
        storage_provider="local",
        collection_name="serving-collection-a",
        embedding_provider="ollama",
        embedding_model="serving-model",
        embedding_dimension=3,
        embedding_version="v-serving",
        chunking_version="v-serving",
        indexed_at=fixed_indexed_at,
    )
    fields.update(overrides)
    return _document(**fields)


def _assert_metadata_untouched(document: Document) -> None:
    assert document.collection_name == "serving-collection-a"
    assert document.embedding_provider == "ollama"
    assert document.embedding_model == "serving-model"
    assert document.embedding_dimension == 3
    assert document.embedding_version == "v-serving"
    assert document.chunking_version == "v-serving"
    assert document.indexed_at == datetime(2026, 1, 1, tzinfo=UTC)


async def test_extraction_failure_leaves_document_metadata_unchanged(tmp_path: Path, monkeypatch) -> None:
    vector_store = _FakeVectorStore()
    monkeypatch.setattr(reindex_service_module, "get_vector_store", lambda settings: vector_store)

    document = _serving_document(storage_key="does-not-exist.txt")
    session = _FakeSession()

    with pytest.raises(DocumentTextExtractionError):
        await build_reindex_target(
            document,
            session,
            _base_settings(),
            LocalFileStorage(root=tmp_path),
            _target_config(),
            target_chunk_size=500,
            target_chunk_overlap=50,
        )

    _assert_metadata_untouched(document)  # test 19


async def test_chunking_failure_leaves_document_metadata_unchanged(tmp_path: Path, monkeypatch) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world " * 50, encoding="utf-8")

    vector_store = _FakeVectorStore()
    monkeypatch.setattr(reindex_service_module, "get_vector_store", lambda settings: vector_store)
    monkeypatch.setattr(
        DocumentChunker, "chunk", lambda self, extracted: (_ for _ in ()).throw(RuntimeError("chunk failure"))
    )

    document = _serving_document(storage_key=file_path.name)
    session = _FakeSession()

    with pytest.raises(RuntimeError, match="chunk failure"):
        await build_reindex_target(
            document,
            session,
            _base_settings(),
            LocalFileStorage(root=tmp_path),
            _target_config(),
            target_chunk_size=500,
            target_chunk_overlap=50,
        )

    _assert_metadata_untouched(document)  # test 20


async def test_embedding_failure_leaves_document_metadata_unchanged(tmp_path: Path, monkeypatch) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world " * 50, encoding="utf-8")

    vector_store = _FakeVectorStore()
    monkeypatch.setattr(
        reindex_service_module, "get_embedding_provider", lambda settings: _FailingEmbeddingProvider()
    )
    monkeypatch.setattr(reindex_service_module, "get_vector_store", lambda settings: vector_store)

    document = _serving_document(storage_key=file_path.name)
    session = _FakeSession()

    with pytest.raises(RuntimeError, match="embedding unavailable"):
        await build_reindex_target(
            document,
            session,
            _base_settings(),
            LocalFileStorage(root=tmp_path),
            _target_config(),
            target_chunk_size=500,
            target_chunk_overlap=50,
        )

    _assert_metadata_untouched(document)  # test 21
    assert vector_store.upserted == {}


async def test_validation_failure_leaves_document_metadata_unchanged(tmp_path: Path, monkeypatch) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world " * 50, encoding="utf-8")

    wrong_dimension_provider = _CapturingEmbeddingProvider(vector=[0.1, 0.2])  # 2-dim, target expects 3
    vector_store = _FakeVectorStore()
    monkeypatch.setattr(
        reindex_service_module, "get_embedding_provider", lambda settings: wrong_dimension_provider
    )
    monkeypatch.setattr(reindex_service_module, "get_vector_store", lambda settings: vector_store)

    document = _serving_document(storage_key=file_path.name)
    session = _FakeSession()

    with pytest.raises(EmbeddingDimensionMismatchError):
        await build_reindex_target(
            document,
            session,
            _base_settings(),
            LocalFileStorage(root=tmp_path),
            _target_config(dimension=3),
            target_chunk_size=500,
            target_chunk_overlap=50,
        )

    _assert_metadata_untouched(document)  # test 22
    assert vector_store.upserted == {}


async def test_qdrant_write_failure_leaves_document_metadata_unchanged(tmp_path: Path, monkeypatch) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world " * 50, encoding="utf-8")

    embedding_provider = _CapturingEmbeddingProvider()
    vector_store = _FakeVectorStore()
    vector_store.raise_on_upsert = True
    monkeypatch.setattr(reindex_service_module, "get_embedding_provider", lambda settings: embedding_provider)
    monkeypatch.setattr(reindex_service_module, "get_vector_store", lambda settings: vector_store)

    document = _serving_document(storage_key=file_path.name)
    session = _FakeSession()

    with pytest.raises(RuntimeError, match="Qdrant unreachable"):
        await build_reindex_target(
            document,
            session,
            _base_settings(),
            LocalFileStorage(root=tmp_path),
            _target_config(),
            target_chunk_size=500,
            target_chunk_overlap=50,
        )

    _assert_metadata_untouched(document)  # test 23
    assert vector_store.upserted == {}


async def test_build_service_uses_existing_provider_factory() -> None:
    """build_reindex_target must resolve providers via the existing factory, never construct clients."""
    source = inspect.getsource(reindex_service_module)
    assert "from app.rag.providers.provider_factory import get_embedding_provider, get_vector_store" in source


# --- activate_reindexed_document ----------------------------------------------------------------


async def test_activation_updates_document_metadata_to_target() -> None:
    document = _document(collection_name="serving-collection-a")
    session = _FakeSession()
    target = _target_config()

    result = await activate_reindexed_document(document, session, target)

    assert result.activated is True
    assert document.collection_name == target.collection_name
    assert document.embedding_provider == target.provider
    assert document.embedding_model == target.model
    assert document.embedding_dimension == target.dimension
    assert document.embedding_version == target.embedding_version
    assert document.chunking_version == target.chunking_version
    assert document.indexed_at is not None


async def test_activation_captures_previous_collection_before_mutation() -> None:
    document = _document(collection_name="serving-collection-a")
    session = _FakeSession()
    target = _target_config()

    result = await activate_reindexed_document(document, session, target)

    assert result.cleanup_job is not None
    assert result.cleanup_job.collection_name == "serving-collection-a"
    assert result.cleanup_job.collection_name != target.collection_name  # never the new collection


async def test_activation_creates_cleanup_obligation_for_previous_collection() -> None:
    document = _document(collection_name="serving-collection-a")
    session = _FakeSession()
    target = _target_config()

    result = await activate_reindexed_document(document, session, target)

    assert result.cleanup_job in session.added
    assert isinstance(result.cleanup_job, VectorCleanupJob)
    assert result.cleanup_job.status == VectorCleanupStatus.PENDING
    assert result.cleanup_job.attempts == 0
    assert result.cleanup_job.last_error is None


async def test_metadata_switch_precedes_the_single_commit() -> None:
    document = _document(collection_name="serving-collection-a")
    session = _FakeSession()
    session.watch(document)
    target = _target_config()

    await activate_reindexed_document(document, session, target)

    assert session.commit_snapshots == [target.collection_name]
    assert session.commit_count == 1


async def test_commit_failure_rolls_back_both_metadata_and_cleanup_obligation() -> None:
    document = _document(collection_name="serving-collection-a")
    session = _FakeSession(fail_commit=True)
    target = _target_config()

    with pytest.raises(RuntimeError, match="db unavailable"):
        await activate_reindexed_document(document, session, target)

    assert session.rollback_count == 1
    assert document in session.expired


async def test_activation_never_deletes_qdrant_vectors() -> None:
    signature = inspect.signature(activate_reindexed_document)
    assert not any("vector_store" in name.lower() for name in signature.parameters)

    source = inspect.getsource(activate_reindexed_document)
    assert "delete_by_document_id" not in source


async def test_no_cleanup_job_when_there_is_no_previous_collection() -> None:
    document = _document(collection_name=None)
    session = _FakeSession()
    target = _target_config()

    result = await activate_reindexed_document(document, session, target)

    assert result.activated is True
    assert result.cleanup_job is None
    assert not any(isinstance(row, VectorCleanupJob) for row in session.added)


async def test_no_cleanup_job_when_previous_and_target_collections_are_identical() -> None:
    target = _target_config()
    document = _document(collection_name=target.collection_name)
    session = _FakeSession()

    result = await activate_reindexed_document(document, session, target)

    assert result.activated is False
    assert result.cleanup_job is None
    assert session.added == []
    assert session.commit_count == 0


async def test_repeated_activation_is_idempotent() -> None:
    document = _document(collection_name="serving-collection-a")
    session = _FakeSession()
    target = _target_config()

    first = await activate_reindexed_document(document, session, target)
    assert first.activated is True
    added_after_first = list(session.added)

    second = await activate_reindexed_document(document, session, target)

    assert second.activated is False
    assert second.cleanup_job is None
    assert session.added == added_after_first  # no additional row added
