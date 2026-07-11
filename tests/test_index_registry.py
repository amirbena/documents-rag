"""Tests for app/services/index_registry.py — collection safety and staleness, no real database
(a minimal fake session/vector-store double is enough for these unit-level checks)."""

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from app.models.document import Document
from app.models.index_collection import IndexCollection
from app.models.vector_cleanup_job import VectorCleanupJob, VectorCleanupStatus
from app.rag.embedding_config import EmbeddingIndexConfig
from app.services.index_registry import (
    IncompatibleIndexConfigurationError,
    create_cleanup_job,
    delete_document_vectors,
    ensure_active_collection,
    get_pending_cleanup_jobs,
    is_document_stale,
    mark_document_indexed,
    retire_collection,
    retry_cleanup_job,
)


class _FakeVectorStore:
    def __init__(
        self, existing_dimension: int | None = None, fail_delete_for: set[str] | None = None
    ) -> None:
        self.existing_dimension = existing_dimension
        self.created_collections: list[tuple[str, int]] = []
        self.deleted: list[tuple[str, str]] = []
        self._fail_delete_for = fail_delete_for or set()

    async def get_collection_vector_size(self, collection_name: str) -> int | None:
        return self.existing_dimension

    async def create_collection_if_not_exists(self, collection_name: str, vector_size: int) -> None:
        self.created_collections.append((collection_name, vector_size))

    async def delete_by_document_id(self, collection_name: str, document_id: str) -> None:
        if collection_name in self._fail_delete_for:
            raise RuntimeError(f"could not delete from {collection_name}")
        self.deleted.append((collection_name, document_id))


class _FakeSession:
    def __init__(self) -> None:
        self._index_collections: dict[str, IndexCollection] = {}
        self._cleanup_jobs: dict[str, VectorCleanupJob] = {}
        self.commit_count = 0

    def add(self, instance: object) -> None:
        if isinstance(instance, IndexCollection):
            self._index_collections[instance.collection_name] = instance
        elif isinstance(instance, VectorCleanupJob):
            self._cleanup_jobs[instance.id] = instance

    async def get(self, model: type, instance_id: str) -> object | None:
        if model is IndexCollection:
            return self._index_collections.get(instance_id)
        return None

    async def execute(self, stmt: Any):
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


# --- VectorCleanupJob tracking/retry --------------------------------------------------------------


async def test_create_cleanup_job_without_error_is_pending() -> None:
    """create_cleanup_job() with no error records a fresh PENDING job with zero attempts."""
    session = _FakeSession()
    document = _document()

    job = await create_cleanup_job(session, document.id, "old-collection")

    assert job.status == VectorCleanupStatus.PENDING
    assert job.attempts == 0
    assert job.last_error is None
    assert session.commit_count == 1


async def test_create_cleanup_job_with_error_is_failed_with_one_attempt() -> None:
    """create_cleanup_job() with an error records it as already FAILED, one attempt logged."""
    session = _FakeSession()
    document = _document()

    job = await create_cleanup_job(session, document.id, "old-collection", error="boom")

    assert job.status == VectorCleanupStatus.FAILED
    assert job.attempts == 1
    assert job.last_error == "boom"


async def test_get_pending_cleanup_jobs_excludes_completed() -> None:
    """A COMPLETED job must never be returned by get_pending_cleanup_jobs()."""
    session = _FakeSession()
    document = _document()
    pending_job = await create_cleanup_job(session, document.id, "collection-a")
    completed_job = await create_cleanup_job(session, document.id, "collection-b")
    completed_job.status = VectorCleanupStatus.COMPLETED

    jobs = await get_pending_cleanup_jobs(session, document_id=document.id)

    assert [job.collection_name for job in jobs] == [pending_job.collection_name]


async def test_get_pending_cleanup_jobs_returns_multiple_historical_collections() -> None:
    """Two failed cleanups for the same document (different collections) never overwrite each other."""
    session = _FakeSession()
    document = _document()
    await create_cleanup_job(session, document.id, "collection-a", error="first failure")
    await create_cleanup_job(session, document.id, "collection-b", error="second failure")

    jobs = await get_pending_cleanup_jobs(session, document_id=document.id)

    assert {job.collection_name for job in jobs} == {"collection-a", "collection-b"}


async def test_get_pending_cleanup_jobs_scopes_to_document_id() -> None:
    """A cleanup job for a different document must not leak into another document's results."""
    session = _FakeSession()
    document_a = _document()
    document_b = _document()
    await create_cleanup_job(session, document_a.id, "collection-a", error="failure")
    await create_cleanup_job(session, document_b.id, "collection-b", error="failure")

    jobs = await get_pending_cleanup_jobs(session, document_id=document_a.id)

    assert [job.document_id for job in jobs] == [document_a.id]


async def test_retry_cleanup_job_marks_completed_on_success() -> None:
    """A successful retry marks the job COMPLETED with completed_at set."""
    session = _FakeSession()
    document = _document()
    job = await create_cleanup_job(session, document.id, "old-collection", error="first failure")
    vector_store = _FakeVectorStore()

    succeeded = await retry_cleanup_job(session, vector_store, job)

    assert succeeded is True
    assert job.status == VectorCleanupStatus.COMPLETED
    assert job.completed_at is not None
    assert job.last_error is None
    assert job.attempts == 2
    assert vector_store.deleted == [("old-collection", document.id)]


async def test_retry_cleanup_job_stays_failed_on_repeated_failure() -> None:
    """A repeated failure increments attempts and records the latest error, stays FAILED."""
    session = _FakeSession()
    document = _document()
    job = await create_cleanup_job(session, document.id, "old-collection", error="first failure")
    vector_store = _FakeVectorStore(fail_delete_for={"old-collection"})

    succeeded = await retry_cleanup_job(session, vector_store, job)

    assert succeeded is False
    assert job.status == VectorCleanupStatus.FAILED
    assert job.attempts == 2
    assert "old-collection" in (job.last_error or "")


async def test_retry_cleanup_job_is_retried_even_when_document_is_no_longer_stale() -> None:
    """Cleanup retry does not depend on is_document_stale() — it is tracked independently."""
    config = _config()
    document = _document(
        collection_name=config.collection_name,
        embedding_provider=config.provider,
        embedding_model=config.model,
        embedding_dimension=config.dimension,
        embedding_version=config.embedding_version,
        chunking_version=config.chunking_version,
    )
    assert is_document_stale(document, config) is False

    session = _FakeSession()
    job = await create_cleanup_job(session, document.id, "old-collection", error="first failure")
    vector_store = _FakeVectorStore()

    succeeded = await retry_cleanup_job(session, vector_store, job)

    assert succeeded is True


async def test_delete_document_vectors_also_cleans_pending_historical_collections() -> None:
    """Deleting a document must clean its current collection AND every pending legacy collection."""
    session = _FakeSession()
    document = _document(collection_name="current-collection")
    await create_cleanup_job(session, document.id, "legacy-collection-1", error="failure")
    await create_cleanup_job(session, document.id, "legacy-collection-2", error="failure")
    vector_store = _FakeVectorStore()

    await delete_document_vectors(document, vector_store, session)

    assert set(vector_store.deleted) == {
        ("current-collection", document.id),
        ("legacy-collection-1", document.id),
        ("legacy-collection-2", document.id),
    }


async def test_delete_document_vectors_without_session_only_cleans_current_collection() -> None:
    """Backward-compat: omitting `session` still cleans the tracked collection, no cleanup lookup."""
    document = _document(collection_name="current-collection")
    vector_store = _FakeVectorStore()

    await delete_document_vectors(document, vector_store)

    assert vector_store.deleted == [("current-collection", document.id)]
