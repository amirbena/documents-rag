"""Unit tests for app/services/indexing/vector_deletion_service.py — partial vs. full vector
cleanup, no real database/Qdrant.
"""

import inspect
import uuid

from app.models.reindex_job import ReindexJobStatus
from app.services.indexing.cleanup_job_service import create_cleanup_job
from app.services.indexing.vector_deletion_service import (
    delete_all_tracked_document_vectors,
    delete_current_document_vectors,
)
from tests.support.indexing.builders import build_document, build_reindex_job
from tests.support.indexing.fakes import FakeIndexSession, FakeVectorStore


async def test_delete_current_document_vectors_targets_only_the_tracked_collection() -> None:
    """delete_current_document_vectors() must delete from the document's own collection_name only."""
    vector_store = FakeVectorStore()
    document = build_document(collection_name="documents__ollama__m__ev1__cv1__d768")

    await delete_current_document_vectors(document, vector_store)

    assert vector_store.deleted == [("documents__ollama__m__ev1__cv1__d768", document.id)]


async def test_delete_current_document_vectors_is_a_noop_for_a_never_indexed_document() -> None:
    """A document with no collection_name has nothing to delete — no vector-store call at all."""

    class _AssertNoDeleteVectorStore(FakeVectorStore):
        async def delete_by_document_id(self, collection_name: str, document_id: str) -> None:
            raise AssertionError("must not attempt to delete a never-indexed document's vectors")

    document = build_document(collection_name=None)

    await delete_current_document_vectors(document, _AssertNoDeleteVectorStore())


async def test_delete_current_document_vectors_never_consults_cleanup_jobs() -> None:
    """The explicitly-partial function must never touch VectorCleanupJob bookkeeping at all."""
    document = build_document(collection_name="current-collection")
    vector_store = FakeVectorStore()

    await delete_current_document_vectors(document, vector_store)

    # No `session` parameter exists on this function at all — this is the type-level guarantee
    # that it cannot accidentally perform historical cleanup.
    assert "session" not in inspect.signature(delete_current_document_vectors).parameters


async def test_delete_all_tracked_document_vectors_cleans_current_and_historical_collections() -> None:
    """Full deletion must clean the current collection AND every pending legacy collection."""
    session = FakeIndexSession()
    document = build_document(collection_name="current-collection")
    await create_cleanup_job(session, document.id, "legacy-collection-1", error="failure")
    await create_cleanup_job(session, document.id, "legacy-collection-2", error="failure")
    vector_store = FakeVectorStore()

    await delete_all_tracked_document_vectors(document, vector_store, session)

    assert set(vector_store.deleted) == {
        ("current-collection", document.id),
        ("legacy-collection-1", document.id),
        ("legacy-collection-2", document.id),
    }


async def test_delete_all_tracked_document_vectors_requires_a_session() -> None:
    """Full deletion has no `session=None` escape hatch — omitting it must fail at the call site,
    not silently degrade to partial cleanup.
    """
    signature = inspect.signature(delete_all_tracked_document_vectors)
    assert signature.parameters["session"].default is inspect.Parameter.empty


async def test_delete_all_tracked_document_vectors_is_idempotent() -> None:
    """Calling full deletion twice must not error, and must not double-delete or duplicate calls."""
    session = FakeIndexSession()
    document = build_document(collection_name="current-collection")
    await create_cleanup_job(session, document.id, "legacy-collection-1", error="failure")
    vector_store = FakeVectorStore()

    await delete_all_tracked_document_vectors(document, vector_store, session)
    await delete_all_tracked_document_vectors(document, vector_store, session)

    # Each call performs its own delete-by-filter; repeating it is a harmless no-op against
    # already-empty collections in real Qdrant — here we only assert it doesn't raise and the
    # same two collections are targeted both times.
    assert vector_store.deleted.count(("current-collection", document.id)) == 2
    assert vector_store.deleted.count(("legacy-collection-1", document.id)) == 2


async def test_delete_all_tracked_document_vectors_attempts_every_collection_despite_a_failure() -> None:
    """A failure on one historical collection must not stop attempts against another."""
    session = FakeIndexSession()
    document = build_document(collection_name="current-collection")
    await create_cleanup_job(session, document.id, "legacy-collection-1", error="failure")
    await create_cleanup_job(session, document.id, "legacy-collection-2", error="failure")
    vector_store = FakeVectorStore(fail_delete_for={"legacy-collection-1"})

    result = await delete_all_tracked_document_vectors(document, vector_store, session)

    # Both historical collections were attempted, not just the one before the failure.
    attempted = {r.collection_name for r in result.collection_results}
    assert attempted == {"current-collection", "legacy-collection-1", "legacy-collection-2"}
    assert ("current-collection", document.id) in vector_store.deleted
    assert ("legacy-collection-2", document.id) in vector_store.deleted
    assert result.fully_deleted is False

    by_name = {r.collection_name: r for r in result.collection_results}
    assert by_name["current-collection"].succeeded is True
    assert by_name["legacy-collection-1"].succeeded is False
    assert by_name["legacy-collection-1"].error is not None
    assert by_name["legacy-collection-2"].succeeded is True


async def test_delete_all_tracked_document_vectors_still_attempts_active_collection_after_it_fails() -> None:
    """A failure on the active collection must not skip or abort the historical collections."""
    session = FakeIndexSession()
    document = build_document(collection_name="current-collection")
    await create_cleanup_job(session, document.id, "legacy-collection-1", error="failure")
    vector_store = FakeVectorStore(fail_delete_for={"current-collection"})

    result = await delete_all_tracked_document_vectors(document, vector_store, session)

    assert ("legacy-collection-1", document.id) in vector_store.deleted
    by_name = {r.collection_name: r for r in result.collection_results}
    assert by_name["current-collection"].succeeded is False
    assert by_name["legacy-collection-1"].succeeded is True
    assert result.fully_deleted is False


async def test_delete_all_tracked_document_vectors_dedupes_duplicate_historical_collection_names() -> None:
    """Two VectorCleanupJob rows for the same historical collection must be attempted once."""
    session = FakeIndexSession()
    document = build_document(collection_name="current-collection")
    await create_cleanup_job(session, document.id, "legacy-collection-1", error="first failure")
    await create_cleanup_job(session, document.id, "legacy-collection-1", error="second failure")
    vector_store = FakeVectorStore()

    result = await delete_all_tracked_document_vectors(document, vector_store, session)

    assert result.attempted_collections.count("legacy-collection-1") == 1
    assert vector_store.deleted.count(("legacy-collection-1", document.id)) == 1
    assert result.fully_deleted is True


async def test_delete_all_tracked_document_vectors_does_not_double_attempt_active_as_historical() -> None:
    """A cleanup job pointing at the document's own active collection is not a second attempt."""
    session = FakeIndexSession()
    document = build_document(collection_name="current-collection")
    await create_cleanup_job(session, document.id, "current-collection", error="stale failure")
    vector_store = FakeVectorStore()

    result = await delete_all_tracked_document_vectors(document, vector_store, session)

    assert result.attempted_collections == ("current-collection",)
    assert vector_store.deleted.count(("current-collection", document.id)) == 1
    assert result.fully_deleted is True


async def test_delete_all_tracked_document_vectors_retries_all_collections_after_partial_failure() -> None:
    """A repeated call after a partial failure must retry every tracked collection again, not
    just the ones that previously failed.
    """
    session = FakeIndexSession()
    document = build_document(collection_name="current-collection")
    await create_cleanup_job(session, document.id, "legacy-collection-1", error="failure")
    vector_store = FakeVectorStore(fail_delete_for={"legacy-collection-1"})

    first = await delete_all_tracked_document_vectors(document, vector_store, session)
    assert first.fully_deleted is False

    vector_store._fail_delete_for = set()  # simulate the transient failure clearing up
    second = await delete_all_tracked_document_vectors(document, vector_store, session)

    assert second.fully_deleted is True
    # The active collection (which already succeeded the first time) was attempted again too.
    assert vector_store.deleted.count(("current-collection", document.id)) == 2
    assert vector_store.deleted.count(("legacy-collection-1", document.id)) == 1


async def test_delete_all_tracked_document_vectors_includes_a_pending_cleanup_collection() -> None:
    """A PENDING (never-yet-attempted) cleanup job's collection must also be attempted."""
    session = FakeIndexSession()
    document = build_document(collection_name="current-collection")
    await create_cleanup_job(session, document.id, "legacy-collection-1")  # no error -> PENDING

    result = await delete_all_tracked_document_vectors(document, FakeVectorStore(), session)

    assert "legacy-collection-1" in result.attempted_collections


# --- completed re-index targets (Phase 2.8.6, subtask 3) -----------------------------------------


async def test_delete_all_tracked_document_vectors_includes_a_completed_reindex_target() -> None:
    """A COMPLETED re-index job's target collection must be attempted, even though
    Document.collection_name still points at the (different) serving collection."""
    session = FakeIndexSession()
    document = build_document(collection_name="serving-collection")
    job = build_reindex_job(document.id, ReindexJobStatus.COMPLETED, target_collection_name="target-b")
    session.reindex_jobs[job.id] = job
    vector_store = FakeVectorStore()

    result = await delete_all_tracked_document_vectors(document, vector_store, session)

    assert set(result.attempted_collections) == {"serving-collection", "target-b"}
    assert ("target-b", document.id) in vector_store.deleted


async def test_delete_all_tracked_document_vectors_excludes_a_pending_reindex_target() -> None:
    session = FakeIndexSession()
    document = build_document(collection_name="serving-collection")
    job = build_reindex_job(document.id, ReindexJobStatus.PENDING, target_collection_name="target-b")
    session.reindex_jobs[job.id] = job

    result = await delete_all_tracked_document_vectors(document, FakeVectorStore(), session)

    assert "target-b" not in result.attempted_collections


async def test_delete_all_tracked_document_vectors_excludes_a_processing_reindex_target() -> None:
    session = FakeIndexSession()
    document = build_document(collection_name="serving-collection")
    job = build_reindex_job(document.id, ReindexJobStatus.PROCESSING, target_collection_name="target-b")
    session.reindex_jobs[job.id] = job

    result = await delete_all_tracked_document_vectors(document, FakeVectorStore(), session)

    assert "target-b" not in result.attempted_collections


async def test_delete_all_tracked_document_vectors_excludes_a_failed_reindex_target() -> None:
    """A FAILED re-index job never proves a complete target vector set exists — must be excluded."""
    session = FakeIndexSession()
    document = build_document(collection_name="serving-collection")
    job = build_reindex_job(document.id, ReindexJobStatus.FAILED, target_collection_name="target-b")
    session.reindex_jobs[job.id] = job

    result = await delete_all_tracked_document_vectors(document, FakeVectorStore(), session)

    assert "target-b" not in result.attempted_collections


async def test_completed_reindex_target_equal_to_current_collection_is_deduplicated() -> None:
    """After activation, Document.collection_name may equal a completed job's own target — one attempt."""
    session = FakeIndexSession()
    document = build_document(collection_name="target-b")
    job = build_reindex_job(document.id, ReindexJobStatus.COMPLETED, target_collection_name="target-b")
    session.reindex_jobs[job.id] = job
    vector_store = FakeVectorStore()

    result = await delete_all_tracked_document_vectors(document, vector_store, session)

    assert result.attempted_collections == ("target-b",)
    assert vector_store.deleted.count(("target-b", document.id)) == 1


async def test_completed_reindex_target_equal_to_cleanup_job_collection_is_deduplicated() -> None:
    session = FakeIndexSession()
    document = build_document(collection_name="serving-collection")
    await create_cleanup_job(session, document.id, "legacy-a", error="boom")
    job = build_reindex_job(document.id, ReindexJobStatus.COMPLETED, target_collection_name="legacy-a")
    session.reindex_jobs[job.id] = job
    vector_store = FakeVectorStore()

    result = await delete_all_tracked_document_vectors(document, vector_store, session)

    assert result.attempted_collections.count("legacy-a") == 1
    assert vector_store.deleted.count(("legacy-a", document.id)) == 1


async def test_multiple_completed_reindex_targets_are_all_attempted() -> None:
    """A -> B completed, B -> C completed: both B and C must be attempted, not just the latest."""
    session = FakeIndexSession()
    document = build_document(collection_name="collection-a")
    job_b = build_reindex_job(document.id, ReindexJobStatus.COMPLETED, target_collection_name="collection-b")
    job_c = build_reindex_job(document.id, ReindexJobStatus.COMPLETED, target_collection_name="collection-c")
    session.reindex_jobs[job_b.id] = job_b
    session.reindex_jobs[job_c.id] = job_c
    vector_store = FakeVectorStore()

    result = await delete_all_tracked_document_vectors(document, vector_store, session)

    assert set(result.attempted_collections) == {"collection-a", "collection-b", "collection-c"}


async def test_duplicate_historical_completed_targets_result_in_one_attempt_per_collection() -> None:
    """Two separate COMPLETED jobs both targeting the same collection must be attempted once."""
    session = FakeIndexSession()
    document = build_document(collection_name="collection-a")
    job_1 = build_reindex_job(document.id, ReindexJobStatus.COMPLETED, target_collection_name="collection-b")
    job_2 = build_reindex_job(document.id, ReindexJobStatus.COMPLETED, target_collection_name="collection-b")
    session.reindex_jobs[job_1.id] = job_1
    session.reindex_jobs[job_2.id] = job_2
    vector_store = FakeVectorStore()

    result = await delete_all_tracked_document_vectors(document, vector_store, session)

    assert result.attempted_collections.count("collection-b") == 1
    assert vector_store.deleted.count(("collection-b", document.id)) == 1


async def test_completed_reindex_target_from_an_unrelated_document_is_excluded() -> None:
    session = FakeIndexSession()
    document = build_document(collection_name="serving-collection")
    other_document_id = str(uuid.uuid4())
    job = build_reindex_job(other_document_id, ReindexJobStatus.COMPLETED, target_collection_name="target-b")
    session.reindex_jobs[job.id] = job

    result = await delete_all_tracked_document_vectors(document, FakeVectorStore(), session)

    assert "target-b" not in result.attempted_collections


async def test_full_deletion_never_touches_object_storage() -> None:
    """Full vector deletion must never read/write object storage — it is a Qdrant-only operation."""
    source = inspect.getsource(delete_all_tracked_document_vectors)
    assert "FileStorage" not in source
    assert "file_storage" not in source


async def test_full_deletion_does_not_modify_document_metadata() -> None:
    session = FakeIndexSession()
    document = build_document(
        collection_name="serving-collection",
        embedding_provider="ollama",
        embedding_model="serving-model",
        embedding_version="v-serving",
        chunking_version="v-serving",
    )
    job = build_reindex_job(document.id, ReindexJobStatus.COMPLETED, target_collection_name="target-b")
    session.reindex_jobs[job.id] = job

    await delete_all_tracked_document_vectors(document, FakeVectorStore(), session)

    assert document.collection_name == "serving-collection"
    assert document.embedding_provider == "ollama"
    assert document.embedding_model == "serving-model"
    assert document.embedding_version == "v-serving"
    assert document.chunking_version == "v-serving"


async def test_full_deletion_does_not_change_reindex_job_status() -> None:
    session = FakeIndexSession()
    document = build_document(collection_name="serving-collection")
    job = build_reindex_job(document.id, ReindexJobStatus.COMPLETED, target_collection_name="target-b")
    session.reindex_jobs[job.id] = job

    await delete_all_tracked_document_vectors(document, FakeVectorStore(), session)

    assert session.reindex_jobs[job.id].status == ReindexJobStatus.COMPLETED


async def test_full_deletion_does_not_create_or_mutate_a_cleanup_job_for_a_completed_reindex_target() -> None:
    """Resolving a completed re-index target must be read-only — no VectorCleanupJob side effect."""
    session = FakeIndexSession()
    document = build_document(collection_name="serving-collection")
    job = build_reindex_job(document.id, ReindexJobStatus.COMPLETED, target_collection_name="target-b")
    session.reindex_jobs[job.id] = job

    await delete_all_tracked_document_vectors(document, FakeVectorStore(), session)

    assert session.cleanup_jobs == {}
    assert len(session.reindex_jobs) == 1  # only the one seeded above — nothing new was added
