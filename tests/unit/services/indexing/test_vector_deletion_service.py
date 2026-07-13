"""Unit tests for app/services/indexing/vector_deletion_service.py — partial vs. full vector
cleanup, no real database/Qdrant.
"""

import inspect

from app.services.indexing.cleanup_job_service import create_cleanup_job
from app.services.indexing.vector_deletion_service import (
    delete_all_tracked_document_vectors,
    delete_current_document_vectors,
)
from tests.support.indexing.builders import build_document
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
