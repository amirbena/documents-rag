"""Unit tests for app/services/indexing/cleanup_job_service.py — VectorCleanupJob persistence and
retry, no real database.
"""

from app.models.vector_cleanup_job import VectorCleanupStatus
from app.services.indexing.cleanup_job_service import (
    create_cleanup_job,
    get_pending_cleanup_jobs,
    retry_cleanup_job,
)
from app.services.indexing.collection_registry import is_document_stale
from tests.support.indexing.builders import build_document, build_embedding_config
from tests.support.indexing.fakes import FakeIndexSession, FakeVectorStore


async def test_create_cleanup_job_without_error_is_pending() -> None:
    """create_cleanup_job() with no error records a fresh PENDING job with zero attempts."""
    session = FakeIndexSession()
    document = build_document()

    job = await create_cleanup_job(session, document.id, "old-collection")

    assert job.status == VectorCleanupStatus.PENDING
    assert job.attempts == 0
    assert job.last_error is None
    assert session.commit_count == 1


async def test_create_cleanup_job_with_error_is_failed_with_one_attempt() -> None:
    """create_cleanup_job() with an error records it as already FAILED, one attempt logged."""
    session = FakeIndexSession()
    document = build_document()

    job = await create_cleanup_job(session, document.id, "old-collection", error="boom")

    assert job.status == VectorCleanupStatus.FAILED
    assert job.attempts == 1
    assert job.last_error == "boom"


async def test_get_pending_cleanup_jobs_excludes_completed() -> None:
    """A COMPLETED job must never be returned by get_pending_cleanup_jobs()."""
    session = FakeIndexSession()
    document = build_document()
    pending_job = await create_cleanup_job(session, document.id, "collection-a")
    completed_job = await create_cleanup_job(session, document.id, "collection-b")
    completed_job.status = VectorCleanupStatus.COMPLETED

    jobs = await get_pending_cleanup_jobs(session, document_id=document.id)

    assert [job.collection_name for job in jobs] == [pending_job.collection_name]


async def test_get_pending_cleanup_jobs_returns_multiple_historical_collections() -> None:
    """Two failed cleanups for the same document (different collections) never overwrite each other."""
    session = FakeIndexSession()
    document = build_document()
    await create_cleanup_job(session, document.id, "collection-a", error="first failure")
    await create_cleanup_job(session, document.id, "collection-b", error="second failure")

    jobs = await get_pending_cleanup_jobs(session, document_id=document.id)

    assert {job.collection_name for job in jobs} == {"collection-a", "collection-b"}


async def test_get_pending_cleanup_jobs_scopes_to_document_id() -> None:
    """A cleanup job for a different document must not leak into another document's results."""
    session = FakeIndexSession()
    document_a = build_document()
    document_b = build_document()
    await create_cleanup_job(session, document_a.id, "collection-a", error="failure")
    await create_cleanup_job(session, document_b.id, "collection-b", error="failure")

    jobs = await get_pending_cleanup_jobs(session, document_id=document_a.id)

    assert [job.document_id for job in jobs] == [document_a.id]


async def test_retry_cleanup_job_marks_completed_on_success() -> None:
    """A successful retry marks the job COMPLETED with completed_at set."""
    session = FakeIndexSession()
    document = build_document()
    job = await create_cleanup_job(session, document.id, "old-collection", error="first failure")
    vector_store = FakeVectorStore()

    succeeded = await retry_cleanup_job(session, vector_store, job)

    assert succeeded is True
    assert job.status == VectorCleanupStatus.COMPLETED
    assert job.completed_at is not None
    assert job.last_error is None
    assert job.attempts == 2
    assert vector_store.deleted == [("old-collection", document.id)]


async def test_retry_cleanup_job_stays_failed_on_repeated_failure() -> None:
    """A repeated failure increments attempts and records the latest error, stays FAILED."""
    session = FakeIndexSession()
    document = build_document()
    job = await create_cleanup_job(session, document.id, "old-collection", error="first failure")
    vector_store = FakeVectorStore(fail_delete_for={"old-collection"})

    succeeded = await retry_cleanup_job(session, vector_store, job)

    assert succeeded is False
    assert job.status == VectorCleanupStatus.FAILED
    assert job.attempts == 2
    assert "old-collection" in (job.last_error or "")


async def test_retry_cleanup_job_is_retried_even_when_document_is_no_longer_stale() -> None:
    """Cleanup retry does not depend on is_document_stale() — it is tracked independently."""
    config = build_embedding_config()
    document = build_document(
        collection_name=config.collection_name,
        embedding_provider=config.provider,
        embedding_model=config.model,
        embedding_dimension=config.dimension,
        embedding_version=config.embedding_version,
        chunking_version=config.chunking_version,
    )
    assert is_document_stale(document, config) is False

    session = FakeIndexSession()
    job = await create_cleanup_job(session, document.id, "old-collection", error="first failure")
    vector_store = FakeVectorStore()

    succeeded = await retry_cleanup_job(session, vector_store, job)

    assert succeeded is True
