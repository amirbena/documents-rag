"""VectorCleanupJob persistence: the durable, retryable record of a legacy collection's vectors
still needing deletion after a document was re-indexed into a new collection (see
`app.services.indexing.reindex_service`).

A cleanup failure is tracked here independently of whether the document itself is still
considered stale, so it stays discoverable and retryable even after the document is already
current under the active configuration, and multiple historical collections pending cleanup for
the same document are never conflated into a single record.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.vector_cleanup_job import VectorCleanupJob, VectorCleanupStatus
from app.rag.providers.vector_store import VectorStore


async def create_cleanup_job(
    session: AsyncSession, document_id: str, collection_name: str, error: str | None = None
) -> VectorCleanupJob:
    """Persist a new legacy-vector cleanup for `collection_name`, and commit it.

    Called after a re-index whose new collection/Document metadata already committed
    successfully, but whose immediately-previous-collection vector deletion failed (or was
    never attempted) — the re-index itself is not a failure, so this is tracked separately and
    retryably. Pass `error` (the first attempt's exception, stringified) to record the job as
    already FAILED with one attempt logged; omit it to record a fresh PENDING job.
    """
    job = VectorCleanupJob(
        id=str(uuid.uuid4()),
        document_id=document_id,
        collection_name=collection_name,
        status=VectorCleanupStatus.FAILED if error is not None else VectorCleanupStatus.PENDING,
        attempts=1 if error is not None else 0,
        last_error=error,
    )
    session.add(job)
    await session.commit()
    return job


async def get_pending_cleanup_jobs(
    session: AsyncSession, document_id: str | None = None
) -> list[VectorCleanupJob]:
    """Return every PENDING or FAILED VectorCleanupJob, optionally scoped to one document.

    Multiple rows for the same document are returned independently — a second failed cleanup
    for a different historical collection never overwrites or hides the first.
    """
    stmt = select(VectorCleanupJob).where(
        VectorCleanupJob.status.in_([VectorCleanupStatus.PENDING, VectorCleanupStatus.FAILED])
    )
    result = await session.execute(stmt)
    jobs = list(result.scalars().all())
    if document_id is not None:
        jobs = [job for job in jobs if job.document_id == document_id]
    return jobs


async def retry_cleanup_job(
    session: AsyncSession, vector_store: VectorStore, job: VectorCleanupJob
) -> bool:
    """Retry deleting `job`'s collection's vectors for its document; update and commit status.

    Retried regardless of whether the document itself is still considered stale — cleanup
    success/failure is tracked independently of `is_document_stale()`. Idempotent: retrying a
    cleanup whose vectors were already removed (e.g. a partially-succeeded prior attempt) is a
    harmless no-op delete-by-filter call. Returns True (and marks the job COMPLETED) on success,
    False (and marks it FAILED, incrementing `attempts`/recording `last_error`) on failure.
    """
    try:
        await vector_store.delete_by_document_id(job.collection_name, job.document_id)
    except Exception as exc:
        job.status = VectorCleanupStatus.FAILED
        job.attempts += 1
        job.last_error = str(exc)
        await session.commit()
        return False

    job.status = VectorCleanupStatus.COMPLETED
    job.attempts += 1
    job.last_error = None
    job.completed_at = datetime.now(UTC)
    await session.commit()
    return True


__all__ = [
    "create_cleanup_job",
    "get_pending_cleanup_jobs",
    "retry_cleanup_job",
]
