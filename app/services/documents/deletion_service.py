"""Request-scoped document deletion: scheduling, status reads, and public error sanitization.

`request_document_deletion()` is the service behind `DELETE /api/v1/documents/{id}`. It schedules
a deletion by creating a `PENDING` `DocumentDeletionJob` row — it never performs the actual
cross-system cleanup itself, so the HTTP request never blocks on unbounded external I/O. A
`PARTIALLY_FAILED` row is never reset — retrying always creates a brand-new `PENDING` row for the
same `document_id` (append-only, exactly like ingestion retry). Background execution
(`DocumentDeletionWorker`, the vector/storage cleanup orchestration) lives in the sibling
`deletion_worker` module — this module must never import from it (see that module's docstring for
the dependency-direction rule this package enforces).

## PostgreSQL remains authoritative; the Document row is never physically deleted

A successful deletion never removes the `Document` row, nor any `IngestionJob`/`VectorCleanupJob`/
`DocumentDeletionJob` history — see module docstrings on those models. Only the document's
external resources (Qdrant vectors, the stored object) are removed. `Document.collection_name`
etc. are also left untouched — the lifecycle status derivation in
`app/services/documents/query_service.py` uses the latest `DocumentDeletionJob` to override
whatever the ingestion-derived status would otherwise be, so a completed deletion can never look
"indexed" again even though the underlying columns are unchanged.

## One active deletion job per document

At most one `DocumentDeletionJob` per `document_id` may be `PENDING`/`PROCESSING` at a time,
enforced by the partial unique index `ix_document_deletion_jobs_one_active_per_document`
(migration `c8f3a2b6d1e7`). `request_document_deletion()` takes a blocking
`SELECT ... FOR UPDATE` on the document's existing deletion-job rows before deciding whether to
insert, and falls back to catching the index's `IntegrityError` (re-reading and returning the
now-active job) for the residual race the lock alone cannot close — identical strategy to
`retry_ingestion()`.

## Retry is append-only, not resumable-in-place

A `PARTIALLY_FAILED` job's `vector_cleanup_completed`/`storage_cleanup_completed` flags describe
*that* attempt only. Retrying (calling `request_document_deletion()` again) creates a brand-new
`PENDING` row that re-attempts both steps from scratch — always safe, because
`delete_all_tracked_document_vectors()` and `FileStorage.delete()` are both independently
idempotent (re-deleting already-absent vectors/objects is a harmless no-op success), so re-running
an already-completed step costs nothing but the extra I/O. This keeps the implementation free of
any cross-job "resume" bookkeeping while still satisfying "retry after interruption": the
interruption scenarios there are recovered by scheduling a fresh attempt, not by resuming an old
row.
"""

import uuid
from dataclasses import dataclass
from enum import StrEnum

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_deletion_job import DocumentDeletionJob, DocumentDeletionStatus
from app.models.ingestion_job import IngestionJob, IngestionStatus

_ACTIVE_STATUSES = (DocumentDeletionStatus.PENDING, DocumentDeletionStatus.PROCESSING)


class DeletionErrorCode(StrEnum):
    """Stable, machine-identifiable public error codes for a failed deletion step."""

    DOCUMENT_VECTOR_CLEANUP_FAILED = "document_vector_cleanup_failed"
    DOCUMENT_STORAGE_CLEANUP_FAILED = "document_storage_cleanup_failed"


_SAFE_DELETION_FAILURE_MESSAGES = {
    DeletionErrorCode.DOCUMENT_VECTOR_CLEANUP_FAILED: (
        "Document vector cleanup failed. See server logs for the underlying error."
    ),
    DeletionErrorCode.DOCUMENT_STORAGE_CLEANUP_FAILED: (
        "Document storage cleanup failed. See server logs for the underlying error."
    ),
}
_SAFE_DELETION_FAILURE_FALLBACK = "Document deletion failed. See server logs for the underlying error."


def sanitize_deletion_error(error_code: str | None) -> str | None:
    """Map a stored DocumentDeletionJob.error_code to a fixed, safe public message.

    Never returns the raw `error_message` (which may embed a provider connection detail) — the
    raw value stays in Postgres for operator/log inspection only. Returns None when there is no
    error to report (no error_code set).
    """
    if error_code is None:
        return None
    try:
        code = DeletionErrorCode(error_code)
    except ValueError:
        return _SAFE_DELETION_FAILURE_FALLBACK
    return _SAFE_DELETION_FAILURE_MESSAGES.get(code, _SAFE_DELETION_FAILURE_FALLBACK)


class DeletionRequestOutcome(StrEnum):
    """The decided outcome of one request_document_deletion() call — drives the route's HTTP status."""

    DOCUMENT_NOT_FOUND = "document_not_found"
    INGESTION_ACTIVE = "ingestion_active"
    CREATED = "created"
    ALREADY_ACTIVE = "already_active"
    ALREADY_DELETED = "already_deleted"


@dataclass(frozen=True)
class DeletionRequestResult:
    """Typed outcome of request_document_deletion(): the outcome plus the relevant job."""

    outcome: DeletionRequestOutcome
    job: DocumentDeletionJob | None


async def get_latest_deletion_job(session: AsyncSession, document_id: str) -> DocumentDeletionJob | None:
    """Return `document_id`'s most recent DocumentDeletionJob (created_at DESC, id DESC), or None."""
    stmt = (
        select(DocumentDeletionJob)
        .where(DocumentDeletionJob.document_id == document_id)
        .order_by(DocumentDeletionJob.created_at.desc(), DocumentDeletionJob.id.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalars().first()


async def get_latest_deletion_jobs_for_documents(
    session: AsyncSession, document_ids: list[str]
) -> dict[str, DocumentDeletionJob]:
    """Return each document_id's latest DocumentDeletionJob in one batched query — avoids N+1.

    Mirrors `query_service.get_latest_jobs_for_documents`'s exact shape, for the list
    endpoint's lifecycle-status derivation.
    """
    if not document_ids:
        return {}

    stmt = select(DocumentDeletionJob).where(DocumentDeletionJob.document_id.in_(document_ids))
    result = await session.execute(stmt)

    latest_by_document: dict[str, DocumentDeletionJob] = {}
    for job in result.scalars().all():
        current = latest_by_document.get(job.document_id)
        if current is None or (job.created_at, job.id) > (current.created_at, current.id):
            latest_by_document[job.document_id] = job
    return latest_by_document


async def _latest_active_deletion_job(session: AsyncSession, document_id: str) -> DocumentDeletionJob | None:
    """Re-read `document_id`'s latest active deletion job — used after an IntegrityError race."""
    stmt = (
        select(DocumentDeletionJob)
        .where(
            DocumentDeletionJob.document_id == document_id,
            DocumentDeletionJob.status.in_(_ACTIVE_STATUSES),
        )
        .order_by(DocumentDeletionJob.created_at.desc(), DocumentDeletionJob.id.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalars().first()


async def _latest_active_ingestion_job(session: AsyncSession, document_id: str) -> IngestionJob | None:
    """Return document_id's latest ingestion job iff it is PENDING/PROCESSING, else None.

    Queried directly here (not via query_service.get_latest_ingestion_job) to avoid a
    module import cycle: query_service imports this module for lifecycle derivation, so
    this module must not import back from query_service.
    """
    stmt = (
        select(IngestionJob)
        .where(IngestionJob.document_id == document_id)
        .order_by(IngestionJob.created_at.desc(), IngestionJob.id.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    latest = result.scalars().first()
    if latest is not None and latest.status in (IngestionStatus.PENDING, IngestionStatus.PROCESSING):
        return latest
    return None


async def request_document_deletion(session: AsyncSession, document_id: str) -> DeletionRequestResult:
    """Schedule (or report the existing) deletion attempt for `document_id`.

    Decision table (see module docstring for the full design rationale):
    - no Document row -> DOCUMENT_NOT_FOUND (route maps to 404).
    - latest deletion job COMPLETED -> ALREADY_DELETED, idempotent (route maps to 200); no new
      job is created.
    - latest deletion job PENDING/PROCESSING -> ALREADY_ACTIVE (route maps to 202); the existing
      job is returned, no duplicate is created.
    - latest deletion job PARTIALLY_FAILED, or no deletion job yet -> if the latest IngestionJob
      is PENDING/PROCESSING, INGESTION_ACTIVE (route maps to 409) — deletion must never race an
      in-flight ingestion. Otherwise CREATED: a new PENDING DocumentDeletionJob is inserted (route
      maps to 202).

    Takes a blocking `SELECT ... FOR UPDATE` on the document's existing deletion-job rows first,
    so concurrent delete requests for the same document serialize instead of racing; a residual
    insert race is closed by catching the partial unique index's IntegrityError and returning the
    now-existing active job instead of a duplicate.
    """
    document = await session.get(Document, document_id)
    if document is None:
        return DeletionRequestResult(outcome=DeletionRequestOutcome.DOCUMENT_NOT_FOUND, job=None)

    stmt = (
        select(DocumentDeletionJob)
        .where(DocumentDeletionJob.document_id == document_id)
        .order_by(DocumentDeletionJob.created_at.desc(), DocumentDeletionJob.id.desc())
        .with_for_update()
    )
    result = await session.execute(stmt)
    jobs = list(result.scalars().all())
    latest = jobs[0] if jobs else None

    if latest is not None and latest.status == DocumentDeletionStatus.COMPLETED:
        return DeletionRequestResult(outcome=DeletionRequestOutcome.ALREADY_DELETED, job=latest)

    if latest is not None and latest.status in _ACTIVE_STATUSES:
        return DeletionRequestResult(outcome=DeletionRequestOutcome.ALREADY_ACTIVE, job=latest)

    active_ingestion = await _latest_active_ingestion_job(session, document_id)
    if active_ingestion is not None:
        return DeletionRequestResult(outcome=DeletionRequestOutcome.INGESTION_ACTIVE, job=None)

    new_job = DocumentDeletionJob(
        id=str(uuid.uuid4()), document_id=document_id, status=DocumentDeletionStatus.PENDING
    )
    session.add(new_job)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        existing = await _latest_active_deletion_job(session, document_id)
        return DeletionRequestResult(outcome=DeletionRequestOutcome.ALREADY_ACTIVE, job=existing)

    return DeletionRequestResult(outcome=DeletionRequestOutcome.CREATED, job=new_job)


__all__ = [
    "DeletionErrorCode",
    "DeletionRequestOutcome",
    "DeletionRequestResult",
    "get_latest_deletion_job",
    "get_latest_deletion_jobs_for_documents",
    "request_document_deletion",
    "sanitize_deletion_error",
]
