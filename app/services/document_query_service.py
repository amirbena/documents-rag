"""Read-only queries and lifecycle-status derivation for documents and their ingestion jobs.

Owns every read query behind the document-read APIs (`app/api/v1/routes/documents.py`) — list,
detail, ingestion-status, failure, and download lookups — plus the pure lifecycle-status
derivation rule. Nothing here ever writes to Postgres, object storage, or Qdrant: this module is
strictly read-only, mirroring the flat function-based style of `app/services/index_registry.py`
(no `app/repositories/` abstraction exists in this codebase).

## Lifecycle status derivation

The source of truth for a document's lifecycle status is its *latest* `IngestionJob` (ordered by
`created_at` DESC, `id` DESC as a deterministic tiebreaker), plus `Document.indexed_at` to confirm
genuine indexed state:

- No `IngestionJob` row exists at all -> `UPLOADED`. In practice this is unreachable via the
  normal upload flow: `app.services.document_upload_service.upload_document()` always creates
  exactly one `Document` and one `IngestionJob` row in the same commit, so every document created
  through the API has at least one job. This status exists defensively, for any pre-existing or
  malformed data outside that flow.
- Latest job `PENDING` -> `PENDING`.
- Latest job `PROCESSING` -> `PROCESSING`.
- Latest job `FAILED` -> `FAILED`.
- Latest job `COMPLETED` -> `INDEXED`. `app.services.ingestion_worker`'s
  `IngestionWorker.process_next_job()` calls `mark_document_indexed()` (which sets
  `document.indexed_at`) and then commits the job's `COMPLETED` status together with the
  document's indexing columns in the *same* `session.commit()` call — so a `COMPLETED` job should
  always imply `document.indexed_at is not None`, including for zero-chunk documents (see
  "Zero-chunk behavior" in ARCHITECTURE.md). If a `COMPLETED` job is ever observed with
  `indexed_at is None` (e.g. from data written before this invariant existed), this function still
  reports `INDEXED` rather than fabricating a new status — the job itself is the authoritative
  completion signal — but this is a genuine, documented edge case, not a proven-impossible one.

## Deletion precedence (Phase 2.8.4)

If a `DocumentDeletionJob` exists for the document, it always takes precedence over the
ingestion-derived status above:

- latest deletion job PENDING/PROCESSING -> DELETING.
- latest deletion job PARTIALLY_FAILED -> DELETION_FAILED.
- latest deletion job COMPLETED -> DELETED.
- no deletion job at all -> fall through to the ingestion-derived rule above.

Once a document is DELETED it can never appear INDEXED/PENDING/etc. again merely because its
(unchanged) IngestionJob/indexing columns still describe a prior successful index — the deletion
job's status is checked first and, if present, is authoritative.
"""

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_deletion_job import DocumentDeletionJob, DocumentDeletionStatus
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.schemas.documents import (
    DocumentDetailResponse,
    DocumentLifecycleStatus,
    DocumentListResponse,
    DocumentSummaryResponse,
    IngestionFailureResponse,
    IngestionStatusResponse,
)
from app.services.documents.deletion_service import (
    get_latest_deletion_job,
    get_latest_deletion_jobs_for_documents,
)
from app.storage.contract import FileStorage
from app.storage.errors import StorageObjectNotFoundError, StorageReadError, StorageUnavailableError
from app.storage.keys import resolve_document_storage_key

DEFAULT_LIST_LIMIT = 20
MAX_LIST_LIMIT = 100

# A fixed, generic message returned to API clients for a failed ingestion job — the raw
# `IngestionJob.error_message` (e.g. a QdrantVectorStoreError's stringified `{exc}`, which can
# embed a connection/host detail) is never returned verbatim. See "Ingestion failure
# sanitization" below.
_SAFE_INGESTION_FAILURE_MESSAGE = (
    "Document ingestion failed. See server logs for the underlying error."
)


def derive_lifecycle_status(
    document: Document,
    latest_job: IngestionJob | None,
    latest_deletion_job: DocumentDeletionJob | None = None,
) -> DocumentLifecycleStatus:
    """Derive a document's lifecycle status; a deletion job, if any, always takes precedence.

    See the module docstring's "Deletion precedence" section for the full rule.
    """
    if latest_deletion_job is not None:
        if latest_deletion_job.status in (
            DocumentDeletionStatus.PENDING,
            DocumentDeletionStatus.PROCESSING,
        ):
            return DocumentLifecycleStatus.DELETING
        if latest_deletion_job.status == DocumentDeletionStatus.PARTIALLY_FAILED:
            return DocumentLifecycleStatus.DELETION_FAILED
        if latest_deletion_job.status == DocumentDeletionStatus.COMPLETED:
            return DocumentLifecycleStatus.DELETED

    if latest_job is None:
        return DocumentLifecycleStatus.UPLOADED
    if latest_job.status == IngestionStatus.PENDING:
        return DocumentLifecycleStatus.PENDING
    if latest_job.status == IngestionStatus.PROCESSING:
        return DocumentLifecycleStatus.PROCESSING
    if latest_job.status == IngestionStatus.FAILED:
        return DocumentLifecycleStatus.FAILED
    # COMPLETED. See module docstring: this should always imply indexed_at is set, but the
    # job's own status is treated as authoritative even in the (undocumented-elsewhere,
    # theoretically-unreachable) case where indexed_at is somehow still None.
    del document
    return DocumentLifecycleStatus.INDEXED


def sanitize_ingestion_error(_raw_error_message: str | None) -> str:
    """Map a stored (potentially unsafe) IngestionJob.error_message to a fixed, safe message.

    Real `error_message` values come from `str(exc)` of internal exceptions (e.g.
    `QdrantVectorStoreError`, `DocumentTextExtractionError`) whose text can embed a connection
    detail or host (e.g. an httpx `ConnectError` string). Rather than attempt to pattern-match
    "safe" substrings out of arbitrary exception text, this always returns one fixed, generic
    message — mirroring `app/api/v1/routes/chat.py`'s `_SAFE_ERROR_MESSAGE` pattern. The raw
    message stays in Postgres (`ingestion_jobs.error_message`) for operator/log inspection; it is
    never included in an API response body.
    """
    return _SAFE_INGESTION_FAILURE_MESSAGE


async def list_documents(
    session: AsyncSession, *, limit: int, offset: int
) -> tuple[list[Document], int]:
    """Return one page of Documents (created_at DESC, id DESC) plus the total row count."""
    total = await session.execute(select(func.count()).select_from(Document))
    total_count = total.scalar_one()

    stmt = (
        select(Document)
        .order_by(Document.created_at.desc(), Document.id.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all()), total_count


async def get_document(session: AsyncSession, document_id: str) -> Document | None:
    """Return the Document with `document_id`, or None if it does not exist."""
    return await session.get(Document, document_id)


async def get_latest_ingestion_job(
    session: AsyncSession, document_id: str
) -> IngestionJob | None:
    """Return `document_id`'s most recent IngestionJob (created_at DESC, id DESC), or None."""
    stmt = (
        select(IngestionJob)
        .where(IngestionJob.document_id == document_id)
        .order_by(IngestionJob.created_at.desc(), IngestionJob.id.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalars().first()


async def get_latest_failed_ingestion_job(
    session: AsyncSession, document_id: str
) -> IngestionJob | None:
    """Return `document_id`'s most recent FAILED IngestionJob (created_at DESC, id DESC), or None."""
    stmt = (
        select(IngestionJob)
        .where(IngestionJob.document_id == document_id, IngestionJob.status == IngestionStatus.FAILED)
        .order_by(IngestionJob.created_at.desc(), IngestionJob.id.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalars().first()


async def get_latest_jobs_for_documents(
    session: AsyncSession, document_ids: list[str]
) -> dict[str, IngestionJob]:
    """Return each document_id's latest IngestionJob in one batched query — avoids N+1.

    Fetches every job for the given document_ids in a single query, then picks the latest per
    document in Python (by created_at, then id, matching get_latest_ingestion_job's ordering).
    """
    if not document_ids:
        return {}

    stmt = select(IngestionJob).where(IngestionJob.document_id.in_(document_ids))
    result = await session.execute(stmt)

    latest_by_document: dict[str, IngestionJob] = {}
    for job in result.scalars().all():
        current = latest_by_document.get(job.document_id)
        if current is None or (job.created_at, job.id) > (current.created_at, current.id):
            latest_by_document[job.document_id] = job
    return latest_by_document


def _to_summary(
    document: Document,
    latest_job: IngestionJob | None,
    latest_deletion_job: DocumentDeletionJob | None,
) -> DocumentSummaryResponse:
    """Build a DocumentSummaryResponse from a Document and its (already-resolved) latest jobs."""
    return DocumentSummaryResponse(
        id=document.id,
        original_filename=document.original_filename,
        content_type=document.content_type,
        size_bytes=document.file_size,
        status=derive_lifecycle_status(document, latest_job, latest_deletion_job),
        created_at=document.created_at,
        latest_ingestion_activity_at=latest_job.updated_at if latest_job is not None else None,
    )


async def build_document_list_response(
    session: AsyncSession, *, limit: int, offset: int
) -> DocumentListResponse:
    """List one page of documents with their derived lifecycle status; always HTTP 200.

    Deleted documents remain listed (lifecycle=DELETED) — this endpoint never filters them out.
    """
    documents, total = await list_documents(session, limit=limit, offset=offset)
    document_ids = [document.id for document in documents]
    latest_jobs = await get_latest_jobs_for_documents(session, document_ids)
    latest_deletion_jobs = await get_latest_deletion_jobs_for_documents(session, document_ids)

    items = [
        _to_summary(document, latest_jobs.get(document.id), latest_deletion_jobs.get(document.id))
        for document in documents
    ]
    return DocumentListResponse(items=items, total=total, limit=limit, offset=offset)


@dataclass(frozen=True)
class DocumentDetailResult:
    """Typed outcome of a document-detail lookup: the response body plus the HTTP status to apply."""

    response: DocumentDetailResponse | None
    status_code: int


async def get_document_detail_result(session: AsyncSession, document_id: str) -> DocumentDetailResult:
    """Look up one document's detail view; 404 (empty response) if it does not exist.

    A successfully deleted document remains inspectable here (lifecycle=DELETED) — this endpoint
    never returns 404 for a document that exists but was deleted.
    """
    document = await get_document(session, document_id)
    if document is None:
        return DocumentDetailResult(response=None, status_code=404)

    latest_job = await get_latest_ingestion_job(session, document_id)
    latest_deletion_job = await get_latest_deletion_job(session, document_id)
    response = DocumentDetailResponse(
        id=document.id,
        original_filename=document.original_filename,
        content_type=document.content_type,
        size_bytes=document.file_size,
        storage_provider=document.storage_provider,
        status=derive_lifecycle_status(document, latest_job, latest_deletion_job),
        collection_name=document.collection_name,
        embedding_version=document.embedding_version,
        chunking_version=document.chunking_version,
        indexed_at=document.indexed_at,
        latest_ingestion_job_id=latest_job.id if latest_job is not None else None,
        latest_ingestion_status=latest_job.status if latest_job is not None else None,
        created_at=document.created_at,
    )
    return DocumentDetailResult(response=response, status_code=200)


@dataclass(frozen=True)
class DocumentIngestionResult:
    """Typed outcome of an ingestion-status lookup: response body plus the HTTP status to apply."""

    response: IngestionStatusResponse | None
    status_code: int


async def get_document_ingestion_result(
    session: AsyncSession, document_id: str
) -> DocumentIngestionResult:
    """Look up one document's latest ingestion status; 404 if the document itself doesn't exist.

    A document with no ingestion job yet (see module docstring — effectively unreachable via the
    normal upload flow, but not proven impossible) is a legitimate 200 response with
    job_id/status/created_at/updated_at all null, not a 404 — the document exists, it simply has
    no current job to describe.
    """
    document = await get_document(session, document_id)
    if document is None:
        return DocumentIngestionResult(response=None, status_code=404)

    latest_job = await get_latest_ingestion_job(session, document_id)
    response = IngestionStatusResponse(
        document_id=document_id,
        job_id=latest_job.id if latest_job is not None else None,
        status=latest_job.status if latest_job is not None else None,
        created_at=latest_job.created_at if latest_job is not None else None,
        updated_at=latest_job.updated_at if latest_job is not None else None,
    )
    return DocumentIngestionResult(response=response, status_code=200)


@dataclass(frozen=True)
class DocumentFailureResult:
    """Typed outcome of a failure lookup: response body plus the HTTP status to apply.

    404 is returned both when the document itself does not exist and when the document exists
    but has no failed ingestion job — "inspect the failure" implies a failure should exist to
    inspect, so an absent one is treated the same as a missing resource, not a 200-with-null body.
    This is a deliberate, documented choice (the 200-with-null alternative is equally defensible
    for the "no job yet" case on the ingestion-status endpoint above; the failure endpoint is
    narrower in intent, so 404 was chosen for it specifically).
    """

    response: IngestionFailureResponse | None
    status_code: int


async def get_document_failure_result(session: AsyncSession, document_id: str) -> DocumentFailureResult:
    """Look up one document's latest failed ingestion job; 404 if none exists (see class docstring)."""
    document = await get_document(session, document_id)
    if document is None:
        return DocumentFailureResult(response=None, status_code=404)

    failed_job = await get_latest_failed_ingestion_job(session, document_id)
    if failed_job is None:
        return DocumentFailureResult(response=None, status_code=404)

    response = IngestionFailureResponse(
        document_id=document_id,
        job_id=failed_job.id,
        status=failed_job.status,
        safe_message=sanitize_ingestion_error(failed_job.error_message),
        failed_at=failed_job.updated_at,
    )
    return DocumentFailureResult(response=response, status_code=200)


@dataclass(frozen=True)
class DocumentDownloadResult:
    """Typed outcome of a download attempt: content (on success) plus the HTTP status to apply.

    `detail` carries a safe, fixed message for non-200 outcomes — never a raw storage exception
    message. Storage I/O (`FileStorage.read`) happens here, not in the route, so the route stays
    a thin dependency-injection + status-copy controller.
    """

    status_code: int
    content: bytes | None = None
    content_type: str | None = None
    original_filename: str | None = None
    detail: str | None = None


def resolve_download_key(document: Document) -> str:
    """Resolve the storage key to read a document's original content from — see app.storage.keys."""
    return resolve_document_storage_key(document)


async def download_document(
    session: AsyncSession, document_id: str, storage: FileStorage
) -> DocumentDownloadResult:
    """Resolve a document, read its original bytes from `storage`, and return a typed result.

    404 if the document row doesn't exist; 410 if the document was successfully deleted (Phase
    2.8.4 — the Postgres resource still exists, only its content was intentionally removed, so
    this is Gone rather than Not Found); 409 if the row exists but its storage object is
    otherwise missing (a real document/storage inconsistency); 503 if the storage backend itself
    is unreachable/failing for a reason other than not-found. Never returns a raw storage
    exception message or a local filesystem path.
    """
    document = await get_document(session, document_id)
    if document is None:
        return DocumentDownloadResult(status_code=404, detail="Document not found.")

    latest_deletion_job = await get_latest_deletion_job(session, document_id)
    if latest_deletion_job is not None and latest_deletion_job.status == DocumentDeletionStatus.COMPLETED:
        return DocumentDownloadResult(status_code=410, detail="Document has been deleted.")

    key = resolve_download_key(document)
    try:
        content = await storage.read(key)
    except StorageObjectNotFoundError:
        return DocumentDownloadResult(
            status_code=409, detail="Document exists but its content is unavailable in storage."
        )
    except (StorageUnavailableError, StorageReadError):
        return DocumentDownloadResult(status_code=503, detail="Storage backend is unavailable.")

    return DocumentDownloadResult(
        status_code=200,
        content=content,
        content_type=document.content_type,
        original_filename=document.original_filename,
    )


__all__ = [
    "DEFAULT_LIST_LIMIT",
    "MAX_LIST_LIMIT",
    "DocumentDetailResult",
    "DocumentDownloadResult",
    "DocumentFailureResult",
    "DocumentIngestionResult",
    "DocumentLifecycleStatus",
    "build_document_list_response",
    "derive_lifecycle_status",
    "download_document",
    "get_document",
    "get_document_detail_result",
    "get_document_failure_result",
    "get_document_ingestion_result",
    "get_latest_failed_ingestion_job",
    "get_latest_ingestion_job",
    "get_latest_jobs_for_documents",
    "list_documents",
    "resolve_download_key",
    "sanitize_ingestion_error",
]
