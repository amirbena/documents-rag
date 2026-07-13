"""Document upload + read-only inspection/download endpoints, plus deletion scheduling/status.

Upload does not parse, chunk, embed, or index the document — that happens asynchronously via
`app.services.ingestion_worker`. The five original read routes here (list, detail, ingestion
status, failure, download) are strictly read-only: no mutation of Postgres, object storage,
Qdrant, ingestion jobs, or cleanup records happens on any of them. `POST .../ingestion/retry`
(Phase 2.8.3) only ever inserts a new PENDING IngestionJob row via
`app.services.ingestion_retry_service`. `DELETE /documents/{id}` and `GET .../deletion` (Phase
2.8.4) schedule/report full document deletion via `app.services.document_deletion_service` — the
DELETE route only ever inserts a new PENDING DocumentDeletionJob row (or reports an existing one);
the actual cross-system cleanup runs out-of-band via
`scripts/process_pending_document_deletions.py` / `DocumentDeletionWorker`, never inline in this
request. All business/aggregation/locking logic (lifecycle derivation, latest-job selection,
sanitization, N+1-avoidance, retry/deletion transaction semantics) lives in the service modules —
routes here only parse/inject/call/copy-status, per CLAUDE.md's "Route Layer Style".
"""

from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, status
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.session import get_db_session
from app.rag.providers.provider_factory import get_vector_store as _resolve_vector_store
from app.rag.providers.vector_store import VectorStore
from app.schemas.documents import (
    DocumentDeletionResponse,
    DocumentDeletionStatusResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentUploadResponse,
    IngestionFailureResponse,
    IngestionRetryResponse,
    IngestionStatusResponse,
)
from app.services.document_deletion_service import (
    DeletionRequestOutcome,
    get_latest_deletion_job,
    request_document_deletion,
    sanitize_deletion_error,
)
from app.services.document_query_service import (
    DEFAULT_LIST_LIMIT,
    MAX_LIST_LIMIT,
    build_document_list_response,
    download_document,
    get_document_detail_result,
    get_document_failure_result,
    get_document_ingestion_result,
)
from app.services.document_upload_service import upload_document
from app.services.ingestion_retry_service import RetryOutcome, retry_ingestion
from app.storage.contract import FileStorage
from app.storage.factory import create_file_storage

router = APIRouter()


def get_file_storage() -> FileStorage:
    """Build the configured FileStorage implementation via the storage factory."""
    return create_file_storage()


def get_vector_store() -> VectorStore:
    """Build the configured VectorStore implementation via the provider factory."""
    return _resolve_vector_store()


@router.post(
    "/documents",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_document_route(
    file: UploadFile,
    db: AsyncSession = Depends(get_db_session),
    storage: FileStorage = Depends(get_file_storage),
) -> DocumentUploadResponse:
    """Save the uploaded file, create Document + pending IngestionJob rows, return 202."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    original_filename = file.filename or "unnamed"
    content_type = file.content_type or "application/octet-stream"

    document, job = await upload_document(
        content=content,
        original_filename=original_filename,
        content_type=content_type,
        storage=storage,
        session=db,
    )

    return DocumentUploadResponse(document_id=document.id, job_id=job.id, status=job.status)


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents_route(
    limit: int = Query(default=DEFAULT_LIST_LIMIT, ge=1, le=MAX_LIST_LIMIT),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db_session),
) -> DocumentListResponse:
    """List documents (newest first) with their derived lifecycle status; always 200."""
    return await build_document_list_response(db, limit=limit, offset=offset)


@router.get("/documents/{document_id}", response_model=DocumentDetailResponse)
async def get_document_route(
    document_id: str, db: AsyncSession = Depends(get_db_session)
) -> DocumentDetailResponse:
    """Return one document's detail view; 404 if it does not exist."""
    result = await get_document_detail_result(db, document_id)
    if result.response is None:
        raise HTTPException(status_code=result.status_code, detail="Document not found.")
    return result.response


@router.get("/documents/{document_id}/ingestion", response_model=IngestionStatusResponse)
async def get_document_ingestion_route(
    document_id: str, db: AsyncSession = Depends(get_db_session)
) -> IngestionStatusResponse:
    """Return one document's latest ingestion status; 404 only if the document itself is missing.

    A document with no ingestion job yet is a 200 with null job_id/status/created_at/updated_at,
    not a 404 — see app.services.document_query_service's module docstring.
    """
    result = await get_document_ingestion_result(db, document_id)
    if result.response is None:
        raise HTTPException(status_code=result.status_code, detail="Document not found.")
    return result.response


@router.get("/documents/{document_id}/failure", response_model=IngestionFailureResponse)
async def get_document_failure_route(
    document_id: str, db: AsyncSession = Depends(get_db_session)
) -> IngestionFailureResponse:
    """Return one document's latest failed ingestion job; 404 if the document or a failure is missing.

    See DocumentFailureResult's docstring in document_query_service.py for why "no failure to
    inspect" maps to 404 here rather than a 200-with-null body.
    """
    result = await get_document_failure_result(db, document_id)
    if result.response is None:
        raise HTTPException(status_code=result.status_code, detail="No failed ingestion job found.")
    return result.response


_RETRY_OUTCOME_ERRORS = {
    RetryOutcome.DOCUMENT_NOT_FOUND: (status.HTTP_404_NOT_FOUND, "Document not found."),
    RetryOutcome.ALREADY_COMPLETED: (
        status.HTTP_409_CONFLICT,
        "Document is already indexed; use the re-index path instead of retry.",
    ),
    RetryOutcome.DELETION_ACTIVE: (
        status.HTTP_409_CONFLICT,
        "Document is being deleted, deletion failed, or the document was deleted; "
        "ingestion retry is not available.",
    ),
}


@router.post("/documents/{document_id}/ingestion/retry", response_model=IngestionRetryResponse)
async def retry_document_ingestion_route(
    document_id: str, response: Response, db: AsyncSession = Depends(get_db_session)
) -> IngestionRetryResponse:
    """Schedule a new ingestion attempt for a FAILED/stale document, or report its active job.

    202 with `created=True` when a new PENDING job was inserted; 200 with `created=False` when an
    already-active job was returned instead (nothing new was scheduled); 404 if the document does
    not exist; 409 if the latest job is already COMPLETED (see
    `app.services.ingestion_retry_service.retry_ingestion` for the full decision table).
    """
    settings = get_settings()
    result = await retry_ingestion(
        db, document_id, stale_after_seconds=settings.ingestion_stale_after_seconds
    )

    if result.outcome in _RETRY_OUTCOME_ERRORS:
        status_code, detail = _RETRY_OUTCOME_ERRORS[result.outcome]
        raise HTTPException(status_code=status_code, detail=detail)

    assert result.job is not None
    created = result.outcome == RetryOutcome.CREATED
    response.status_code = status.HTTP_202_ACCEPTED if created else status.HTTP_200_OK
    return IngestionRetryResponse(
        document_id=document_id,
        job_id=result.job.id,
        status=result.job.status,
        created=created,
    )


def _content_disposition_header(original_filename: str) -> str:
    """Build an RFC 5987/6266-compliant Content-Disposition header for a (possibly Unicode) filename.

    HTTP headers are Latin-1/ASCII, so a raw Hebrew (or any non-ASCII) filename cannot be
    interpolated directly. Provides both forms browsers actually respect: an ASCII-only
    `filename="..."` fallback (non-ASCII characters replaced) and the percent-encoded
    `filename*=UTF-8''...` form carrying the exact original name. `original_filename` is
    user-controlled (an uploader picks it, unsanitized) — the `filename*=` form is inherently safe
    because `quote()` percent-encodes every byte outside its unreserved set (including CR/LF/`"`),
    but the `filename="..."` fallback is interpolated into a quoted string, so control characters
    (which could inject a CRLF header-splitting sequence) and quote/backslash characters (which
    could break out of the quoted string and inject extra header parameters) are stripped from it
    first — replaced with `_`, matching the existing non-ASCII-replacement character.
    """
    ascii_fallback = original_filename.encode("ascii", errors="replace").decode("ascii")
    ascii_fallback = "".join(
        "_" if char in ('"', "\\", "?") or ord(char) < 0x20 or ord(char) == 0x7F else char
        for char in ascii_fallback
    )
    encoded = quote(original_filename, safe="")
    return f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{encoded}'


@router.get("/documents/{document_id}/download")
async def download_document_route(
    document_id: str,
    db: AsyncSession = Depends(get_db_session),
    storage: FileStorage = Depends(get_file_storage),
) -> Response:
    """Stream a document's original bytes back to the client.

    Reads the full object into memory via `FileStorage.read()` (both LocalFileStorage and
    MinioFileStorage return `bytes`, not a stream) and returns it in one `Response` — the same
    unbounded-memory characteristic `POST /api/v1/documents` already has today (it also does
    `await file.read()` with no size limit anywhere in the codebase), not a new risk introduced
    here. 404 if the document doesn't exist, 409 if the document exists but its storage object is
    missing (a real inconsistency, not "not found"), 503 if the storage backend itself is
    unreachable. Never touches a local filesystem path or provider SDK type directly — always
    through the injected `FileStorage`.
    """
    result = await download_document(db, document_id, storage)
    if result.status_code != status.HTTP_200_OK:
        raise HTTPException(status_code=result.status_code, detail=result.detail)

    assert result.content is not None
    assert result.content_type is not None
    assert result.original_filename is not None

    return Response(
        content=result.content,
        media_type=result.content_type,
        headers={"Content-Disposition": _content_disposition_header(result.original_filename)},
    )


@router.delete("/documents/{document_id}", response_model=DocumentDeletionResponse)
async def delete_document_route(
    document_id: str, response: Response, db: AsyncSession = Depends(get_db_session)
) -> DocumentDeletionResponse:
    """Schedule full deletion of a document, or report its existing/completed deletion attempt.

    202 with `created=True` when a new PENDING DocumentDeletionJob was inserted; 202 with
    `created=False` when an already-active (PENDING/PROCESSING) job was returned instead; 200
    with `created=False` when the document was already fully deleted (idempotent); 404 if the
    document does not exist; 409 if the document's latest ingestion job is still PENDING/
    PROCESSING (deletion never races an in-flight ingestion). Never performs the actual
    cross-system cleanup inline — see `app.services.document_deletion_service.
    request_document_deletion` and `DocumentDeletionWorker`.
    """
    result = await request_document_deletion(db, document_id)

    if result.outcome == DeletionRequestOutcome.DOCUMENT_NOT_FOUND:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    if result.outcome == DeletionRequestOutcome.INGESTION_ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document has an active ingestion job; cannot delete until it resolves.",
        )

    assert result.job is not None
    created = result.outcome == DeletionRequestOutcome.CREATED
    response.status_code = (
        status.HTTP_200_OK
        if result.outcome == DeletionRequestOutcome.ALREADY_DELETED
        else status.HTTP_202_ACCEPTED
    )
    return DocumentDeletionResponse(
        document_id=document_id,
        deletion_job_id=result.job.id,
        status=result.job.status,
        created=created,
    )


@router.get("/documents/{document_id}/deletion", response_model=DocumentDeletionStatusResponse)
async def get_document_deletion_route(
    document_id: str, db: AsyncSession = Depends(get_db_session)
) -> DocumentDeletionStatusResponse:
    """Return one document's latest deletion attempt; 404 if no deletion was ever requested.

    Never exposes a storage key/bucket, Qdrant collection internals, or a raw provider exception —
    `safe_message` is always `sanitize_deletion_error()`'s fixed, generic text.
    """
    job = await get_latest_deletion_job(db, document_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No deletion attempt found.")

    return DocumentDeletionStatusResponse(
        document_id=document_id,
        deletion_job_id=job.id,
        status=job.status,
        vector_cleanup_completed=job.vector_cleanup_completed,
        storage_cleanup_completed=job.storage_cleanup_completed,
        safe_message=sanitize_deletion_error(job.error_code),
        created_at=job.created_at,
        updated_at=job.updated_at,
        completed_at=job.completed_at,
    )
