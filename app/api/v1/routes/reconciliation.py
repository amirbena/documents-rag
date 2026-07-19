"""Read-only bounded batch document lifecycle audit endpoint (Phase 2.8.7, subtask 3).

Thin HTTP boundary over the already-implemented `audit_document_lifecycle_batch()`
(`app/services/reconciliation/document_audit_batch_service.py`) — this route never audits a
document itself, never mutates `Document`/job rows, and never recalculates the per-document
classification or the aggregate summary counts the service already computed. Per CLAUDE.md's
"Route Layer Style" (same convention as `app/api/v1/routes/documents.py`/`reindex.py`), it only
parses/injects/calls-one-service/maps-the-result.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.session import get_db_session
from app.rag.providers.provider_factory import get_vector_store as _resolve_vector_store
from app.rag.providers.vector_store import VectorStore
from app.schemas.reconciliation import (
    DocumentAuditFindingResponse,
    DocumentBatchAuditItemResponse,
    DocumentBatchAuditResponse,
    DocumentBatchAuditSummaryResponse,
)
from app.services.reconciliation.document_audit_batch_service import (
    DEFAULT_BATCH_LIMIT,
    MAX_BATCH_LIMIT,
    MIN_BATCH_LIMIT,
    DocumentAuditSummary,
    InvalidAuditBatchLimitError,
    InvalidAuditCursorError,
    audit_document_lifecycle_batch,
)
from app.services.reconciliation.document_audit_service import DocumentLifecycleFinding
from app.storage.contract import FileStorage
from app.storage.factory import create_file_storage

router = APIRouter()

# Never exposes the underlying cursor's implementation (Base64/JSON) or the service's own
# validation-failure message — a single fixed detail for every InvalidAuditCursorError case.
_CURSOR_ERROR_DETAIL = "The cursor is invalid or has expired."


def get_file_storage() -> FileStorage:
    """Build the configured FileStorage implementation via the storage factory."""
    return create_file_storage()


def get_vector_store() -> VectorStore:
    """Build the configured VectorStore implementation via the provider factory."""
    return _resolve_vector_store()


def _finding_response(finding: DocumentLifecycleFinding) -> DocumentAuditFindingResponse:
    return DocumentAuditFindingResponse(
        code=finding.code,
        severity=finding.severity,
        summary=finding.summary,
        expected_state=finding.expected_state,
        actual_state=finding.actual_state,
        suggested_action=finding.suggested_action,
        destructive_risk=finding.destructive_risk,
    )


def _item_response(summary: DocumentAuditSummary) -> DocumentBatchAuditItemResponse:
    return DocumentBatchAuditItemResponse(
        document_id=summary.document_id,
        original_filename=summary.original_filename,
        created_at=summary.created_at,
        overall_status=summary.overall_status,
        classification=summary.classification,
        issues=[_finding_response(finding) for finding in summary.findings],
    )


@router.get("/reconciliation/documents/audit", response_model=DocumentBatchAuditResponse)
async def audit_documents_batch_route(
    limit: int = Query(default=DEFAULT_BATCH_LIMIT, ge=MIN_BATCH_LIMIT, le=MAX_BATCH_LIMIT),
    cursor: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db_session),
    file_storage: FileStorage = Depends(get_file_storage),
    vector_store: VectorStore = Depends(get_vector_store),
) -> DocumentBatchAuditResponse:
    """Bounded, read-only, oldest-first page of the document lifecycle batch audit.

    Delegates entirely to `audit_document_lifecycle_batch()`, exactly once. 200 for every page,
    including an empty repository (empty `items`, zeroed `summary`, `next_cursor=null`) — never
    404. 400 for a malformed/invalid `cursor`. FastAPI's own query validation already rejects an
    out-of-bounds `limit` before this function runs; `InvalidAuditBatchLimitError` is still handled
    here as defense in depth, since the service remains the authoritative validator. Any other
    exception (an unexpected service/database failure) propagates as a normal 500 — no partial
    result is ever silently returned for those.
    """
    settings = get_settings()
    try:
        result = await audit_document_lifecycle_batch(
            db, settings, file_storage, vector_store, limit=limit, cursor=cursor
        )
    except InvalidAuditCursorError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_CURSOR_ERROR_DETAIL) from exc
    except InvalidAuditBatchLimitError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return DocumentBatchAuditResponse(
        items=[_item_response(summary) for summary in result.documents],
        summary=DocumentBatchAuditSummaryResponse(
            total=result.scanned_count,
            consistent=result.consistent_count,
            transitional=result.transitional_count,
            warning=result.warning_count,
            inconsistent=result.inconsistent_count,
            not_found=result.not_found_count,
            dependency_unavailable=result.dependency_unavailable_count,
            finding_counts=result.finding_counts,
        ),
        limit=limit,
        next_cursor=result.next_cursor,
    )
