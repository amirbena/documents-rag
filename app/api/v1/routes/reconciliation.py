"""Read-only reconciliation reporting endpoints: bounded batch document audit (Phase 2.8.7,
subtask 3), single-document audit, and collection consistency report (both subtask 5).

Thin HTTP boundary over already-implemented services — `audit_document_lifecycle_batch()`,
`audit_document_lifecycle()`, and `build_collection_reconciliation_report()`
(`app/services/reconciliation/*`). No route here audits a document/collection itself, mutates any
row, or recalculates a classification/count the service already computed. Per CLAUDE.md's "Route
Layer Style" (same convention as `app/api/v1/routes/documents.py`/`reindex.py`), every route only
parses/injects/calls-one-service/maps-the-result.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.session import get_db_session
from app.rag.providers.provider_factory import get_vector_store as _resolve_vector_store
from app.rag.providers.vector_store import VectorStore
from app.schemas.reconciliation import (
    CollectionReportFindingResponse,
    CollectionReportResponse,
    DocumentAuditFindingResponse,
    DocumentBatchAuditItemResponse,
    DocumentBatchAuditResponse,
    DocumentBatchAuditSummaryResponse,
    DocumentDatabaseStateResponse,
    DocumentFileStorageStateResponse,
    DocumentLifecycleAuditResponse,
    DocumentVectorStoreStateResponse,
)
from app.services.reconciliation.collection_reconciliation_report_service import (
    CollectionReportFinding,
    InvalidCollectionNameError,
    build_collection_reconciliation_report,
)
from app.services.reconciliation.document_audit_batch_service import (
    DEFAULT_BATCH_LIMIT,
    MAX_BATCH_LIMIT,
    MIN_BATCH_LIMIT,
    DocumentAuditSummary,
    InvalidAuditBatchLimitError,
    InvalidAuditCursorError,
    audit_document_lifecycle_batch,
    classify_document_audit,
)
from app.services.reconciliation.document_audit_service import (
    DocumentLifecycleAuditResult,
    DocumentLifecycleFinding,
    audit_document_lifecycle,
)
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


def _database_response(result: DocumentLifecycleAuditResult) -> DocumentDatabaseStateResponse:
    state = result.postgres_state
    if state is None:
        return DocumentDatabaseStateResponse(
            document_exists=False,
            collection_name=None,
            document_created_at=None,
            latest_ingestion_status=None,
            latest_deletion_status=None,
            latest_reindex_status=None,
            latest_reindex_activated=False,
            pending_cleanup_collections=[],
        )
    return DocumentDatabaseStateResponse(
        document_exists=True,
        collection_name=state.collection_name,
        document_created_at=state.document_created_at,
        latest_ingestion_status=state.latest_ingestion_status,
        latest_deletion_status=state.latest_deletion_status,
        latest_reindex_status=state.latest_reindex_status,
        latest_reindex_activated=state.latest_reindex_activated,
        pending_cleanup_collections=list(state.pending_cleanup_collections),
    )


def _file_storage_state_response(
    result: DocumentLifecycleAuditResult,
) -> DocumentFileStorageStateResponse | None:
    state = result.storage_state
    if state is None:
        return None
    return DocumentFileStorageStateResponse(inspected=state.inspected, source_file_exists=state.exists)


def _vector_store_state_response(
    result: DocumentLifecycleAuditResult,
) -> DocumentVectorStoreStateResponse | None:
    state = result.vector_state
    if state is None:
        return None
    collection_name = result.postgres_state.collection_name if result.postgres_state is not None else None
    return DocumentVectorStoreStateResponse(
        inspected=state.inspected,
        collection_name=collection_name,
        collection_exists=state.collection_exists,
        has_vectors=state.has_vectors,
        vector_count=state.vector_count,
    )


def _collection_finding_response(finding: CollectionReportFinding) -> CollectionReportFindingResponse:
    return CollectionReportFindingResponse(
        code=finding.code,
        severity=finding.severity,
        summary=finding.summary,
        expected_state=finding.expected_state,
        actual_state=finding.actual_state,
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


@router.get(
    "/reconciliation/documents/{document_id}/audit", response_model=DocumentLifecycleAuditResponse
)
async def audit_single_document_route(
    document_id: str,
    db: AsyncSession = Depends(get_db_session),
    file_storage: FileStorage = Depends(get_file_storage),
    vector_store: VectorStore = Depends(get_vector_store),
) -> DocumentLifecycleAuditResponse:
    """Read-only lifecycle audit for exactly one document.

    Delegates entirely to `audit_document_lifecycle()`, exactly once — never routed through the
    batch auditor. Always 200, including when the document doesn't exist at all
    (`classification=not_found`, `database.document_exists=false`) — the service already
    represents that as a typed result, not an exception, so this route preserves that rather than
    inventing a 404. `classification` is computed via the same shared `classify_document_audit()`
    the batch endpoint uses internally — never recalculated ad hoc here. Any unexpected exception
    (a Postgres/Object-Storage/Qdrant failure the service doesn't already turn into a finding)
    propagates as a normal 500.
    """
    settings = get_settings()
    result = await audit_document_lifecycle(db, document_id, settings, file_storage, vector_store)

    return DocumentLifecycleAuditResponse(
        document_id=result.document_id,
        overall_status=result.overall_status,
        classification=classify_document_audit(result.overall_status, result.findings),
        issues=[_finding_response(finding) for finding in result.findings],
        database=_database_response(result),
        file_storage=_file_storage_state_response(result),
        vector_store=_vector_store_state_response(result),
    )


@router.get(
    "/reconciliation/collections/{collection_name}/report", response_model=CollectionReportResponse
)
async def collection_reconciliation_report_route(
    collection_name: str,
    db: AsyncSession = Depends(get_db_session),
    vector_store: VectorStore = Depends(get_vector_store),
) -> CollectionReportResponse:
    """Read-only consistency report for exactly one collection.

    Delegates entirely to `build_collection_reconciliation_report()`, exactly once. Always 200,
    including when the collection doesn't exist at all (`classification=missing`, `exists=false`,
    `actual_vector_count=0`) — a diagnostic snapshot, never a 404. 400 for a malformed
    `collection_name`. Any unexpected exception (an unreachable Qdrant, a database failure)
    propagates as a normal 500 — this report never fabricates a count when its one Qdrant call
    fails, see the service module's own docstring for why.
    """
    settings = get_settings()
    try:
        report = await build_collection_reconciliation_report(db, collection_name, settings, vector_store)
    except InvalidCollectionNameError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return CollectionReportResponse(
        collection_name=report.collection_name,
        classification=report.classification,
        exists=report.exists,
        is_active=report.is_active,
        index_collection_status=report.index_collection_status,
        embedding_provider=report.embedding_provider,
        embedding_model=report.embedding_model,
        embedding_dimension=report.embedding_dimension,
        embedding_version=report.embedding_version,
        chunking_version=report.chunking_version,
        document_count=report.document_count,
        expected_vector_count=report.expected_vector_count,
        actual_vector_count=report.actual_vector_count,
        difference=report.difference,
        issues=[_collection_finding_response(finding) for finding in report.findings],
        generated_at=report.generated_at,
    )
