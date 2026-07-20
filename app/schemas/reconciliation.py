"""Response schemas for the read-only reconciliation reporting API: batch document audit
(Phase 2.8.7, subtask 3), single-document audit, and collection report (both subtask 5).

Every schema field maps directly onto the reconciliation service layer's own result types
(`document_audit_service.DocumentLifecycleAuditResult`/`DocumentLifecycleFinding`,
`document_audit_batch_service.DocumentAuditSummary`/`DocumentLifecycleAuditBatchResult`,
`collection_reconciliation_report_service.CollectionReconciliationReport`/
`CollectionReportFinding`) — this module adds no new lifecycle classification or finding taxonomy
of its own; it only shapes each existing service result for the HTTP boundary. Every enum
(`AuditOverallStatus`/`FindingSeverity`/`DocumentLifecycleFindingCode`/
`DocumentAuditClassification`/`CollectionReportClassification`/`CollectionReportFindingCode`/
`IndexCollectionStatus`) is imported directly from its owning service/model module rather than
mirrored here, since none of those modules import back from `app/schemas/*` — no cycle risk,
unlike `DocumentLifecycleStatus` in `app/schemas/documents.py`.
"""

from datetime import datetime

from pydantic import BaseModel

from app.models.document_deletion_job import DocumentDeletionStatus
from app.models.index_collection import IndexCollectionStatus
from app.models.ingestion_job import IngestionStatus
from app.models.reindex_job import ReindexJobStatus
from app.services.reconciliation.collection_reconciliation_report_service import (
    CollectionReportClassification,
    CollectionReportFindingCode,
)
from app.services.reconciliation.document_audit_batch_service import DocumentAuditClassification
from app.services.reconciliation.document_audit_service import (
    AuditOverallStatus,
    DocumentLifecycleFindingCode,
    FindingSeverity,
)


class DocumentAuditFindingResponse(BaseModel):
    """One finding from the single-document auditor, unchanged — see `DocumentLifecycleFinding`.

    Never includes a stack trace, credential, or raw provider exception — the underlying service
    finding already guarantees this; this schema just exposes its existing fields as-is.
    """

    code: DocumentLifecycleFindingCode
    severity: FindingSeverity
    summary: str
    expected_state: str
    actual_state: str
    suggested_action: str
    destructive_risk: bool


class DocumentBatchAuditItemResponse(BaseModel):
    """One document's audit outcome within a batch page — see `DocumentAuditSummary`.

    `classification` is copied verbatim from the service's own `DocumentAuditSummary.
    classification` — the same bucket already counted into `summary` below, never recomputed here.
    """

    document_id: str
    original_filename: str
    created_at: datetime
    overall_status: AuditOverallStatus
    classification: DocumentAuditClassification
    issues: list[DocumentAuditFindingResponse]


class DocumentBatchAuditSummaryResponse(BaseModel):
    """Aggregate counts for one batch page — field names mirror
    `DocumentLifecycleAuditBatchResult`'s own count fields and `DocumentAuditClassification`'s
    bucket names exactly; never recalculated independently of the service's own counters."""

    total: int
    consistent: int
    transitional: int
    warning: int
    inconsistent: int
    not_found: int
    dependency_unavailable: int
    finding_counts: dict[DocumentLifecycleFindingCode, int]


class DocumentBatchAuditResponse(BaseModel):
    """Shape returned by GET /api/v1/reconciliation/documents/audit.

    `next_cursor` is opaque to API consumers — pass it back verbatim as the `cursor` query
    parameter to continue; `null` means the page returned is the last one. `limit` echoes back the
    limit actually applied for this page (after service-side validation).
    """

    items: list[DocumentBatchAuditItemResponse]
    summary: DocumentBatchAuditSummaryResponse
    limit: int
    next_cursor: str | None


# --- single-document audit (Phase 2.8.7, subtask 5) ---------------------------------------------


class DocumentDatabaseStateResponse(BaseModel):
    """PostgreSQL-side state — see `document_audit_service.PostgresLifecycleState`.

    `document_exists=False` means every other field is `None`/empty; the audit performed no
    further inspection past confirming the `Document` row itself is missing.
    """

    document_exists: bool
    collection_name: str | None
    document_created_at: datetime | None
    latest_ingestion_status: IngestionStatus | None
    latest_deletion_status: DocumentDeletionStatus | None
    latest_reindex_status: ReindexJobStatus | None
    latest_reindex_activated: bool
    pending_cleanup_collections: list[str]


class DocumentFileStorageStateResponse(BaseModel):
    """Object Storage state — see `document_audit_service.StorageLifecycleState`.

    `source_file_exists=None` means inspection was unavailable (a `STORAGE_INSPECTION_UNAVAILABLE`
    finding accompanies it) — never proof the object is actually absent.
    """

    inspected: bool
    source_file_exists: bool | None


class DocumentVectorStoreStateResponse(BaseModel):
    """Qdrant state — see `document_audit_service.VectorLifecycleState`.

    `vector_count`/`has_vectors`/`collection_exists` are all `None` when inspection was
    unavailable (a `VECTOR_INSPECTION_UNAVAILABLE` finding accompanies it) — never proof of
    absence.
    """

    inspected: bool
    collection_name: str | None
    collection_exists: bool | None
    has_vectors: bool | None
    vector_count: int | None


class DocumentLifecycleAuditResponse(BaseModel):
    """Shape returned by GET /api/v1/reconciliation/documents/{document_id}/audit.

    Always `200`, including when the document doesn't exist at all — `classification=not_found`
    and `database.document_exists=false` represent that case, matching how the batch endpoint
    already represents a missing document as data, never as an HTTP error.
    """

    document_id: str
    overall_status: AuditOverallStatus
    classification: DocumentAuditClassification
    issues: list[DocumentAuditFindingResponse]
    database: DocumentDatabaseStateResponse
    file_storage: DocumentFileStorageStateResponse | None
    vector_store: DocumentVectorStoreStateResponse | None


# --- collection reconciliation report (Phase 2.8.7, subtask 5) ------------------------------------


class CollectionReportFindingResponse(BaseModel):
    """One finding from the collection report — see `CollectionReportFinding`."""

    code: CollectionReportFindingCode
    severity: FindingSeverity
    summary: str
    expected_state: str
    actual_state: str


class CollectionReportResponse(BaseModel):
    """Shape returned by GET /api/v1/reconciliation/collections/{collection_name}/report.

    Always `200`, including when the collection doesn't exist at all (`classification=missing`,
    `exists=false`, `actual_vector_count=0`) — a diagnostic snapshot, never a `404`.
    `expected_vector_count` is a document-count-based proxy, not a tracked chunk count — see
    `collection_reconciliation_report_service`'s module docstring for exactly why, and why only a
    *deficit* (`difference` negative) is ever treated as inconsistent.
    """

    collection_name: str
    classification: CollectionReportClassification
    exists: bool
    is_active: bool
    index_collection_status: IndexCollectionStatus | None
    embedding_provider: str | None
    embedding_model: str | None
    embedding_dimension: int | None
    embedding_version: str | None
    chunking_version: str | None
    document_count: int
    expected_vector_count: int
    actual_vector_count: int
    difference: int
    issues: list[CollectionReportFindingResponse]
    generated_at: datetime
