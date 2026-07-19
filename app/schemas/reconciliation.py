"""Response schemas for the read-only batch document lifecycle audit API (Phase 2.8.7, subtask 3).

Every schema field maps directly onto `app.services.reconciliation.document_audit_batch_service`'s
existing `DocumentAuditSummary`/`DocumentLifecycleAuditBatchResult` (and, for findings,
`document_audit_service.DocumentLifecycleFinding`) — this module adds no new lifecycle
classification or finding taxonomy of its own; it only shapes the existing service result for the
HTTP boundary. `AuditOverallStatus`/`FindingSeverity`/`DocumentLifecycleFindingCode` and
`DocumentAuditClassification` are imported directly from the service layer rather than mirrored
here, since neither service module imports back from `app/schemas/*` — no cycle risk, unlike
`DocumentLifecycleStatus` in `app/schemas/documents.py`.
"""

from datetime import datetime

from pydantic import BaseModel

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
