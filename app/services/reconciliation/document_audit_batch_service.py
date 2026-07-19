"""Bounded, read-only batch lifecycle audit over one page of documents (Phase 2.8.7, subtask 2).

`audit_document_lifecycle_batch()` selects a deterministic bounded page of PostgreSQL-owned
`Document` rows (keyset-paginated, ascending `created_at`/`id`), audits each one via the existing
`audit_document_lifecycle()`, and aggregates the results into typed summary counts plus a
continuation cursor. It reuses the single-document auditor's result contract unchanged
(`DocumentLifecycleAuditResult`/`DocumentLifecycleFinding`/`DocumentLifecycleFindingCode`/
`AuditOverallStatus`/`FindingSeverity`) and introduces no new finding codes. The one addition on
top of that contract is `DocumentAuditClassification` — the same five triage buckets the aggregate
counts already used internally, now also stamped onto each `DocumentAuditSummary` via `_classify()`
so a consumer (Phase 2.8.7 subtask 3's read-only API included) never needs to re-derive it.

## Why sequential, not concurrent

`audit_document_lifecycle()` takes one `AsyncSession`; that same session is shared across every
document audited here, and concurrent use of a single `AsyncSession` is unsafe. Documents are
therefore audited one at a time, in page order. Given `limit <= MAX_BATCH_LIMIT`, this bounded
N+1 query pattern is accepted for what is currently an on-demand diagnostic operation, not a hot
request path.

## Cursor contract

The cursor is a URL-safe Base64-encoded JSON payload of `{"created_at": ..., "id": ...}` — it
encodes keyset-pagination position only, not a security boundary. It requires no encryption or
signing. A malformed cursor always raises `InvalidAuditCursorError`; there is no silent fallback
to the first page.

## Deferred (explicitly out of scope for this subtask)

Repair, a public/admin API, a CLI, a scheduler, external orphan discovery, storage-bucket or
Qdrant-collection scanning, persisted audit history, concurrent auditing, and a session-factory
refactor for batched lifecycle preloading.
"""

import base64
import binascii
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.models.document import Document
from app.rag.providers.vector_store import VectorStore
from app.services.reconciliation.document_audit_service import (
    AuditOverallStatus,
    DocumentLifecycleFinding,
    DocumentLifecycleFindingCode,
    FindingSeverity,
    audit_document_lifecycle,
)
from app.storage.contract import FileStorage

MIN_BATCH_LIMIT = 1
DEFAULT_BATCH_LIMIT = 20
MAX_BATCH_LIMIT = 50

# The finding codes the single-document auditor reports when Object Storage or Qdrant could not
# be inspected at all — distinct from a genuine missing-object/missing-vector finding.
_DEPENDENCY_UNAVAILABLE_CODES = frozenset(
    {
        DocumentLifecycleFindingCode.STORAGE_INSPECTION_UNAVAILABLE,
        DocumentLifecycleFindingCode.VECTOR_INSPECTION_UNAVAILABLE,
    }
)

_MAX_CURSOR_SIZE_BYTES = 2048


class InvalidAuditBatchLimitError(ValueError):
    """Raised when a requested batch `limit` falls outside [MIN_BATCH_LIMIT, MAX_BATCH_LIMIT]."""


class InvalidAuditCursorError(ValueError):
    """Raised when a supplied continuation cursor is malformed or fails validation."""


class DocumentAuditClassification(StrEnum):
    """The same five triage buckets used for this batch result's aggregate counts, exposed per
    document too — see `_classify()`, the single place both are derived from."""

    NOT_FOUND = "not_found"
    INCONSISTENT = "inconsistent"
    CONSISTENT = "consistent"
    TRANSITIONAL = "transitional"
    WARNING = "warning"


@dataclass(frozen=True)
class AuditCursor:
    """Decoded keyset-pagination position: the last-returned document's `created_at` and `id`."""

    created_at: datetime
    document_id: str


@dataclass(frozen=True)
class DocumentAuditSummary:
    """One document's audit outcome, bounded to what batch triage needs — reuses findings as-is.

    `classification` is the exact same bucket this document was counted into in the batch result's
    aggregate counts (see `_classify()`) — a consumer never needs to re-derive it from
    `overall_status`/`findings` itself.
    """

    document_id: str
    original_filename: str
    created_at: datetime
    overall_status: AuditOverallStatus
    classification: DocumentAuditClassification
    findings: tuple[DocumentLifecycleFinding, ...]


@dataclass(frozen=True)
class DocumentLifecycleAuditBatchResult:
    """Typed outcome of audit_document_lifecycle_batch(). Never mutates state; requires no commit."""

    scanned_count: int
    consistent_count: int
    transitional_count: int
    warning_count: int
    inconsistent_count: int
    not_found_count: int
    dependency_unavailable_count: int
    finding_counts: dict[DocumentLifecycleFindingCode, int]
    documents: tuple[DocumentAuditSummary, ...]
    next_cursor: str | None
    has_more: bool


def encode_audit_cursor(created_at: datetime, document_id: str) -> str:
    """Encode a keyset-pagination position as a URL-safe Base64 JSON payload."""
    payload = json.dumps(
        {"created_at": created_at.isoformat(), "id": document_id}, separators=(",", ":")
    )
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")


def decode_audit_cursor(cursor: str) -> AuditCursor:
    """Decode and validate a cursor produced by `encode_audit_cursor()`.

    Raises `InvalidAuditCursorError` for malformed Base64/JSON, a non-object payload, missing
    fields, an invalid or non-timezone-aware datetime, or an invalid document id. Never falls
    back to the first page silently.
    """
    if len(cursor) > _MAX_CURSOR_SIZE_BYTES:
        raise InvalidAuditCursorError("Cursor exceeds the maximum allowed size.")

    try:
        raw = base64.urlsafe_b64decode(cursor.encode("ascii"))
    except (binascii.Error, ValueError) as exc:
        raise InvalidAuditCursorError("Cursor is not valid URL-safe Base64.") from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidAuditCursorError("Cursor does not decode to valid JSON.") from exc

    if not isinstance(payload, dict):
        raise InvalidAuditCursorError("Cursor payload must be a JSON object.")

    if "created_at" not in payload or "id" not in payload:
        raise InvalidAuditCursorError("Cursor payload is missing required fields.")

    created_at_raw = payload["created_at"]
    document_id = payload["id"]

    if not isinstance(created_at_raw, str) or not isinstance(document_id, str):
        raise InvalidAuditCursorError("Cursor payload fields have an invalid type.")

    try:
        datetime.fromisoformat(created_at_raw)
    except ValueError as exc:
        raise InvalidAuditCursorError("Cursor created_at is not a valid ISO datetime.") from exc

    created_at = datetime.fromisoformat(created_at_raw)
    if created_at.tzinfo is None:
        raise InvalidAuditCursorError("Cursor created_at must be timezone-aware.")

    try:
        uuid.UUID(document_id)
    except ValueError as exc:
        raise InvalidAuditCursorError("Cursor id is not a valid document identifier.") from exc

    return AuditCursor(created_at=created_at, document_id=document_id)


def _validate_limit(limit: int) -> None:
    if limit < MIN_BATCH_LIMIT or limit > MAX_BATCH_LIMIT:
        raise InvalidAuditBatchLimitError(
            f"limit must be between {MIN_BATCH_LIMIT} and {MAX_BATCH_LIMIT}, got {limit}."
        )


async def _select_document_page(
    session: AsyncSession, *, limit: int, cursor: AuditCursor | None
) -> list[Document]:
    stmt = select(Document).order_by(Document.created_at.asc(), Document.id.asc()).limit(limit + 1)
    if cursor is not None:
        stmt = stmt.where(
            (Document.created_at > cursor.created_at)
            | ((Document.created_at == cursor.created_at) & (Document.id > cursor.document_id))
        )
    result = await session.execute(stmt)
    return list(result.scalars().all())


def _classify(
    overall_status: AuditOverallStatus, findings: tuple[DocumentLifecycleFinding, ...]
) -> DocumentAuditClassification:
    """Single source of truth for both `DocumentAuditSummary.classification` and the aggregate
    bucket counts below — never re-derive this rule anywhere else (including the API layer)."""
    if overall_status == AuditOverallStatus.NOT_FOUND:
        return DocumentAuditClassification.NOT_FOUND
    if overall_status == AuditOverallStatus.INCONSISTENT:
        return DocumentAuditClassification.INCONSISTENT
    if not findings:
        return DocumentAuditClassification.CONSISTENT
    if all(finding.severity == FindingSeverity.INFO for finding in findings):
        return DocumentAuditClassification.TRANSITIONAL
    return DocumentAuditClassification.WARNING


class _BatchCounters:
    """Mutable aggregation state for one batch run — internal to this module only."""

    def __init__(self) -> None:
        self.consistent_count = 0
        self.transitional_count = 0
        self.warning_count = 0
        self.inconsistent_count = 0
        self.not_found_count = 0
        self.dependency_unavailable_count = 0
        self.finding_counts: dict[DocumentLifecycleFindingCode, int] = {}

    def record(self, summary: DocumentAuditSummary) -> None:
        if summary.classification == DocumentAuditClassification.NOT_FOUND:
            self.not_found_count += 1
        elif summary.classification == DocumentAuditClassification.INCONSISTENT:
            self.inconsistent_count += 1
        elif summary.classification == DocumentAuditClassification.CONSISTENT:
            self.consistent_count += 1
        elif summary.classification == DocumentAuditClassification.TRANSITIONAL:
            self.transitional_count += 1
        else:
            self.warning_count += 1

        for finding in summary.findings:
            self.finding_counts[finding.code] = self.finding_counts.get(finding.code, 0) + 1

        if any(finding.code in _DEPENDENCY_UNAVAILABLE_CODES for finding in summary.findings):
            self.dependency_unavailable_count += 1


async def audit_document_lifecycle_batch(
    session: AsyncSession,
    settings: Settings,
    file_storage: FileStorage,
    vector_store: VectorStore,
    *,
    limit: int = DEFAULT_BATCH_LIMIT,
    cursor: str | None = None,
) -> DocumentLifecycleAuditBatchResult:
    """Audit a deterministic bounded page of documents, oldest-first, sequentially.

    Read-only: never mutates a `Document`/job row and never commits. Calls
    `audit_document_lifecycle()` exactly once per selected document, in ascending
    `created_at`/`id` order, never concurrently. Raises `InvalidAuditBatchLimitError`/
    `InvalidAuditCursorError` for invalid input; any other exception raised while auditing a
    document propagates rather than being folded into a finding.
    """
    _validate_limit(limit)
    decoded_cursor = decode_audit_cursor(cursor) if cursor is not None else None

    rows = await _select_document_page(session, limit=limit, cursor=decoded_cursor)
    has_more = len(rows) > limit
    page = rows[:limit]

    counters = _BatchCounters()
    summaries: list[DocumentAuditSummary] = []
    for document in page:
        audit = await audit_document_lifecycle(session, document.id, settings, file_storage, vector_store)
        summary = DocumentAuditSummary(
            document_id=document.id,
            original_filename=document.original_filename,
            created_at=document.created_at,
            overall_status=audit.overall_status,
            classification=_classify(audit.overall_status, audit.findings),
            findings=audit.findings,
        )
        summaries.append(summary)
        counters.record(summary)

    next_cursor = encode_audit_cursor(page[-1].created_at, page[-1].id) if has_more and page else None

    return DocumentLifecycleAuditBatchResult(
        scanned_count=len(summaries),
        consistent_count=counters.consistent_count,
        transitional_count=counters.transitional_count,
        warning_count=counters.warning_count,
        inconsistent_count=counters.inconsistent_count,
        not_found_count=counters.not_found_count,
        dependency_unavailable_count=counters.dependency_unavailable_count,
        finding_counts=counters.finding_counts,
        documents=tuple(summaries),
        next_cursor=next_cursor,
        has_more=has_more,
    )


__all__ = [
    "DEFAULT_BATCH_LIMIT",
    "MAX_BATCH_LIMIT",
    "MIN_BATCH_LIMIT",
    "AuditCursor",
    "DocumentAuditClassification",
    "DocumentAuditSummary",
    "DocumentLifecycleAuditBatchResult",
    "InvalidAuditBatchLimitError",
    "InvalidAuditCursorError",
    "audit_document_lifecycle_batch",
    "decode_audit_cursor",
    "encode_audit_cursor",
]
