"""Read-only, single-collection consistency report (Phase 2.8.7, subtask 5).

`build_collection_reconciliation_report()` answers, for one Qdrant collection name: does it
exist, is it the platform's currently active collection, how many vectors does Postgres imply it
should have, how many does Qdrant actually report, and is that a real discrepancy. It never
mutates anything, never repairs anything, and never enumerates individual points/payloads — only
aggregate counts are ever read.

## "Expected vector count" is a document-count proxy, not a tracked chunk count

No column anywhere in this schema durably tracks "how many chunks/vectors document X produced" —
chunking happens at ingestion/re-index time and only its *result* is written to Qdrant; Postgres
never persists the count. `expected_vector_count` here is therefore the number of `Document` rows
currently claiming this `collection_name` (a real, honest, cheap aggregate query) — not a claim
that every document produces exactly one vector. A document can legitimately produce zero chunks
(see ARCHITECTURE.md's "Zero-chunk behavior") or many, so `actual_vector_count` routinely exceeds
`expected_vector_count` in a perfectly healthy collection; that surplus is never itself flagged.
Only a **deficit** (`actual_vector_count < expected_vector_count`) is treated as a real signal —
fewer vectors exist than documents claim to be using this collection, mirroring exactly the same
coarse "has at least one vector" philosophy `document_audit_service.audit_document_lifecycle()`
already applies at the single-document level (see its module docstring).

## "Is this collection active" has two independent, non-identical signals

`IndexCollection.status == ACTIVE` is a per-row bookkeeping flag (never automatically exclusive —
an old collection isn't guaranteed to be flipped to `RETIRED` the moment a new one becomes
current). The platform's single currently-*desired* collection is a different, authoritative
signal: `get_active_embedding_config(settings).collection_name` — the same comparison
`collection_registry.is_document_stale()`/`reindex_scheduling_service.schedule_reindex()` already
use everywhere else in this codebase. This report exposes both: `is_active` (desired-collection
match — the operationally meaningful one) and `index_collection_status` (the row's own bookkeeping
flag, or `None` if no `IndexCollection` row exists at all).

## Unexpected dependency failures are never downgraded to a fabricated report

Unlike the single-document auditor (which turns a Qdrant failure into a WARNING finding, since a
document-level audit has other signals worth reporting even when Qdrant is unreachable), this
collection report has no other useful signal to fall back on if the one aggregate Qdrant call
fails — so `QdrantVectorStoreError` is deliberately left to propagate here, never caught, never
turned into a fabricated `actual_vector_count=0`. The router lets it become a normal 500.
"""

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.models.document import Document
from app.models.index_collection import IndexCollection, IndexCollectionStatus
from app.rag.embedding_config import get_active_embedding_config
from app.rag.providers.vector_store import VectorStore
from app.services.reconciliation.document_audit_service import FindingSeverity

# The exact charset app.rag.embedding_config._sanitize() ever produces for a real collection name
# — this project never generates a name outside it, so rejecting anything else is not inventing a
# competing rule, only enforcing the one already implied by how names are actually constructed.
_COLLECTION_NAME_PATTERN = re.compile(r"^[a-z0-9_-]{1,255}$")


class InvalidCollectionNameError(ValueError):
    """Raised when a supplied collection_name is empty or contains a disallowed character."""


class CollectionReportClassification(StrEnum):
    """Top-level classification of one collection reconciliation report."""

    HEALTHY = "healthy"
    INCONSISTENT = "inconsistent"
    MISSING = "missing"
    UNMANAGED = "unmanaged"


class CollectionReportFindingCode(StrEnum):
    """Closed vocabulary of every finding this report can produce."""

    COLLECTION_MISSING = "collection_missing"
    COLLECTION_UNMANAGED = "collection_unmanaged"
    VECTOR_COUNT_DEFICIT = "vector_count_deficit"


@dataclass(frozen=True)
class CollectionReportFinding:
    """One bounded, operationally-safe-to-display finding — mirrors
    `document_audit_service.DocumentLifecycleFinding`'s shape exactly, for the same reason."""

    code: CollectionReportFindingCode
    severity: FindingSeverity
    summary: str
    expected_state: str
    actual_state: str


@dataclass(frozen=True)
class CollectionReconciliationReport:
    """Typed outcome of build_collection_reconciliation_report(). Never mutates state."""

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
    findings: tuple[CollectionReportFinding, ...]
    generated_at: datetime


def validate_collection_name(collection_name: str) -> None:
    """Raise InvalidCollectionNameError unless collection_name matches the platform's own
    generated charset — rejects empty values, path separators, and control characters alike."""
    if not _COLLECTION_NAME_PATTERN.fullmatch(collection_name):
        raise InvalidCollectionNameError(
            "collection_name must be 1-255 characters of lowercase letters, digits, '_' or '-'."
        )


async def _document_count_for_collection(session: AsyncSession, collection_name: str) -> int:
    stmt = select(func.count()).select_from(Document).where(Document.collection_name == collection_name)
    result = await session.execute(stmt)
    return result.scalar_one()


async def build_collection_reconciliation_report(
    session: AsyncSession,
    collection_name: str,
    settings: Settings,
    vector_store: VectorStore,
) -> CollectionReconciliationReport:
    """Build a read-only consistency report for one collection.

    Raises `InvalidCollectionNameError` for a malformed `collection_name`, before any query runs.
    Any other exception (an unexpected Postgres or Qdrant failure) propagates unchanged — see the
    module docstring for why this report never downgrades that into a fabricated result.
    """
    validate_collection_name(collection_name)

    generated_at = datetime.now(UTC)
    findings: list[CollectionReportFinding] = []

    index_collection = await session.get(IndexCollection, collection_name)
    document_count = await _document_count_for_collection(session, collection_name)
    desired_collection_name = get_active_embedding_config(settings).collection_name
    is_active = collection_name == desired_collection_name

    actual_vector_count = await vector_store.count_collection_vectors(collection_name)
    exists = actual_vector_count is not None

    if not exists:
        findings.append(
            CollectionReportFinding(
                code=CollectionReportFindingCode.COLLECTION_MISSING,
                severity=FindingSeverity.ERROR,
                summary="The collection does not exist in Qdrant.",
                expected_state=f"Collection {collection_name!r} exists.",
                actual_state="No such collection was found.",
            )
        )
        return CollectionReconciliationReport(
            collection_name=collection_name,
            classification=CollectionReportClassification.MISSING,
            exists=False,
            is_active=is_active,
            index_collection_status=index_collection.status if index_collection is not None else None,
            embedding_provider=index_collection.embedding_provider if index_collection else None,
            embedding_model=index_collection.embedding_model if index_collection else None,
            embedding_dimension=index_collection.embedding_dimension if index_collection else None,
            embedding_version=index_collection.embedding_version if index_collection else None,
            chunking_version=index_collection.chunking_version if index_collection else None,
            document_count=document_count,
            expected_vector_count=document_count,
            actual_vector_count=0,
            difference=0 - document_count,
            findings=tuple(findings),
            generated_at=generated_at,
        )

    assert actual_vector_count is not None  # narrowed by `exists` above, for mypy
    difference = actual_vector_count - document_count

    if index_collection is None:
        classification = CollectionReportClassification.UNMANAGED
        findings.append(
            CollectionReportFinding(
                code=CollectionReportFindingCode.COLLECTION_UNMANAGED,
                severity=FindingSeverity.WARNING,
                summary="The collection exists in Qdrant but has no persisted IndexCollection row.",
                expected_state="Every collection this platform serves from has a persisted "
                "IndexCollection row.",
                actual_state="No IndexCollection row was found for this collection name.",
            )
        )
    elif difference < 0:
        classification = CollectionReportClassification.INCONSISTENT
        findings.append(
            CollectionReportFinding(
                code=CollectionReportFindingCode.VECTOR_COUNT_DEFICIT,
                severity=FindingSeverity.ERROR,
                summary="Fewer vectors exist than documents currently claim this collection.",
                expected_state=f"At least {document_count} vectors (one per claiming document).",
                actual_state=f"{actual_vector_count} vectors found.",
            )
        )
    else:
        classification = CollectionReportClassification.HEALTHY

    return CollectionReconciliationReport(
        collection_name=collection_name,
        classification=classification,
        exists=True,
        is_active=is_active,
        index_collection_status=index_collection.status if index_collection is not None else None,
        embedding_provider=index_collection.embedding_provider if index_collection else None,
        embedding_model=index_collection.embedding_model if index_collection else None,
        embedding_dimension=index_collection.embedding_dimension if index_collection else None,
        embedding_version=index_collection.embedding_version if index_collection else None,
        chunking_version=index_collection.chunking_version if index_collection else None,
        document_count=document_count,
        expected_vector_count=document_count,
        actual_vector_count=actual_vector_count,
        difference=difference,
        findings=tuple(findings),
        generated_at=generated_at,
    )


__all__ = [
    "CollectionReconciliationReport",
    "CollectionReportClassification",
    "CollectionReportFinding",
    "CollectionReportFindingCode",
    "InvalidCollectionNameError",
    "build_collection_reconciliation_report",
    "validate_collection_name",
]
