"""Read-only, bounded, single-document lifecycle audit (Phase 2.8.7, subtask 1).

`audit_document_lifecycle()` inspects one document's state across PostgreSQL, Object Storage, and
Qdrant, and classifies whether it is currently consistent — it never mutates anything, never
repairs anything, and never discovers ownership from external systems. PostgreSQL remains the
lifecycle authority throughout: Object Storage/Qdrant are inspected only through resources
PostgreSQL already claims to own (a persisted `storage_key`, a persisted `collection_name`), never
by scanning a bucket or a Qdrant collection list for orphans. See "Deferred" below for what this
subtask deliberately does not attempt.

## Why this module lives outside both `documents/` and `indexing/`

An audit spans both domains — it reads `IngestionJob`/`DocumentDeletionJob` (owned by
`app.services.documents.*`) and `ReindexJob`/`VectorCleanupJob` (owned by
`app.services.indexing.*`). CLAUDE.md's dependency rule is one-directional and specific
(`indexing/*` must never import from `documents/*`); a new sibling package importing from *both* is
not a violation of that rule — it is the same relationship `app/api/v1/routes/*` already has to
both packages. This module therefore imports the existing public lookup helpers directly
(`get_document`, `get_latest_ingestion_job`, `get_latest_deletion_job`, `get_latest_reindex_job`,
`get_pending_cleanup_jobs`) rather than duplicating any of them locally — unlike `indexing/*`
modules, which duplicate tiny lookups specifically to avoid an illegal reverse import, this module
has no such constraint to work around.

## Dependency-failure handling

A `StorageError`/`QdrantVectorStoreError` raised while inspecting Object Storage/Qdrant is never
treated as "the resource is missing" — it becomes its own finding
(`STORAGE_INSPECTION_UNAVAILABLE`/`VECTOR_INSPECTION_UNAVAILABLE`) with a fixed, sanitized message;
the raw provider exception text is never stored in a finding. A dependency failure is not proof of
absence.

## Deferred (explicitly out of scope for this subtask)

Repair, automatic retry, stale-job recovery, orphan object/vector discovery, storage-bucket or
Qdrant-collection scanning, collection retirement, batch audit, a reconciliation worker/scheduler,
a public API, and any destructive action. `STALE_REINDEX_JOB` is also deferred: no approved stale
threshold exists yet for `ReindexJob` (only `IngestionJob` has one,
`Settings.ingestion_stale_after_seconds`) — inventing one here would be exactly the kind of
un-asked-for scope this subtask must avoid, so a `ReindexJob` stuck in `PROCESSING` is not (yet)
flagged as stale.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.models.document_deletion_job import DocumentDeletionJob, DocumentDeletionStatus
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.models.reindex_job import ReindexJobStatus
from app.rag.providers.qdrant_vector_store import QdrantVectorStoreError
from app.rag.providers.vector_store import VectorStore
from app.services.documents.deletion_service import get_latest_deletion_job
from app.services.documents.query_service import get_document, get_latest_ingestion_job
from app.services.indexing.cleanup_job_service import get_pending_cleanup_jobs
from app.services.indexing.reindex_scheduling_service import get_latest_reindex_job
from app.storage.contract import FileStorage
from app.storage.errors import StorageError
from app.storage.keys import resolve_document_storage_key

_BLOCKING_DELETION_STATUSES_FOR_EXTERNAL_CHECKS = (DocumentDeletionStatus.COMPLETED,)

_STORAGE_UNAVAILABLE_MESSAGE = (
    "Object Storage could not be inspected; this is not proof the object is absent."
)
_VECTOR_UNAVAILABLE_MESSAGE = (
    "Qdrant could not be inspected; this is not proof the collection/vectors are absent."
)


class AuditOverallStatus(StrEnum):
    """Top-level classification of one audit run."""

    NOT_FOUND = "not_found"
    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"


class FindingSeverity(StrEnum):
    """A finding's severity — distinguishes healthy/transitional/degraded/inconsistent.

    `INFO`: a valid, expected transitional state (e.g. an in-progress job, a built-but-not-yet-
    activated re-index target) — operationally normal, not corruption.
    `WARNING`: incomplete persisted work that is not itself corruption but warrants operator
    attention (a stale job, an incomplete deletion, an unresolved cleanup obligation).
    `ERROR`: a genuine inconsistency — persisted state claims something external systems
    contradict (a missing object/collection/vectors, incomplete index metadata after a completed
    ingestion). Any `ERROR` finding makes the overall result `INCONSISTENT`.
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class DocumentLifecycleFindingCode(StrEnum):
    """Closed vocabulary of every finding this audit can report — see each docstring above."""

    DOCUMENT_MISSING = "document_missing"
    OBJECT_MISSING = "object_missing"
    STORAGE_INSPECTION_UNAVAILABLE = "storage_inspection_unavailable"
    ACTIVE_COLLECTION_MISSING = "active_collection_missing"
    ACTIVE_VECTORS_MISSING = "active_vectors_missing"
    VECTOR_INSPECTION_UNAVAILABLE = "vector_inspection_unavailable"
    INDEX_METADATA_INCOMPLETE = "index_metadata_incomplete"
    INGESTION_IN_PROGRESS = "ingestion_in_progress"
    STALE_INGESTION_JOB = "stale_ingestion_job"
    DELETION_PENDING = "deletion_pending"
    DELETION_PROCESSING = "deletion_processing"
    DELETION_PARTIALLY_FAILED = "deletion_partially_failed"
    VECTOR_CLEANUP_INCOMPLETE = "vector_cleanup_incomplete"
    REINDEX_TARGET_BUILT_NOT_ACTIVATED = "reindex_target_built_not_activated"
    REINDEX_CLEANUP_PENDING = "reindex_cleanup_pending"


@dataclass(frozen=True)
class DocumentLifecycleFinding:
    """One bounded, operationally-safe-to-display audit finding.

    Never includes a stack trace, credential, storage-provider internal, or raw provider
    exception — `expected_state`/`actual_state`/`suggested_action` are always short, fixed,
    human-readable strings.
    """

    code: DocumentLifecycleFindingCode
    severity: FindingSeverity
    summary: str
    expected_state: str
    actual_state: str
    suggested_action: str
    destructive_risk: bool


@dataclass(frozen=True)
class PostgresLifecycleState:
    """A bounded snapshot of the PostgreSQL-side lifecycle state this audit inspected."""

    collection_name: str | None
    latest_ingestion_status: IngestionStatus | None
    latest_deletion_status: DocumentDeletionStatus | None
    latest_reindex_status: ReindexJobStatus | None
    latest_reindex_activated: bool
    pending_cleanup_collections: tuple[str, ...]


@dataclass(frozen=True)
class StorageLifecycleState:
    """A bounded snapshot of what this audit observed in Object Storage."""

    inspected: bool
    exists: bool | None  # None when inspection was unavailable


@dataclass(frozen=True)
class VectorLifecycleState:
    """A bounded snapshot of what this audit observed in Qdrant."""

    inspected: bool
    collection_exists: bool | None  # None when inspection was unavailable
    has_vectors: bool | None  # None when inspection was unavailable or the collection is missing


@dataclass(frozen=True)
class DocumentLifecycleAuditResult:
    """Typed outcome of audit_document_lifecycle(). Never mutates state; requires no commit."""

    document_id: str
    overall_status: AuditOverallStatus
    findings: tuple[DocumentLifecycleFinding, ...]
    postgres_state: PostgresLifecycleState | None
    storage_state: StorageLifecycleState | None
    vector_state: VectorLifecycleState | None


def _not_found_result(document_id: str) -> DocumentLifecycleAuditResult:
    finding = DocumentLifecycleFinding(
        code=DocumentLifecycleFindingCode.DOCUMENT_MISSING,
        severity=FindingSeverity.ERROR,
        summary="No document row exists for this id.",
        expected_state="A Document row exists.",
        actual_state="No Document row was found.",
        suggested_action="Verify the document id; no further inspection was performed.",
        destructive_risk=False,
    )
    return DocumentLifecycleAuditResult(
        document_id=document_id,
        overall_status=AuditOverallStatus.NOT_FOUND,
        findings=(finding,),
        postgres_state=None,
        storage_state=None,
        vector_state=None,
    )


def _is_ingestion_job_stale(job: IngestionJob, *, stale_after_seconds: int, now: datetime) -> bool:
    """A PROCESSING job is stale if its row hasn't been updated within the stale threshold.

    Mirrors `app.services.ingestion.retry_service`'s own `_is_stale_processing` comparison exactly
    — re-derived here rather than imported, since that helper is a private, non-exported name.
    """
    updated_at = job.updated_at if job.updated_at.tzinfo is not None else job.updated_at.replace(tzinfo=UTC)
    return (now - updated_at).total_seconds() > stale_after_seconds


def _deletion_finding(job: DocumentDeletionJob) -> DocumentLifecycleFinding | None:
    if job.status == DocumentDeletionStatus.PENDING:
        code = DocumentLifecycleFindingCode.DELETION_PENDING
        summary = "A deletion has been requested but not yet started."
    elif job.status == DocumentDeletionStatus.PROCESSING:
        code = DocumentLifecycleFindingCode.DELETION_PROCESSING
        summary = "Deletion is currently in progress."
    elif job.status == DocumentDeletionStatus.PARTIALLY_FAILED:
        code = DocumentLifecycleFindingCode.DELETION_PARTIALLY_FAILED
        summary = "Deletion partially completed; some resources may remain."
    else:
        return None

    return DocumentLifecycleFinding(
        code=code,
        severity=FindingSeverity.WARNING,
        summary=summary,
        expected_state="Deletion lifecycle is either not started or fully completed.",
        actual_state=f"Latest deletion job status is {job.status.value}.",
        suggested_action="Allow the existing deletion worker to process this job, or inspect it "
        "via the deletion status endpoint; this audit never retries deletion.",
        destructive_risk=False,
    )


async def audit_document_lifecycle(
    session: AsyncSession,
    document_id: str,
    settings: Settings,
    file_storage: FileStorage,
    vector_store: VectorStore,
) -> DocumentLifecycleAuditResult:
    """Audit one document's lifecycle across PostgreSQL, Object Storage, and Qdrant.

    Read-only: never mutates `Document`/`IngestionJob`/`DocumentDeletionJob`/`ReindexJob`/
    `VectorCleanupJob`, never calls a mutating storage/vector-store method, and never commits.
    External systems are inspected only through resources PostgreSQL already references — this
    function never scans a storage bucket or lists Qdrant collections looking for orphans.
    """
    document = await get_document(session, document_id)
    if document is None:
        return _not_found_result(document_id)

    findings: list[DocumentLifecycleFinding] = []

    latest_ingestion = await get_latest_ingestion_job(session, document_id)
    latest_deletion = await get_latest_deletion_job(session, document_id)
    latest_reindex = await get_latest_reindex_job(session, document_id)
    pending_cleanup_jobs = await get_pending_cleanup_jobs(session, document_id=document_id)

    deletion_completed = (
        latest_deletion is not None
        and latest_deletion.status in _BLOCKING_DELETION_STATUSES_FOR_EXTERNAL_CHECKS
    )

    if latest_deletion is not None:
        deletion_finding = _deletion_finding(latest_deletion)
        if deletion_finding is not None:
            findings.append(deletion_finding)

    # --- ingestion lifecycle -------------------------------------------------------------------
    if latest_ingestion is not None and latest_ingestion.status == IngestionStatus.COMPLETED:
        index_metadata_complete = all(
            value is not None
            for value in (
                document.collection_name,
                document.embedding_provider,
                document.embedding_model,
                document.embedding_version,
                document.chunking_version,
            )
        )
        if not index_metadata_complete:
            findings.append(
                DocumentLifecycleFinding(
                    code=DocumentLifecycleFindingCode.INDEX_METADATA_INCOMPLETE,
                    severity=FindingSeverity.ERROR,
                    summary="Ingestion completed but active index metadata is incomplete.",
                    expected_state="collection_name and embedding_* metadata are all set.",
                    actual_state="One or more active index metadata fields are null.",
                    suggested_action="Inspect the document's indexing history; do not re-index "
                    "automatically from this finding alone.",
                    destructive_risk=False,
                )
            )
    elif latest_ingestion is not None and latest_ingestion.status == IngestionStatus.PROCESSING:
        if _is_ingestion_job_stale(
            latest_ingestion,
            stale_after_seconds=settings.ingestion_stale_after_seconds,
            now=datetime.now(UTC),
        ):
            findings.append(
                DocumentLifecycleFinding(
                    code=DocumentLifecycleFindingCode.STALE_INGESTION_JOB,
                    severity=FindingSeverity.WARNING,
                    summary="The active ingestion job has not progressed within the stale threshold.",
                    expected_state=f"A PROCESSING job updates within "
                    f"{settings.ingestion_stale_after_seconds}s.",
                    actual_state="The job has not been updated within that threshold.",
                    suggested_action="Consider running the existing stale-ingestion recovery "
                    "script; this audit never recovers it automatically.",
                    destructive_risk=False,
                )
            )
        else:
            findings.append(
                DocumentLifecycleFinding(
                    code=DocumentLifecycleFindingCode.INGESTION_IN_PROGRESS,
                    severity=FindingSeverity.INFO,
                    summary="Ingestion is actively in progress.",
                    expected_state="A PROCESSING job resolves to COMPLETED or FAILED.",
                    actual_state="The job is PROCESSING and within the stale threshold.",
                    suggested_action="No action needed; this is a normal transitional state.",
                    destructive_risk=False,
                )
            )
    elif latest_ingestion is not None and latest_ingestion.status == IngestionStatus.PENDING:
        findings.append(
            DocumentLifecycleFinding(
                code=DocumentLifecycleFindingCode.INGESTION_IN_PROGRESS,
                severity=FindingSeverity.INFO,
                summary="Ingestion is queued but not yet started.",
                expected_state="A PENDING job is eventually claimed by the ingestion worker.",
                actual_state="The job is PENDING.",
                suggested_action="No action needed; this is a normal transitional state.",
                destructive_risk=False,
            )
        )

    # --- re-index lifecycle ----------------------------------------------------------------------
    reindex_cleanup_collections: set[str] = set()
    if latest_reindex is not None and latest_reindex.status == ReindexJobStatus.COMPLETED:
        if latest_reindex.activated_at is None:
            findings.append(
                DocumentLifecycleFinding(
                    code=DocumentLifecycleFindingCode.REINDEX_TARGET_BUILT_NOT_ACTIVATED,
                    severity=FindingSeverity.INFO,
                    summary="A re-index target was built successfully but has not been activated.",
                    expected_state="Either activation happens, or the build remains pending review.",
                    actual_state=f"Target {latest_reindex.target_collection_name!r} is built; "
                    "the document still serves its prior collection.",
                    suggested_action="Activate explicitly via the existing re-index activation "
                    "path when ready; this audit never activates automatically.",
                    destructive_risk=False,
                )
            )
        else:
            reindex_cleanup_collections.add(latest_reindex.source_collection_name)
            if any(
                job.collection_name == latest_reindex.source_collection_name
                for job in pending_cleanup_jobs
            ):
                findings.append(
                    DocumentLifecycleFinding(
                        code=DocumentLifecycleFindingCode.REINDEX_CLEANUP_PENDING,
                        severity=FindingSeverity.WARNING,
                        summary="Re-index activated; historical vector cleanup has not yet succeeded.",
                        expected_state="The vacated source collection's vectors are eventually cleaned up.",
                        actual_state=f"Cleanup for {latest_reindex.source_collection_name!r} is "
                        "still pending or failed.",
                        suggested_action="The document may still be serving correctly from the "
                        "target; allow the existing cleanup worker to retry, or inspect the "
                        "cleanup job directly.",
                        destructive_risk=False,
                    )
                )

    for cleanup_job in pending_cleanup_jobs:
        if cleanup_job.collection_name in reindex_cleanup_collections:
            continue  # already reported as REINDEX_CLEANUP_PENDING above — avoid duplicate noise
        findings.append(
            DocumentLifecycleFinding(
                code=DocumentLifecycleFindingCode.VECTOR_CLEANUP_INCOMPLETE,
                severity=FindingSeverity.WARNING,
                summary="A historical-vector cleanup job remains unresolved.",
                expected_state=f"Cleanup for {cleanup_job.collection_name!r} eventually succeeds.",
                actual_state=f"Cleanup job status is {cleanup_job.status.value}.",
                suggested_action="Allow the existing cleanup worker to retry; this audit never "
                "executes or creates cleanup work.",
                destructive_risk=False,
            )
        )

    # --- Object Storage --------------------------------------------------------------------------
    storage_state: StorageLifecycleState | None = None
    if not deletion_completed:
        key = resolve_document_storage_key(document)
        try:
            object_exists = await file_storage.exists(key)
        except StorageError:
            storage_state = StorageLifecycleState(inspected=False, exists=None)
            findings.append(
                DocumentLifecycleFinding(
                    code=DocumentLifecycleFindingCode.STORAGE_INSPECTION_UNAVAILABLE,
                    severity=FindingSeverity.WARNING,
                    summary="Object Storage could not be inspected.",
                    expected_state="Object Storage is reachable.",
                    actual_state=_STORAGE_UNAVAILABLE_MESSAGE,
                    suggested_action="Retry the audit once Object Storage is reachable.",
                    destructive_risk=False,
                )
            )
        else:
            storage_state = StorageLifecycleState(inspected=True, exists=object_exists)
            if not object_exists:
                findings.append(
                    DocumentLifecycleFinding(
                        code=DocumentLifecycleFindingCode.OBJECT_MISSING,
                        severity=FindingSeverity.ERROR,
                        summary="The document's original object is missing from Object Storage.",
                        expected_state="An object exists at the document's persisted storage key.",
                        actual_state="No object was found at that key.",
                        suggested_action="Inspect manually; this audit never deletes or restores "
                        "objects.",
                        destructive_risk=False,
                    )
                )

    # --- Qdrant --------------------------------------------------------------------------------
    vector_state: VectorLifecycleState | None = None
    if document.collection_name is not None and not deletion_completed:
        collection_name = document.collection_name
        try:
            vector_size = await vector_store.get_collection_vector_size(collection_name)
        except QdrantVectorStoreError:
            vector_state = VectorLifecycleState(inspected=False, collection_exists=None, has_vectors=None)
            findings.append(
                DocumentLifecycleFinding(
                    code=DocumentLifecycleFindingCode.VECTOR_INSPECTION_UNAVAILABLE,
                    severity=FindingSeverity.WARNING,
                    summary="Qdrant could not be inspected.",
                    expected_state="Qdrant is reachable.",
                    actual_state=_VECTOR_UNAVAILABLE_MESSAGE,
                    suggested_action="Retry the audit once Qdrant is reachable.",
                    destructive_risk=False,
                )
            )
        else:
            if vector_size is None:
                vector_state = VectorLifecycleState(inspected=True, collection_exists=False, has_vectors=None)
                findings.append(
                    DocumentLifecycleFinding(
                        code=DocumentLifecycleFindingCode.ACTIVE_COLLECTION_MISSING,
                        severity=FindingSeverity.ERROR,
                        summary="The document's active collection does not exist in Qdrant.",
                        expected_state=f"Collection {collection_name!r} exists.",
                        actual_state="No such collection was found.",
                        suggested_action="Inspect manually; this document cannot currently serve "
                        "retrieval from its claimed active collection.",
                        destructive_risk=False,
                    )
                )
            else:
                try:
                    vector_count = await vector_store.count_document_vectors(collection_name, document.id)
                except QdrantVectorStoreError:
                    vector_state = VectorLifecycleState(
                        inspected=False, collection_exists=True, has_vectors=None
                    )
                    findings.append(
                        DocumentLifecycleFinding(
                            code=DocumentLifecycleFindingCode.VECTOR_INSPECTION_UNAVAILABLE,
                            severity=FindingSeverity.WARNING,
                            summary="Qdrant could not be inspected.",
                            expected_state="Qdrant is reachable.",
                            actual_state=_VECTOR_UNAVAILABLE_MESSAGE,
                            suggested_action="Retry the audit once Qdrant is reachable.",
                            destructive_risk=False,
                        )
                    )
                else:
                    has_vectors = vector_count > 0
                    vector_state = VectorLifecycleState(
                        inspected=True, collection_exists=True, has_vectors=has_vectors
                    )
                    if not has_vectors:
                        findings.append(
                            DocumentLifecycleFinding(
                                code=DocumentLifecycleFindingCode.ACTIVE_VECTORS_MISSING,
                                severity=FindingSeverity.ERROR,
                                summary="No vectors exist for this document in its active collection.",
                                expected_state="At least one vector exists for this document.",
                                actual_state="Zero vectors were found.",
                                suggested_action="Inspect manually; this document cannot currently "
                                "be retrieved despite claiming an active index.",
                                destructive_risk=False,
                            )
                        )

    postgres_state = PostgresLifecycleState(
        collection_name=document.collection_name,
        latest_ingestion_status=latest_ingestion.status if latest_ingestion is not None else None,
        latest_deletion_status=latest_deletion.status if latest_deletion is not None else None,
        latest_reindex_status=latest_reindex.status if latest_reindex is not None else None,
        latest_reindex_activated=latest_reindex is not None and latest_reindex.activated_at is not None,
        pending_cleanup_collections=tuple(job.collection_name for job in pending_cleanup_jobs),
    )

    overall_status = (
        AuditOverallStatus.INCONSISTENT
        if any(finding.severity == FindingSeverity.ERROR for finding in findings)
        else AuditOverallStatus.CONSISTENT
    )

    return DocumentLifecycleAuditResult(
        document_id=document_id,
        overall_status=overall_status,
        findings=tuple(findings),
        postgres_state=postgres_state,
        storage_state=storage_state,
        vector_state=vector_state,
    )


__all__ = [
    "AuditOverallStatus",
    "DocumentLifecycleAuditResult",
    "DocumentLifecycleFinding",
    "DocumentLifecycleFindingCode",
    "FindingSeverity",
    "PostgresLifecycleState",
    "StorageLifecycleState",
    "VectorLifecycleState",
    "audit_document_lifecycle",
]
