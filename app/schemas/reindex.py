"""Response schemas for the single-document re-index inspection/schedule/activate API
(Phase 2.8.6, subtask 6).

Every schema field traces to a real `Document`/`ReindexJob` column or a genuinely-derivable value
(see `app/services/indexing/reindex_inspection_service.py`). `ReindexAttemptResponse` never
includes the raw `ReindexJob.error_message` — only `safe_error_message`
(`reindex_inspection_service.sanitize_reindex_error`), mirroring
`IngestionFailureResponse.safe_message`/`DocumentDeletionStatusResponse.safe_message`'s exact
precedent in `app/schemas/documents.py`.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel

from app.models.reindex_job import ReindexJobStatus


class ReindexLifecycleState(StrEnum):
    """A document's derived re-index lifecycle state — see
    `app.services.indexing.reindex_inspection_service.inspect_document_reindex_state`.

    Defined here (not in the service module) so the service can import it without a cycle back
    into this schema module — mirrors `DocumentLifecycleStatus`'s exact precedent in
    `app/schemas/documents.py`. Purely derived from `ReindexJob.status`/`activated_at` plus a
    document-vs-desired-configuration comparison; never a new persisted database status (see
    CLAUDE.md's High-Risk Invariants: `ReindexJob.status` semantics are never changed for this).
    """

    NOT_INDEXED = "not_indexed"
    UP_TO_DATE = "up_to_date"
    STALE = "stale"
    REINDEX_PENDING = "reindex_pending"
    REINDEX_PROCESSING = "reindex_processing"
    TARGET_BUILT = "target_built"
    ACTIVATED = "activated"
    FAILED = "failed"
    DELETION_BLOCKED = "deletion_blocked"


class IndexConfigSnapshotResponse(BaseModel):
    """One indexing configuration snapshot — either a document's active (persisted) index, or the
    platform's currently desired index.

    `chunk_size`/`chunk_overlap` are always null on the active snapshot (`Document` does not
    persist per-document chunk settings — see `app/models/document.py`) and populated on the
    desired snapshot (read live from `Settings` at inspection time — the one legitimate use of
    live settings in this API, per this subtask's spec).
    """

    collection_name: str | None
    provider: str | None
    model: str | None
    dimension: int | None
    embedding_version: str | None
    chunking_version: str | None
    chunk_size: int | None
    chunk_overlap: int | None


class ReindexAttemptResponse(BaseModel):
    """One re-index attempt (`ReindexJob`), sanitized.

    `status=COMPLETED` with `activated_at=null` means the target build succeeded but the document
    still serves its prior collection; `activated_at` non-null means the document was explicitly
    cut over to this attempt's target. No stack trace, exception class, provider credential, or
    storage path is ever included — only `safe_error_message`.
    """

    job_id: str
    status: ReindexJobStatus
    source_collection_name: str
    target_collection_name: str
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None
    activated_at: datetime | None
    safe_error_message: str | None


class ReindexStateResponse(BaseModel):
    """Shape returned by GET /api/v1/documents/{document_id}/reindex.

    `can_schedule`/`can_activate` are best-effort hints, not guarantees — see
    `reindex_inspection_service`'s module docstring for exactly what they do and do not check. The
    corresponding `POST` endpoints remain the sole authority on whether a call actually succeeds.
    """

    document_id: str
    state: ReindexLifecycleState
    is_stale: bool
    active_index: IndexConfigSnapshotResponse
    desired_index: IndexConfigSnapshotResponse
    latest_attempt: ReindexAttemptResponse | None
    can_schedule: bool
    can_activate: bool


class ReindexScheduleResponse(BaseModel):
    """Shape returned by POST /api/v1/documents/{document_id}/reindex.

    `created=True` means a brand-new PENDING ReindexJob was inserted for the existing
    `ReindexWorker` to pick up separately; `created=False` means an already-active
    (PENDING/PROCESSING) job was returned unchanged — nothing new was scheduled. This endpoint
    never builds a target or writes a vector inline.
    """

    document_id: str
    job_id: str
    status: ReindexJobStatus
    source_collection_name: str
    target_collection_name: str
    created: bool


class ReindexActivateResponse(BaseModel):
    """Shape returned by POST /api/v1/documents/{document_id}/reindex/activate.

    `already_activated=True` means this call was an idempotent no-op against a job that had
    already been activated by an earlier call; `already_activated=False` means this call performed
    the cutover. Both cases return 200. This endpoint never executes the deferred vector cleanup
    it creates a `VectorCleanupJob` for — that remains a separate, out-of-band operation.
    """

    document_id: str
    job_id: str
    status: ReindexJobStatus
    activated_at: datetime
    already_activated: bool
