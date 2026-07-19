"""Single-document re-index inspection/schedule/activate endpoints (Phase 2.8.6, subtask 6), plus
a job-id-scoped operator activation endpoint (Phase 2.8.7, subtask 4).

Thin HTTP boundary over the already-implemented re-index lifecycle: inspection
(`reindex_inspection_service.inspect_document_reindex_state`), scheduling
(`reindex_scheduling_service.schedule_reindex`), and activation
(`reindex_activation.activate_reindexed_document`). This module never builds a target, writes a
Qdrant vector, reads/writes object storage, or executes cleanup — `POST .../reindex` only ever
inserts a PENDING `ReindexJob` row (picked up separately by `ReindexWorker`, run out-of-band, never
inline in this request), and both `POST .../reindex/activate` and
`POST /reindex/jobs/{job_id}/activate` only ever perform the metadata cutover already implemented
by `activate_reindexed_document()` — a single shared service call, never duplicated eligibility
logic. All business/lifecycle-derivation/staleness-comparison/locking logic lives in the service
modules — routes here only parse/inject/call/copy-status, per CLAUDE.md's "Route Layer Style"
(same convention as `app/api/v1/routes/documents.py`).
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.session import get_db_session
from app.rag.embedding_config import get_active_embedding_config
from app.rag.providers.provider_factory import get_vector_store as _resolve_vector_store
from app.rag.providers.vector_store import VectorStore
from app.schemas.reindex import (
    IndexConfigSnapshotResponse,
    ReindexActivateResponse,
    ReindexAttemptResponse,
    ReindexJobActivationResponse,
    ReindexScheduleResponse,
    ReindexStateResponse,
)
from app.services.indexing.reindex_activation import ReindexActivationOutcome, activate_reindexed_document
from app.services.indexing.reindex_inspection_service import (
    IndexConfigSnapshot,
    inspect_document_reindex_state,
    sanitize_reindex_error,
)
from app.services.indexing.reindex_scheduling_service import (
    ReindexSchedulingOutcome,
    get_document,
    get_latest_reindex_job,
    get_reindex_job,
    schedule_reindex,
)

router = APIRouter()


def get_vector_store() -> VectorStore:
    """Build the configured VectorStore implementation via the provider factory."""
    return _resolve_vector_store()


def _snapshot_response(snapshot: IndexConfigSnapshot) -> IndexConfigSnapshotResponse:
    return IndexConfigSnapshotResponse(
        collection_name=snapshot.collection_name,
        provider=snapshot.provider,
        model=snapshot.model,
        dimension=snapshot.dimension,
        embedding_version=snapshot.embedding_version,
        chunking_version=snapshot.chunking_version,
        chunk_size=snapshot.chunk_size,
        chunk_overlap=snapshot.chunk_overlap,
    )


@router.get("/documents/{document_id}/reindex", response_model=ReindexStateResponse)
async def get_document_reindex_state_route(
    document_id: str, db: AsyncSession = Depends(get_db_session)
) -> ReindexStateResponse:
    """Report whether a document's active index is stale, plus its latest re-index attempt.

    404 if the document does not exist. Never mutates anything — see
    `reindex_inspection_service.inspect_document_reindex_state`.
    """
    settings = get_settings()
    state = await inspect_document_reindex_state(db, document_id, settings)
    if state is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    latest_attempt = None
    if state.latest_job is not None:
        job = state.latest_job
        latest_attempt = ReindexAttemptResponse(
            job_id=job.id,
            status=job.status,
            source_collection_name=job.source_collection_name,
            target_collection_name=job.target_collection_name,
            created_at=job.created_at,
            updated_at=job.updated_at,
            completed_at=job.completed_at,
            activated_at=job.activated_at,
            safe_error_message=sanitize_reindex_error(job.error_message),
        )

    return ReindexStateResponse(
        document_id=state.document_id,
        state=state.state,
        is_stale=state.is_stale,
        active_index=_snapshot_response(state.active_index),
        desired_index=_snapshot_response(state.desired_index),
        latest_attempt=latest_attempt,
        can_schedule=state.can_schedule,
        can_activate=state.can_activate,
    )


_SCHEDULE_OUTCOME_ERRORS: dict[ReindexSchedulingOutcome, tuple[int, str]] = {
    ReindexSchedulingOutcome.INELIGIBLE_NEVER_INDEXED: (
        status.HTTP_409_CONFLICT,
        "Document has no active successful index; use ingestion retry, not re-index.",
    ),
    ReindexSchedulingOutcome.ALREADY_CURRENT: (
        status.HTTP_409_CONFLICT,
        "Document already matches the desired configuration; no re-index is needed.",
    ),
    ReindexSchedulingOutcome.INGESTION_ACTIVE: (
        status.HTTP_409_CONFLICT,
        "Document has an active ingestion job; cannot schedule a re-index until it resolves.",
    ),
    ReindexSchedulingOutcome.DELETION_ACTIVE: (
        status.HTTP_409_CONFLICT,
        "Document is being deleted; cannot schedule a re-index.",
    ),
    ReindexSchedulingOutcome.DELETION_INCOMPLETE: (
        status.HTTP_409_CONFLICT,
        "Document deletion is incomplete; cannot schedule a re-index until it is resolved.",
    ),
    ReindexSchedulingOutcome.DOCUMENT_DELETED: (
        status.HTTP_409_CONFLICT,
        "Document has been deleted; cannot schedule a re-index.",
    ),
}


@router.post("/documents/{document_id}/reindex", response_model=ReindexScheduleResponse)
async def schedule_document_reindex_route(
    document_id: str,
    response: Response,
    db: AsyncSession = Depends(get_db_session),
    vector_store: VectorStore = Depends(get_vector_store),
) -> ReindexScheduleResponse:
    """Schedule one append-only re-index attempt against the platform's current desired configuration.

    202 with `created=True` when a new PENDING ReindexJob was inserted; 200 with `created=False`
    when an already-active job was returned instead; 404 if the document does not exist; 409 for
    every other blocking condition (see `reindex_scheduling_service.schedule_reindex`'s decision
    table). Never builds a target inline — the existing `ReindexWorker` processes the PENDING job
    separately, out-of-band.
    """
    document = await get_document(db, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    settings = get_settings()
    target_config = get_active_embedding_config(settings)
    result = await schedule_reindex(
        db,
        document,
        vector_store,
        target_config,
        target_chunk_size=settings.chunk_size,
        target_chunk_overlap=settings.chunk_overlap,
    )

    if result.outcome in _SCHEDULE_OUTCOME_ERRORS:
        status_code, detail = _SCHEDULE_OUTCOME_ERRORS[result.outcome]
        raise HTTPException(status_code=status_code, detail=detail)

    assert result.job is not None
    created = result.outcome == ReindexSchedulingOutcome.CREATED
    response.status_code = status.HTTP_202_ACCEPTED if created else status.HTTP_200_OK
    return ReindexScheduleResponse(
        document_id=document_id,
        job_id=result.job.id,
        status=result.job.status,
        source_collection_name=result.job.source_collection_name,
        target_collection_name=result.job.target_collection_name,
        created=created,
    )


_ACTIVATION_OUTCOME_ERRORS: dict[ReindexActivationOutcome, tuple[int, str]] = {
    ReindexActivationOutcome.JOB_NOT_FOUND: (status.HTTP_404_NOT_FOUND, "Re-index job not found."),
    ReindexActivationOutcome.DOCUMENT_MISSING: (status.HTTP_404_NOT_FOUND, "Document not found."),
    ReindexActivationOutcome.NOT_READY: (
        status.HTTP_409_CONFLICT,
        "Re-index job has not completed a successful build yet.",
    ),
    ReindexActivationOutcome.SOURCE_CHANGED: (
        status.HTTP_409_CONFLICT,
        "Document's serving collection has changed since this re-index job was scheduled.",
    ),
    ReindexActivationOutcome.BLOCKED_BY_DELETION: (
        status.HTTP_409_CONFLICT,
        "Document deletion is active or completed; activation is not available.",
    ),
}


@router.post("/documents/{document_id}/reindex/activate", response_model=ReindexActivateResponse)
async def activate_document_reindex_route(
    document_id: str,
    response: Response,
    job_id: str | None = Query(
        default=None, description="Re-index job to activate; defaults to the document's latest attempt."
    ),
    db: AsyncSession = Depends(get_db_session),
) -> ReindexActivateResponse:
    """Explicitly activate one completed re-index attempt — metadata cutover only.

    200 for both a fresh activation and an idempotent already-activated call (see
    `already_activated` in the response); 404 if no re-index job (either the given `job_id`, or —
    if omitted — the document's latest attempt) can be found for this document; 409 for every other
    blocking condition (see `reindex_activation.activate_reindexed_document`'s decision table).
    Never builds a target, waits for a pending/processing job, or executes vector cleanup — the
    `VectorCleanupJob` activation creates is picked up by the existing cleanup machinery
    separately, out-of-band.
    """
    if job_id is not None:
        job = await get_reindex_job(db, job_id)
        if job is None or job.document_id != document_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Re-index job not found for this document.",
            )
        resolved_job_id = job_id
    else:
        latest = await get_latest_reindex_job(db, document_id)
        if latest is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No re-index attempt found for this document.",
            )
        resolved_job_id = latest.id

    result = await activate_reindexed_document(db, resolved_job_id)

    if result.outcome in _ACTIVATION_OUTCOME_ERRORS:
        status_code, detail = _ACTIVATION_OUTCOME_ERRORS[result.outcome]
        raise HTTPException(status_code=status_code, detail=detail)

    assert result.job is not None
    assert result.job.activated_at is not None
    response.status_code = status.HTTP_200_OK
    return ReindexActivateResponse(
        document_id=document_id,
        job_id=result.job.id,
        status=result.job.status,
        activated_at=result.job.activated_at,
        already_activated=result.outcome == ReindexActivationOutcome.ALREADY_ACTIVATED,
    )


@router.post("/reindex/jobs/{job_id}/activate", response_model=ReindexJobActivationResponse)
async def activate_reindex_job_route(
    job_id: str,
    response: Response,
    db: AsyncSession = Depends(get_db_session),
) -> ReindexJobActivationResponse:
    """Explicit, job-id-scoped operator activation (Phase 2.8.7, subtask 4).

    The same cutover as `POST .../documents/{document_id}/reindex/activate`, without requiring the
    caller to already know the document id — everything needed is resolved from `job_id` alone by
    `activate_reindexed_document()` itself. Delegates to that same service function exactly once;
    see its own docstring for the full deterministic precondition order (job existence ->
    already-activated -> build completion -> document existence -> deletion blocking -> source
    staleness -> target existence). 200 for both a fresh activation and an idempotent
    already-activated repeat call (see `already_activated`); 404 if the job does not exist; 409 for
    every other blocking condition (see `_ACTIVATION_OUTCOME_ERRORS`, shared with the sibling
    document-scoped route above — no duplicated outcome-to-status mapping). Never builds a target,
    waits for a pending/processing job, or executes the vector cleanup it creates — the
    `VectorCleanupJob` activation creates is picked up by the existing cleanup machinery
    separately, out-of-band.
    """
    result = await activate_reindexed_document(db, job_id)

    if result.outcome in _ACTIVATION_OUTCOME_ERRORS:
        status_code, detail = _ACTIVATION_OUTCOME_ERRORS[result.outcome]
        raise HTTPException(status_code=status_code, detail=detail)

    assert result.job is not None
    assert result.job.activated_at is not None
    response.status_code = status.HTTP_200_OK
    return ReindexJobActivationResponse(
        job_id=result.job.id,
        document_id=result.job.document_id,
        status=result.job.status,
        activated=True,
        already_activated=result.outcome == ReindexActivationOutcome.ALREADY_ACTIVATED,
        activated_at=result.job.activated_at,
        previous_collection_name=result.job.source_collection_name,
        active_collection_name=result.job.target_collection_name,
    )
