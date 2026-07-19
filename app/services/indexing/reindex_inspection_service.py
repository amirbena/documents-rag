"""Read-only re-index lifecycle inspection for one document (Phase 2.8.6, subtask 6).

`inspect_document_reindex_state()` is the single read path behind
`GET /api/v1/documents/{document_id}/reindex` — it derives whether a document's active index is
stale relative to the platform's current desired configuration, and reports its most recent
re-index attempt (if any), without mutating anything. It reuses the same staleness primitive
(`app.services.indexing.collection_registry.is_document_stale`) and the same job-lookup helpers
(`app.services.indexing.reindex_scheduling_service.get_latest_reindex_job`/
`get_active_reindex_job`) that `schedule_reindex()`/`activate_reindexed_document()` themselves
already use — it never re-implements staleness comparison, target-snapshot construction,
scheduling eligibility, or activation-precondition validation.

## `can_schedule`/`can_activate` are best-effort hints, not guarantees

Reproducing every precondition `schedule_reindex()`/`activate_reindexed_document()` actually
enforce (active-ingestion checks, a row-locked re-validation of source staleness) here would be
exactly the duplication this subtask's spec forbids. `can_schedule`/`can_activate` are therefore a
cheap, directionally-useful summary only — the real `POST` endpoints remain the sole authority on
whether a given call actually succeeds, and either can still return a 409 for a condition this read
path did not attempt to predict (e.g. `schedule_reindex()`'s own `INGESTION_ACTIVE` check, or a
concurrent activation racing this read). Both flags *do* account for the document's blocking
deletion status, since that lookup is already required to derive `DELETION_BLOCKED` below and
reusing it costs nothing extra.

## Why the document/deletion lookups are duplicated locally, not imported

Mirrors `reindex_scheduling_service.py`'s and `reindex_activation.py`'s own precedent exactly:
`app/services/indexing/*` must never import from `app/services/documents/*` (see CLAUDE.md's
dependency rules — `deletion_service.py` itself calls into this package, so the reverse import
would create a cycle), so the tiny "latest deletion job" lookup is duplicated locally here, a third
time, rather than imported.

## Historical jobs never mask current staleness

`latest_job` is the document's single most recent `ReindexJob`, reported unconditionally for
operator visibility — even one that targeted a now-superseded configuration. But the derived
`state` only lets a job drive its classification (`REINDEX_PENDING`/`REINDEX_PROCESSING`/
`TARGET_BUILT`/`ACTIVATED`/`FAILED`) when that job's `target_collection_name` still equals the
*current* desired collection; otherwise `state` falls through to a plain `STALE`/`UP_TO_DATE`
judgment from `is_document_stale()`. An old completed-and-activated job for a configuration nobody
wants anymore must never keep reporting `ACTIVATED` forever once the platform's desired
configuration has moved on again.
"""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.models.document import Document
from app.models.document_deletion_job import DocumentDeletionJob, DocumentDeletionStatus
from app.models.reindex_job import ReindexJob, ReindexJobStatus
from app.rag.embedding_config import EmbeddingIndexConfig, get_active_embedding_config
from app.schemas.reindex import ReindexLifecycleState
from app.services.indexing.collection_registry import is_document_stale
from app.services.indexing.reindex_scheduling_service import (
    get_active_reindex_job,
    get_latest_reindex_job,
)

_BLOCKING_DELETION_STATUSES = (
    DocumentDeletionStatus.PENDING,
    DocumentDeletionStatus.PROCESSING,
    DocumentDeletionStatus.PARTIALLY_FAILED,
    DocumentDeletionStatus.COMPLETED,
)

# A fixed, generic message for API responses — the raw `ReindexJob.error_message` is never
# returned verbatim. Mirrors `app.services.documents.query_service.sanitize_ingestion_error`
# exactly.
_SAFE_REINDEX_FAILURE_MESSAGE = "Re-index build failed. See server logs for the underlying error."


@dataclass(frozen=True)
class IndexConfigSnapshot:
    """One indexing configuration snapshot — a document's active index, or the platform's desired one.

    `chunk_size`/`chunk_overlap` are always None for a document's active snapshot (`Document` does
    not persist per-document chunk settings — see `app/models/document.py`) and populated for the
    desired snapshot (read live from `Settings` at inspection time — the one legitimate use of live
    settings in this module, per this subtask's spec).
    """

    collection_name: str | None
    provider: str | None
    model: str | None
    dimension: int | None
    embedding_version: str | None
    chunking_version: str | None
    chunk_size: int | None
    chunk_overlap: int | None


@dataclass(frozen=True)
class DocumentReindexState:
    """Typed read model returned by `inspect_document_reindex_state()`.

    See module docstring for `latest_job`'s "never masks current staleness" rule and
    `can_schedule`/`can_activate`'s "best-effort hint" caveat.
    """

    document_id: str
    state: ReindexLifecycleState
    is_stale: bool
    active_index: IndexConfigSnapshot
    desired_index: IndexConfigSnapshot
    latest_job: ReindexJob | None
    can_schedule: bool
    can_activate: bool


def sanitize_reindex_error(raw_error_message: str | None) -> str | None:
    """Map a stored (potentially unsafe) ReindexJob.error_message to a fixed, safe message.

    Mirrors `app.services.documents.query_service.sanitize_ingestion_error` exactly: a fixed,
    generic message is always returned for a non-null raw message, never the raw text itself.
    """
    if raw_error_message is None:
        return None
    return _SAFE_REINDEX_FAILURE_MESSAGE


async def _latest_deletion_job(session: AsyncSession, document_id: str) -> DocumentDeletionJob | None:
    """Return `document_id`'s most recent DocumentDeletionJob — duplicated locally, see module docstring."""
    stmt = (
        select(DocumentDeletionJob)
        .where(DocumentDeletionJob.document_id == document_id)
        .order_by(DocumentDeletionJob.created_at.desc(), DocumentDeletionJob.id.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalars().first()


def _active_index_snapshot(document: Document) -> IndexConfigSnapshot:
    return IndexConfigSnapshot(
        collection_name=document.collection_name,
        provider=document.embedding_provider,
        model=document.embedding_model,
        dimension=document.embedding_dimension,
        embedding_version=document.embedding_version,
        chunking_version=document.chunking_version,
        chunk_size=None,
        chunk_overlap=None,
    )


def _desired_index_snapshot(target_config: EmbeddingIndexConfig, settings: Settings) -> IndexConfigSnapshot:
    return IndexConfigSnapshot(
        collection_name=target_config.collection_name,
        provider=target_config.provider,
        model=target_config.model,
        dimension=target_config.dimension,
        embedding_version=target_config.embedding_version,
        chunking_version=target_config.chunking_version,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def _derive_state(
    document: Document,
    latest_job: ReindexJob | None,
    is_blocked_by_deletion: bool,
    desired_collection_name: str,
    is_stale: bool,
) -> ReindexLifecycleState:
    if document.collection_name is None:
        return ReindexLifecycleState.NOT_INDEXED
    if is_blocked_by_deletion:
        return ReindexLifecycleState.DELETION_BLOCKED

    if latest_job is not None and latest_job.target_collection_name == desired_collection_name:
        if latest_job.status == ReindexJobStatus.PENDING:
            return ReindexLifecycleState.REINDEX_PENDING
        if latest_job.status == ReindexJobStatus.PROCESSING:
            return ReindexLifecycleState.REINDEX_PROCESSING
        if latest_job.status == ReindexJobStatus.FAILED:
            return ReindexLifecycleState.FAILED
        if latest_job.status == ReindexJobStatus.COMPLETED:
            if latest_job.activated_at is not None:
                return ReindexLifecycleState.ACTIVATED
            return ReindexLifecycleState.TARGET_BUILT

    return ReindexLifecycleState.STALE if is_stale else ReindexLifecycleState.UP_TO_DATE


async def inspect_document_reindex_state(
    session: AsyncSession, document_id: str, settings: Settings
) -> DocumentReindexState | None:
    """Return `document_id`'s current re-index lifecycle state, or None if the document doesn't exist.

    Never schedules, builds, or activates anything — purely derived from already-persisted state
    plus the platform's current desired `EmbeddingIndexConfig` (the one legitimate live-settings
    read in this module).
    """
    document = await session.get(Document, document_id)
    if document is None:
        return None

    desired_config = get_active_embedding_config(settings)
    is_stale = is_document_stale(document, desired_config)

    latest_job = await get_latest_reindex_job(session, document_id)
    active_job = await get_active_reindex_job(session, document_id)
    latest_deletion = await _latest_deletion_job(session, document_id)
    is_blocked_by_deletion = (
        latest_deletion is not None and latest_deletion.status in _BLOCKING_DELETION_STATUSES
    )

    state = _derive_state(
        document, latest_job, is_blocked_by_deletion, desired_config.collection_name, is_stale
    )

    can_schedule = (
        document.collection_name is not None
        and is_stale
        and active_job is None
        and not is_blocked_by_deletion
    )
    can_activate = (
        latest_job is not None
        and latest_job.status == ReindexJobStatus.COMPLETED
        and latest_job.activated_at is None
        and not is_blocked_by_deletion
    )

    return DocumentReindexState(
        document_id=document_id,
        state=state,
        is_stale=is_stale,
        active_index=_active_index_snapshot(document),
        desired_index=_desired_index_snapshot(desired_config, settings),
        latest_job=latest_job,
        can_schedule=can_schedule,
        can_activate=can_activate,
    )


__all__ = [
    "DocumentReindexState",
    "IndexConfigSnapshot",
    "inspect_document_reindex_state",
    "sanitize_reindex_error",
]
