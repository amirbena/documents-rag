"""Background execution of one PENDING ReindexJob's build (Phase 2.8.6, subtask 4).

Build-only. `ReindexWorker.process_next_job()` claims one `PENDING` `ReindexJob` row
(`SELECT ... FOR UPDATE SKIP LOCKED`, mirroring `IngestionWorker`/`DocumentDeletionWorker`
exactly), transitions it to `PROCESSING` (committed before any external I/O), reconstructs the
job's pinned target `EmbeddingIndexConfig` from its `IndexCollection` foreign key plus its own
persisted `target_chunk_size`/`target_chunk_overlap`, and delegates to
`reindex_service.build_reindex_target()`. It never activates anything: `Document.collection_name`/
`embedding_*`/`chunking_version`/`indexed_at` are never touched, no `VectorCleanupJob` is created,
and no vector is ever deleted from any collection. `ReindexJob.status == COMPLETED` means only
"the pinned target build succeeded" — never "the target is active or serving." Activation is a
separate, later operation (`reindex_service.activate_reindexed_document()`), not performed here.
`ReindexWorker.process_next_job()` is invoked out-of-band by `scripts/process_pending_reindex_jobs.py`
(`make process-pending-reindex-jobs`), never inline from any route or from this module itself.

## Defense in depth against deletion races

Scheduling (`reindex_scheduling_service.schedule_reindex()`) already refuses to create a new
`ReindexJob` while a deletion is active/incomplete/completed for the document, but this worker does
not assume every historical or operational race is impossible: before building, it re-reads the
document's latest `DocumentDeletionJob` status and refuses to build if it is `PENDING`/
`PROCESSING`/`PARTIALLY_FAILED`/`COMPLETED`. This is recorded as `ReindexJob.status = FAILED` (no
new database status is introduced for this) with a stable, internal `error_message`; the worker's
own return value distinguishes this case (`ReindexWorkerOutcome.SKIPPED_DELETED`) from a genuine
build exception (`FAILED`) for the caller's benefit, without a corresponding DB-level distinction.

## `IndexCollection` does not persist `collection_prefix` separately

`EmbeddingIndexConfig.collection_name` is a derived property of six fields, one of which
(`collection_prefix`) is not itself a column on `IndexCollection` (only the fully joined
`collection_name` is persisted there). The worker reconstructs `collection_prefix` from the live
process's own `settings.qdrant_collection_name` — a platform-wide constant that should never
legitimately differ from what was used at scheduling time — and then verifies the reconstructed
config's `collection_name` matches `ReindexJob.target_collection_name` exactly before building,
failing the job cleanly (never silently building into the wrong collection) if it does not.

## Transaction boundaries

Claiming (`PENDING` -> `PROCESSING`) is committed in its own transaction before any external I/O.
No row lock is held across storage reads, extraction, chunking, embedding, or the Qdrant upsert
inside `build_reindex_target()`. Marking the job terminal (`COMPLETED`/`FAILED`) is a second,
separate commit afterward. Every `AsyncSession.commit()` — including ones `build_reindex_target()`
performs internally via `ensure_active_collection()` — expires every object tracked by the session
(`expire_on_commit=True`), so this worker never touches the original in-memory `job`/`document`
references again after such a commit; it captures plain scalar identifiers (`job_id`,
`document_id`, ...) immediately after claiming, and always re-fetches via `session.get(...)`
afterward. This is the same class of defect discovered in Subtask 2 (`MissingGreenlet` from an
unsafe lazy-load on an expired ORM object), avoided here structurally rather than defensively.

## No stale-`PROCESSING` recovery yet

If a worker process crashes between claiming a job and marking it terminal, that `ReindexJob` row
remains `PROCESSING` indefinitely — there is no recovery mechanism for `ReindexJob` in this
subtask, mirroring `IngestionJob`'s/`DocumentDeletionJob`'s own current gap (an explicit,
documented limitation, not silently patched over here).
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.models.document import Document
from app.models.document_deletion_job import DocumentDeletionJob, DocumentDeletionStatus
from app.models.index_collection import IndexCollection
from app.models.reindex_job import ReindexJob, ReindexJobStatus
from app.rag.embedding_config import EmbeddingIndexConfig
from app.services.indexing.reindex_service import build_reindex_target
from app.storage.contract import FileStorage

_MAX_ERROR_MESSAGE_LENGTH = 2048  # matches ReindexJob.error_message's column width
_BLOCKING_DELETION_STATUSES = (
    DocumentDeletionStatus.PENDING,
    DocumentDeletionStatus.PROCESSING,
    DocumentDeletionStatus.PARTIALLY_FAILED,
    DocumentDeletionStatus.COMPLETED,
)


class ReindexWorkerOutcome(StrEnum):
    """The distinct outcomes `ReindexWorker.process_next_job()` can report for one call."""

    NO_JOB = "no_job"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED_DELETED = "skipped_deleted"


@dataclass(frozen=True)
class ReindexWorkerResult:
    """Typed outcome of one `process_next_job()` call.

    `job_id`/`document_id` are `None` only for `NO_JOB` (nothing was claimed at all).
    """

    outcome: ReindexWorkerOutcome
    job_id: str | None
    document_id: str | None


def _bounded_error_message(message: str) -> str:
    """Truncate an internal error message to fit ReindexJob.error_message's column width."""
    return message[:_MAX_ERROR_MESSAGE_LENGTH]


async def _claim_next_pending_reindex_job(session: AsyncSession) -> ReindexJob | None:
    """Select-for-update the oldest pending re-index job, skipping rows locked elsewhere."""
    stmt = (
        select(ReindexJob)
        .where(ReindexJob.status == ReindexJobStatus.PENDING)
        .order_by(ReindexJob.created_at, ReindexJob.id)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def _latest_deletion_status(
    session: AsyncSession, document_id: str
) -> DocumentDeletionStatus | None:
    """Return document_id's latest DocumentDeletionJob status, or None if it never had one.

    Queried directly here (not via deletion_service.get_latest_deletion_job) — app/services/
    indexing/* must never import from app/services/documents/*, exactly the same established
    reasoning as reindex_scheduling_service.py's local `_latest_deletion_job()`.
    """
    stmt = (
        select(DocumentDeletionJob.status)
        .where(DocumentDeletionJob.document_id == document_id)
        .order_by(DocumentDeletionJob.created_at.desc(), DocumentDeletionJob.id.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalars().first()


class ReindexWorker:
    """Claims and processes one pending ReindexJob's build at a time — build-only, never activates.

    Depends only on the injected `FileStorage` abstraction, mirroring `IngestionWorker`'s
    construction convention — never a concrete storage backend directly.
    """

    def __init__(self, file_storage: FileStorage) -> None:
        self._file_storage = file_storage

    async def _fail(
        self,
        session: AsyncSession,
        job_id: str,
        document_id: str,
        message: str,
        outcome: ReindexWorkerOutcome,
    ) -> ReindexWorkerResult:
        """Mark `job_id` FAILED with a bounded message and commit; return the given outcome."""
        job = await session.get(ReindexJob, job_id)
        assert job is not None
        job.status = ReindexJobStatus.FAILED
        job.error_message = _bounded_error_message(message)
        job.completed_at = datetime.now(UTC)
        await session.commit()
        return ReindexWorkerResult(outcome=outcome, job_id=job_id, document_id=document_id)

    async def process_next_job(self, session: AsyncSession, settings: Settings) -> ReindexWorkerResult:
        """Claim one pending re-index job and build it; return the resolved outcome.

        Returns `NO_JOB` if there is no pending job to claim. Never activates the built target —
        see the module docstring. `settings` is the base process settings `build_reindex_target()`
        derives an isolated target-scoped copy from (Subtask 1); it is never used to resolve the
        build target directly.
        """
        job = await _claim_next_pending_reindex_job(session)
        if job is None:
            return ReindexWorkerResult(outcome=ReindexWorkerOutcome.NO_JOB, job_id=None, document_id=None)

        job.status = ReindexJobStatus.PROCESSING
        await session.commit()

        # Captured once, immediately after the claim commit — every subsequent commit in this
        # method (including ones build_reindex_target() performs internally) expires every object
        # in the session, so `job` must never be read again after this point. See module docstring.
        job_id = job.id
        document_id = job.document_id
        target_collection_name = job.target_collection_name
        target_chunk_size = job.target_chunk_size
        target_chunk_overlap = job.target_chunk_overlap

        document = await session.get(Document, document_id)
        if document is None:
            return await self._fail(
                session,
                job_id,
                document_id,
                f"Document {document_id} not found.",
                ReindexWorkerOutcome.FAILED,
            )

        deletion_status = await _latest_deletion_status(session, document_id)
        if deletion_status in _BLOCKING_DELETION_STATUSES:
            return await self._fail(
                session,
                job_id,
                document_id,
                f"Document {document_id} has a blocking deletion state ({deletion_status}); "
                "refusing to build.",
                ReindexWorkerOutcome.SKIPPED_DELETED,
            )

        index_collection = await session.get(IndexCollection, target_collection_name)
        if index_collection is None:
            return await self._fail(
                session,
                job_id,
                document_id,
                f"Target collection {target_collection_name!r} has no IndexCollection row.",
                ReindexWorkerOutcome.FAILED,
            )

        target_config = EmbeddingIndexConfig(
            collection_prefix=settings.qdrant_collection_name,
            provider=index_collection.embedding_provider,
            model=index_collection.embedding_model,
            dimension=index_collection.embedding_dimension,
            embedding_version=index_collection.embedding_version,
            chunking_version=index_collection.chunking_version,
        )
        if target_config.collection_name != target_collection_name:
            return await self._fail(
                session,
                job_id,
                document_id,
                f"Reconstructed target collection {target_config.collection_name!r} does not "
                f"match the pinned target {target_collection_name!r}.",
                ReindexWorkerOutcome.FAILED,
            )

        try:
            await build_reindex_target(
                document,
                session,
                settings,
                self._file_storage,
                target_config,
                target_chunk_size=target_chunk_size,
                target_chunk_overlap=target_chunk_overlap,
            )
        except Exception as exc:
            await session.rollback()
            return await self._fail(
                session, job_id, document_id, str(exc), ReindexWorkerOutcome.FAILED
            )

        completed_job = await session.get(ReindexJob, job_id)
        assert completed_job is not None
        if completed_job.status != ReindexJobStatus.PROCESSING:
            # Should be unreachable — nothing else can claim a row this worker's own commit already
            # moved to PROCESSING — but never silently overwrite an unexpected external change.
            return await self._fail(
                session,
                job_id,
                document_id,
                f"ReindexJob {job_id} was no longer PROCESSING "
                f"(found {completed_job.status.value!r}) after a successful build.",
                ReindexWorkerOutcome.FAILED,
            )

        completed_job.status = ReindexJobStatus.COMPLETED
        completed_job.completed_at = datetime.now(UTC)
        completed_job.error_message = None
        await session.commit()

        return ReindexWorkerResult(
            outcome=ReindexWorkerOutcome.COMPLETED, job_id=job_id, document_id=document_id
        )


__all__ = ["ReindexWorker", "ReindexWorkerOutcome", "ReindexWorkerResult"]
