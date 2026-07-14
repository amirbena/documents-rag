"""Background execution of scheduled document deletion: claim, cross-system cleanup, completion.

`DocumentDeletionWorker.process_next_job()` mirrors `IngestionWorker`
(`app/services/ingestion/worker.py`) — claims one `PENDING` `DocumentDeletionJob` row with
`SELECT ... FOR UPDATE SKIP LOCKED`, transitions it to `PROCESSING` (committed before any external
I/O), then performs vector cleanup strictly before storage cleanup (see "Cleanup order" below).
Invoked by `scripts/process_pending_document_deletions.py`, not by any HTTP route or background
scheduler — this codebase has no deployed worker process for `IngestionJob` either
(`IngestionWorker.process_next_job()` is only ever invoked by test fixtures and `scripts/`), so
this mirrors the existing architecture rather than introducing a new one. Depends on
`deletion_service` for the shared `DeletionErrorCode`/status constants (dependency direction:
worker -> service, never the reverse — see `deletion_service`'s module docstring).

## Cleanup order: vectors before storage, always

`DocumentDeletionWorker` deletes all tracked vectors
(`vector_deletion_service.delete_all_tracked_document_vectors()`) *before* ever calling
`FileStorage.delete()`. If vector cleanup does not fully succeed, storage cleanup is never
attempted in that same job, and the job is marked
`PARTIALLY_FAILED` with `storage_cleanup_completed` left `False` — enforced structurally by the
code path (there is no branch that reaches storage deletion without first observing
`vector_result.fully_deleted is True`), not merely as a documented intention. This ordering
matters because searchable derived content (vectors) must stop being searchable before the
document is ever reported as deleted; the original object can safely be cleaned up afterward
(and retried independently) since it plays no role in retrieval.

## Content-hash release (Phase 2.8.5)

Only the `COMPLETED` transition — the same commit that sets `job.status = COMPLETED` — also sets
`document.content_hash = None`, releasing this document's content identity so a later upload of
the same bytes may claim it again (see `app.services.documents.dedup_service`). Every other
outcome (`PENDING`, `PROCESSING`, any `PARTIALLY_FAILED` branch, a crash before this point) leaves
`content_hash` untouched — a document whose deletion did not genuinely finish must never look
available for reuse.
"""

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_deletion_job import DocumentDeletionJob, DocumentDeletionStatus
from app.rag.providers.vector_store import VectorStore
from app.services.documents.deletion_service import DeletionErrorCode
from app.services.indexing.vector_deletion_service import delete_all_tracked_document_vectors
from app.storage.contract import FileStorage
from app.storage.errors import StorageError
from app.storage.keys import resolve_document_storage_key


class DocumentDeletionWorker:
    """Claims and processes one pending DocumentDeletionJob at a time.

    Mirrors `IngestionWorker`'s claim/process/resolve shape. Depends only on the `VectorStore`/
    `FileStorage` abstractions injected at construction — never a concrete Qdrant/MinIO/local type.
    """

    def __init__(self, vector_store: VectorStore, file_storage: FileStorage) -> None:
        self._vector_store = vector_store
        self._file_storage = file_storage

    async def _claim_next_pending_job(self, session: AsyncSession) -> DocumentDeletionJob | None:
        """Select-for-update the oldest pending deletion job, skipping rows locked elsewhere."""
        stmt = (
            select(DocumentDeletionJob)
            .where(DocumentDeletionJob.status == DocumentDeletionStatus.PENDING)
            .order_by(DocumentDeletionJob.created_at)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def process_next_job(self, session: AsyncSession) -> DocumentDeletionJob | None:
        """Claim one pending deletion job and execute its cleanup order; return the resolved job.

        Returns None if there is no pending job to claim. Cleanup order: vectors are deleted (via
        `delete_all_tracked_document_vectors`, the full-tracked-collection operation — never
        `delete_current_document_vectors`) strictly before the original object is deleted via
        `FileStorage.delete()`. If vector cleanup does not fully succeed, storage cleanup is never
        reached in this call: the method returns with the job PARTIALLY_FAILED immediately after
        recording the vector-cleanup failure, so "vectors clean AND object clean" is the only path
        that reaches COMPLETED.
        """
        job = await self._claim_next_pending_job(session)
        if job is None:
            return None

        job.status = DocumentDeletionStatus.PROCESSING
        await session.commit()

        document = await session.get(Document, job.document_id)
        if document is None:
            # Document row is never physically deleted elsewhere in this codebase; defensive only.
            job.status = DocumentDeletionStatus.PARTIALLY_FAILED
            job.error_code = DeletionErrorCode.DOCUMENT_VECTOR_CLEANUP_FAILED
            job.error_message = f"Document {job.document_id} not found during deletion execution."
            await session.commit()
            return job

        vector_result = await delete_all_tracked_document_vectors(document, self._vector_store, session)
        if not vector_result.fully_deleted:
            failures = [r for r in vector_result.collection_results if not r.succeeded]
            job.status = DocumentDeletionStatus.PARTIALLY_FAILED
            job.error_code = DeletionErrorCode.DOCUMENT_VECTOR_CLEANUP_FAILED
            job.error_message = (
                "Vector cleanup failed for collection(s): "
                + ", ".join(f"{r.collection_name}: {r.error}" for r in failures)
            )
            await session.commit()
            return job

        job.vector_cleanup_completed = True
        await session.commit()

        key = resolve_document_storage_key(document)
        try:
            await self._file_storage.delete(key)
        except StorageError as exc:
            job.status = DocumentDeletionStatus.PARTIALLY_FAILED
            job.error_code = DeletionErrorCode.DOCUMENT_STORAGE_CLEANUP_FAILED
            job.error_message = str(exc)
            await session.commit()
            return job

        job.storage_cleanup_completed = True
        job.status = DocumentDeletionStatus.COMPLETED
        job.completed_at = datetime.now(UTC)
        job.error_code = None
        job.error_message = None
        # Release this document's content identity in the same commit as COMPLETED (Phase
        # 2.8.5) — never on PENDING/PROCESSING/PARTIALLY_FAILED — so a later upload of the same
        # bytes may claim the hash only once deletion has genuinely, fully finished.
        document.content_hash = None
        await session.commit()
        return job


__all__ = ["DocumentDeletionWorker"]
