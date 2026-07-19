"""Deletes a document's vectors from Qdrant — partial (active-collection-only) and full
(every tracked collection) variants.

`delete_current_document_vectors()` is the deliberately partial operation; any document
*lifecycle* deletion must use `delete_all_tracked_document_vectors()` instead, so a partial
cleanup is never silently mistaken for a complete one (see CLAUDE.md's "Full Document Deletion
Style" and "Multilingual RAG Style" governance rules).

Full deletion's tracked-collection set (Phase 2.8.6, subtask 3) is resolved from three sources:
the document's current collection, every pending/failed `VectorCleanupJob` collection, and every
distinct `target_collection_name` from a **COMPLETED** `ReindexJob` for the document — durable
proof that a build-ahead re-index target may already hold a full vector set even though
`Document.collection_name` still points at the serving collection (not yet activated).
`PENDING`/`PROCESSING`/`FAILED` re-index jobs never contribute a target collection here.
"""

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.rag.providers.vector_store import VectorStore
from app.services.indexing.cleanup_job_service import get_pending_cleanup_jobs
from app.services.indexing.reindex_scheduling_service import get_completed_reindex_target_collections


async def delete_current_document_vectors(document: Document, vector_store: VectorStore) -> None:
    """Delete a document's vectors from its *currently tracked* collection only.

    PARTIAL cleanup — the name says so deliberately. Does not consult VectorCleanupJob, so any
    historical collection still pending/failed cleanup for this document is left untouched. This
    exists only for call sites that provably have no historical-cleanup tracking to check (e.g. a
    document that was never re-indexed) or that intentionally want the narrower operation. Any
    document *lifecycle* deletion (removing a document entirely) must use
    `delete_all_tracked_document_vectors()` instead — never this function — so partial cleanup is
    never silently mistaken for full cleanup.
    """
    if document.collection_name is not None:
        await vector_store.delete_by_document_id(document.collection_name, document.id)


@dataclass(frozen=True)
class CollectionVectorDeletionResult:
    """The outcome of attempting to delete one document's vectors from one Qdrant collection."""

    collection_name: str
    succeeded: bool
    error: str | None


@dataclass(frozen=True)
class VectorDeletionResult:
    """The aggregate outcome of a delete_all_tracked_document_vectors() call, per collection.

    A bare exception (or bare None return) cannot represent "some collections succeeded, one
    failed" — the caller needs to know exactly which collections are still dirty so a retry (or
    an alert) can be scoped correctly, and so a partial cleanup is never mistaken for a complete
    one.
    """

    document_id: str
    attempted_collections: tuple[str, ...]
    collection_results: tuple[CollectionVectorDeletionResult, ...]

    @property
    def fully_deleted(self) -> bool:
        """True only if every attempted collection's vectors were confirmed deleted."""
        return all(result.succeeded for result in self.collection_results)


async def delete_all_tracked_document_vectors(
    document: Document, vector_store: VectorStore, session: AsyncSession
) -> VectorDeletionResult:
    """Attempt to delete a document's vectors from every collection they could still exist in.

    The FULL cleanup operation — this is what any document lifecycle/deletion path must call.
    Targets the document's currently tracked collection (`collection_name`), every distinct
    historical collection still tracked by a pending/failed VectorCleanupJob for this document
    (see `get_pending_cleanup_jobs()`), and every distinct target collection from a COMPLETED
    ReindexJob for this document (see `get_completed_reindex_target_collections()`) — so a document
    deleted mid build-ahead migration (vectors already built in a target collection the document
    hasn't been activated onto yet) never leaves that target's vectors behind, and a document
    deleted after one or more failed re-index cleanups never leaves vectors behind in an old
    collection merely because the failure happened to occur before this deletion. `session` is
    mandatory — there is no way to check historical cleanup or completed-build tracking without
    it, and defaulting it to `None` would let a caller silently get only partial cleanup while
    believing deletion was complete.

    Every resolved collection is attempted independently: a failure deleting from one collection
    is recorded and does not stop, skip, or abort attempts against any other collection (active
    or historical), and never causes a silent fallback to active-only semantics. Call this
    repeatedly to retry after a partial failure — it always re-attempts every tracked collection,
    not just the ones that previously failed. The caller must inspect the returned
    `VectorDeletionResult.fully_deleted` (or `collection_results`) to tell full success apart from
    partial failure; this function itself never raises for a single collection's delete failure.

    If the same collection name is targeted twice (the active collection also appears as a
    historical cleanup-job collection, or two cleanup-job rows reference the same collection), it
    is attempted exactly once — real Qdrant deletes are idempotent so a duplicate attempt would be
    harmless, but deduplicating keeps `attempted_collections`/`collection_results` an accurate,
    non-redundant picture of what was actually targeted.

    Idempotent: deleting from a collection with no matching vectors (already cleaned, or never
    populated) is a harmless no-op, reported as a success. Completed cleanup jobs are left
    untouched (retained for audit, per the "successful cleanup records may be retained" project
    convention) — this function only removes Qdrant data, it does not mutate VectorCleanupJob
    bookkeeping, since the document row itself is about to be deleted by the caller.
    """
    collections_to_attempt: list[str] = []
    if document.collection_name is not None:
        collections_to_attempt.append(document.collection_name)

    for job in await get_pending_cleanup_jobs(session, document_id=document.id):
        if job.collection_name not in collections_to_attempt:
            collections_to_attempt.append(job.collection_name)

    for collection_name in await get_completed_reindex_target_collections(session, document.id):
        if collection_name not in collections_to_attempt:
            collections_to_attempt.append(collection_name)

    collection_results: list[CollectionVectorDeletionResult] = []
    for collection_name in collections_to_attempt:
        try:
            await vector_store.delete_by_document_id(collection_name, document.id)
        except Exception as exc:
            collection_results.append(
                CollectionVectorDeletionResult(
                    collection_name=collection_name, succeeded=False, error=str(exc)
                )
            )
        else:
            collection_results.append(
                CollectionVectorDeletionResult(collection_name=collection_name, succeeded=True, error=None)
            )

    return VectorDeletionResult(
        document_id=document.id,
        attempted_collections=tuple(collections_to_attempt),
        collection_results=tuple(collection_results),
    )


__all__ = [
    "CollectionVectorDeletionResult",
    "VectorDeletionResult",
    "delete_all_tracked_document_vectors",
    "delete_current_document_vectors",
]
