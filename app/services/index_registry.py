"""Collection-safety, document-indexing-metadata, and legacy-vector-cleanup-tracking service.

Owns every read/write of IndexCollection and Document's indexing-metadata columns, plus the one
dimension-compatibility check every write/search path must pass through before touching Qdrant.
IngestionWorker (write side) and the re-index service both call `ensure_active_collection()`
before upserting; nothing else creates a Qdrant collection or decides whether a document is
stale.

Also owns VectorCleanupJob — the durable, retryable record of a legacy collection's vectors still
needing deletion after a document was re-indexed into a new collection (see
app/services/reindex_service.py). A cleanup failure is tracked here independently of whether the
document itself is still considered stale, so it stays discoverable and retryable even after the
document is already current under the active configuration, and multiple historical collections
pending cleanup for the same document are never conflated into a single record.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.index_collection import IndexCollection, IndexCollectionStatus
from app.models.vector_cleanup_job import VectorCleanupJob, VectorCleanupStatus
from app.rag.embedding_config import EmbeddingIndexConfig
from app.rag.providers.vector_store import VectorStore


class IncompatibleIndexConfigurationError(Exception):
    """Raised when an existing Qdrant collection's vector size doesn't match the active config.

    Never silently recreated or deleted — an operator must resolve this deliberately (bump
    EMBEDDING_VERSION/CHUNKING_VERSION to roll onto a new collection, or fix the misconfiguration).
    """


async def ensure_active_collection(
    vector_store: VectorStore, session: AsyncSession, config: EmbeddingIndexConfig
) -> None:
    """Ensure `config.collection_name` exists with the right dimension, and is tracked in Postgres.

    Fails explicitly (IncompatibleIndexConfigurationError) if a collection with this exact name
    already exists in Qdrant with a different vector size — this should be unreachable in
    practice, since the name itself encodes the dimension, but is checked anyway as a hard
    safety net against any drift between Qdrant and Postgres.
    """
    existing_dimension = await vector_store.get_collection_vector_size(config.collection_name)
    if existing_dimension is not None and existing_dimension != config.dimension:
        raise IncompatibleIndexConfigurationError(
            f"Collection {config.collection_name!r} already exists with vector size "
            f"{existing_dimension}, but the active configuration expects {config.dimension}."
        )

    await vector_store.create_collection_if_not_exists(config.collection_name, config.dimension)

    record = await session.get(IndexCollection, config.collection_name)
    if record is None:
        session.add(
            IndexCollection(
                collection_name=config.collection_name,
                embedding_provider=config.provider,
                embedding_model=config.model,
                embedding_dimension=config.dimension,
                embedding_version=config.embedding_version,
                chunking_version=config.chunking_version,
                status=IndexCollectionStatus.ACTIVE,
            )
        )
        await session.commit()


def mark_document_indexed(document: Document, config: EmbeddingIndexConfig) -> None:
    """Record the active indexing configuration on a Document after a successful index/re-index.

    Caller is responsible for committing — this only mutates the in-memory ORM object, so a
    failure after this call but before commit never persists a false "indexed" state.
    """
    document.embedding_provider = config.provider
    document.embedding_model = config.model
    document.embedding_dimension = config.dimension
    document.embedding_version = config.embedding_version
    document.chunking_version = config.chunking_version
    document.collection_name = config.collection_name
    document.indexed_at = datetime.now(UTC)


def is_document_stale(document: Document, active_config: EmbeddingIndexConfig) -> bool:
    """Return True if `document` was never indexed, or was indexed under a different config.

    A document with vectors sitting in some collection is not "current" merely because vectors
    exist somewhere — it is current only if its stored indexing configuration matches the
    platform's active configuration exactly.
    """
    return document.collection_name != active_config.collection_name


async def retire_collection(session: AsyncSession, collection_name: str) -> None:
    """Mark a tracked collection as retired — never deletes Qdrant data or the Postgres row.

    Explicit cleanup boundary for Part 5's migration strategy: retiring a collection here is a
    bookkeeping step only. Actually removing the collection from Qdrant is a separate, deliberate
    operational action this function does not perform.
    """
    record = await session.get(IndexCollection, collection_name)
    if record is not None:
        record.status = IndexCollectionStatus.RETIRED
        await session.commit()


async def get_stale_documents(session: AsyncSession, active_config: EmbeddingIndexConfig) -> list[Document]:
    """Return every Document whose stored collection_name is not the active one (or never indexed)."""
    result = await session.execute(select(Document))
    return [document for document in result.scalars().all() if is_document_stale(document, active_config)]


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


async def delete_all_tracked_document_vectors(
    document: Document, vector_store: VectorStore, session: AsyncSession
) -> None:
    """Delete a document's vectors from every collection they could still exist in.

    The FULL cleanup operation — this is what any document lifecycle/deletion path must call.
    Covers the document's currently tracked collection (`collection_name`) *and* every historical
    collection still tracked by a pending/failed VectorCleanupJob for this document (see
    `get_pending_cleanup_jobs()`), so a document deleted after one or more failed re-index
    cleanups never leaves vectors behind in an old collection merely because the failure happened
    to occur before this deletion. `session` is mandatory — there is no way to check historical
    cleanup tracking without it, and defaulting it to `None` would let a caller silently get only
    partial cleanup while believing deletion was complete. Idempotent: deleting from a collection
    with no matching vectors (already cleaned, or never populated) is a harmless no-op. Completed
    cleanup jobs are left untouched (retained for audit, per the "successful cleanup records may
    be retained" project convention) — this function only removes Qdrant data, it does not mutate
    VectorCleanupJob bookkeeping, since the document row itself is about to be deleted by the
    caller.
    """
    await delete_current_document_vectors(document, vector_store)

    for job in await get_pending_cleanup_jobs(session, document_id=document.id):
        if job.collection_name == document.collection_name:
            continue  # already covered by the tracked-collection delete above
        await vector_store.delete_by_document_id(job.collection_name, document.id)


async def create_cleanup_job(
    session: AsyncSession, document_id: str, collection_name: str, error: str | None = None
) -> VectorCleanupJob:
    """Persist a new legacy-vector cleanup for `collection_name`, and commit it.

    Called after a re-index whose new collection/Document metadata already committed
    successfully, but whose immediately-previous-collection vector deletion failed (or was
    never attempted) — the re-index itself is not a failure, so this is tracked separately and
    retryably. Pass `error` (the first attempt's exception, stringified) to record the job as
    already FAILED with one attempt logged; omit it to record a fresh PENDING job.
    """
    job = VectorCleanupJob(
        id=str(uuid.uuid4()),
        document_id=document_id,
        collection_name=collection_name,
        status=VectorCleanupStatus.FAILED if error is not None else VectorCleanupStatus.PENDING,
        attempts=1 if error is not None else 0,
        last_error=error,
    )
    session.add(job)
    await session.commit()
    return job


async def get_pending_cleanup_jobs(
    session: AsyncSession, document_id: str | None = None
) -> list[VectorCleanupJob]:
    """Return every PENDING or FAILED VectorCleanupJob, optionally scoped to one document.

    Multiple rows for the same document are returned independently — a second failed cleanup
    for a different historical collection never overwrites or hides the first.
    """
    stmt = select(VectorCleanupJob).where(
        VectorCleanupJob.status.in_([VectorCleanupStatus.PENDING, VectorCleanupStatus.FAILED])
    )
    result = await session.execute(stmt)
    jobs = list(result.scalars().all())
    if document_id is not None:
        jobs = [job for job in jobs if job.document_id == document_id]
    return jobs


async def retry_cleanup_job(
    session: AsyncSession, vector_store: VectorStore, job: VectorCleanupJob
) -> bool:
    """Retry deleting `job`'s collection's vectors for its document; update and commit status.

    Retried regardless of whether the document itself is still considered stale — cleanup
    success/failure is tracked independently of `is_document_stale()`. Idempotent: retrying a
    cleanup whose vectors were already removed (e.g. a partially-succeeded prior attempt) is a
    harmless no-op delete-by-filter call. Returns True (and marks the job COMPLETED) on success,
    False (and marks it FAILED, incrementing `attempts`/recording `last_error`) on failure.
    """
    try:
        await vector_store.delete_by_document_id(job.collection_name, job.document_id)
    except Exception as exc:
        job.status = VectorCleanupStatus.FAILED
        job.attempts += 1
        job.last_error = str(exc)
        await session.commit()
        return False

    job.status = VectorCleanupStatus.COMPLETED
    job.attempts += 1
    job.last_error = None
    job.completed_at = datetime.now(UTC)
    await session.commit()
    return True
