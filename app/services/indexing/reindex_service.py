"""Backend re-index capability: re-derives a Document's vectors from its already-persisted
stored file, using the platform's *current* active EmbeddingIndexConfig — no new upload required.

Document -> re-load stored file -> re-extract -> re-chunk (active chunking config) -> re-embed
(active embedding config) -> validate real vector dimensions -> upsert into the active versioned
collection -> persist new indexing metadata -> attempt to retire the document's vectors in its
previous collection, if any and if different.

Idempotent: point IDs are derived identically to the initial-ingest path (see
app.services.ingestion.worker.to_vector_point), so re-running against the same active collection
overwrites the same points rather than duplicating them.

Transaction/failure semantics — read carefully, this is intentionally NOT one atomic transaction
across Qdrant and PostgreSQL, and the two systems can observe different outcomes for the same
attempt:

- Extraction/chunking/embedding/validation/collection/upsert failure (before the Postgres commit
  below): the exception propagates, no Document indexing metadata is ever mutated-and-committed,
  and no previously-valid vectors are touched. `is_document_stale()` still reports the document as
  stale afterward.
- The Qdrant upsert can succeed and the following Postgres commit can still fail (e.g. a
  connection drop). In that case: the new points already exist in Qdrant (deterministic point IDs
  make this retry-safe — the next successful re-index attempt overwrites the same points rather
  than duplicating them), but the Document row is rolled back and re-`expire()`d so no in-memory
  attribute change survives the failed commit — the document remains exactly as stale as before.
  This attempt is *not* indistinguishable from one that never ran (Qdrant now holds orphaned
  points until a retry succeeds or a future cleanup pass removes them) — do not describe it that
  way in documentation.
- After the new collection/Document-metadata commit succeeds, deleting the immediately-previous
  collection's vectors is attempted separately. Failure there does not undo or reclassify the
  re-index itself (the document IS current) — it is tracked as a pending VectorCleanupJob (see
  app/services/indexing/cleanup_job_service.py) and reported via
  ReindexOutcome.REINDEXED_WITH_CLEANUP_PENDING, retryable later via `retry_cleanup_job()` even
  once the document is no longer stale.
"""

from dataclasses import dataclass
from enum import StrEnum

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.models.document import Document
from app.rag.embedding_config import get_active_embedding_config
from app.rag.embedding_validation import validate_embeddings
from app.rag.providers.provider_factory import get_embedding_provider, get_vector_store
from app.services.documents.chunker import DocumentChunker
from app.services.documents.text_extractor import DocumentTextExtractor
from app.services.indexing.cleanup_job_service import create_cleanup_job
from app.services.indexing.collection_registry import (
    ensure_active_collection,
    is_document_stale,
    mark_document_indexed,
)
from app.services.ingestion.worker import to_vector_point
from app.storage.contract import FileStorage
from app.storage.factory import create_file_storage


class ReindexOutcome(StrEnum):
    """The distinct outcomes reindex_document() can report for a successful call.

    A raised exception (rather than a returned ReindexResult) means the re-index itself did not
    complete — see the module docstring for exactly which failure modes raise vs. which are
    represented here.
    """

    ALREADY_CURRENT = "already_current"
    REINDEXED = "reindexed"
    REINDEXED_WITH_CLEANUP_PENDING = "reindexed_with_cleanup_pending"
    REINDEXED_EMPTY = "reindexed_empty"


@dataclass(frozen=True)
class ReindexResult:
    """The outcome of one reindex_document() call, plus the (possibly updated) Document."""

    outcome: ReindexOutcome
    document: Document


async def reindex_document(
    document: Document,
    session: AsyncSession,
    settings: Settings | None = None,
    file_storage: FileStorage | None = None,
) -> ReindexResult:
    """Re-extract/re-chunk/re-embed/re-upsert one Document under the active configuration.

    Returns ALREADY_CURRENT (no-op) if the document already matches the active configuration.
    Otherwise re-indexes and returns REINDEXED_EMPTY if extraction/chunking produced zero chunks
    (the document is marked current with no searchable content — see "Zero-chunk behavior" in
    ARCHITECTURE.md), REINDEXED_WITH_CLEANUP_PENDING if the new collection/metadata committed but
    deleting the previous collection's vectors failed, or REINDEXED otherwise. Raises on any
    extraction/embedding/validation/vector-store-write/Postgres-commit failure — see the module
    docstring for the exact per-failure-mode contract.
    """
    settings = settings or get_settings()
    active_config = get_active_embedding_config(settings)

    if not is_document_stale(document, active_config):
        return ReindexResult(outcome=ReindexOutcome.ALREADY_CURRENT, document=document)

    previous_collection_name = document.collection_name

    file_storage = file_storage or create_file_storage(settings)
    extracted = await DocumentTextExtractor(storage=file_storage).extract(document)
    chunker = DocumentChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    chunks = chunker.chunk(extracted)

    vector_store = get_vector_store(settings)

    if chunks:
        embedding_provider = get_embedding_provider(settings)
        vectors = await embedding_provider.embed([chunk.text for chunk in chunks])
        validate_embeddings(vectors, expected_count=len(chunks), expected_dimension=active_config.dimension)
        points = [
            to_vector_point(chunk, vector, document.original_filename)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        await ensure_active_collection(vector_store, session, active_config)
        await vector_store.upsert_vectors(active_config.collection_name, points)

    mark_document_indexed(document, active_config)
    try:
        await session.commit()
    except Exception:
        await session.rollback()
        session.expire(document)
        raise

    outcome = ReindexOutcome.REINDEXED if chunks else ReindexOutcome.REINDEXED_EMPTY

    if previous_collection_name and previous_collection_name != active_config.collection_name:
        try:
            await vector_store.delete_by_document_id(previous_collection_name, document.id)
        except Exception as exc:
            await create_cleanup_job(session, document.id, previous_collection_name, error=str(exc))
            outcome = ReindexOutcome.REINDEXED_WITH_CLEANUP_PENDING

    return ReindexResult(outcome=outcome, document=document)
