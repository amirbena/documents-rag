"""Backend re-index capability: re-derives a Document's vectors from its already-persisted
stored file, using the platform's *current* active EmbeddingIndexConfig — no new upload required.

Document -> re-load stored file -> re-extract -> re-chunk (active chunking config) -> re-embed
(active embedding config) -> upsert into the active versioned collection -> persist new indexing
metadata -> retire the document's vectors in its previous collection, if any and if different.

Idempotent: point IDs are derived identically to the initial-ingest path (see
app.services.ingestion_worker.to_vector_point), so re-running against the same active collection
overwrites the same points rather than duplicating them. A failure at any step propagates without
updating the document's indexing metadata — it stays exactly as stale as it was before the
attempt, never marked current on a failed run. Existing vectors in the document's previous
collection are only removed *after* the new collection's write has already succeeded and been
committed.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.models.document import Document
from app.rag.embedding_config import get_active_embedding_config
from app.rag.embedding_validation import validate_embeddings
from app.rag.providers.provider_factory import get_embedding_provider, get_vector_store
from app.services.document_chunker import DocumentChunker
from app.services.document_text_extractor import DocumentTextExtractor
from app.services.index_registry import ensure_active_collection, is_document_stale, mark_document_indexed
from app.services.ingestion_worker import to_vector_point


async def reindex_document(
    document: Document, session: AsyncSession, settings: Settings | None = None
) -> bool:
    """Re-extract/re-chunk/re-embed/re-upsert one Document under the active configuration.

    Returns True whether the document was already current (no-op) or was freshly re-indexed.
    Raises on any extraction/embedding/vector-store failure; the document's stored indexing
    metadata is left completely untouched on failure, so `is_document_stale()` still reports it
    as stale afterward — a failed re-index attempt is indistinguishable from one that never ran.
    """
    settings = settings or get_settings()
    active_config = get_active_embedding_config(settings)

    if not is_document_stale(document, active_config):
        return True

    previous_collection_name = document.collection_name

    extracted = await DocumentTextExtractor().extract(document)
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
    await session.commit()

    if previous_collection_name and previous_collection_name != active_config.collection_name:
        await vector_store.delete_by_document_id(previous_collection_name, document.id)

    return True
