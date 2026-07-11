"""Collection-safety and document-indexing-metadata service.

Owns every read/write of IndexCollection and Document's indexing-metadata columns, plus the one
dimension-compatibility check every write/search path must pass through before touching Qdrant.
IngestionWorker (write side) and the re-index service both call `ensure_active_collection()`
before upserting; nothing else creates a Qdrant collection or decides whether a document is
stale.
"""

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.index_collection import IndexCollection, IndexCollectionStatus
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
