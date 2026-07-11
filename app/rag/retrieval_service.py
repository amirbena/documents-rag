"""Internal retrieval service: embeds a query and searches Qdrant for relevant chunks.

Document -> query embedding -> Qdrant similarity search -> ranked VectorSearchResult list.
No LLM call, no public retrieval/chat/SSE endpoint, no RAG prompt assembly — this is the
retrieval half of the RAG pipeline in isolation, mirroring how the ingestion worker composes
the embedding/vector-store providers on the write side. Searches the *active* versioned
collection (app.rag.embedding_config.get_active_embedding_config()) — the exact same collection
IngestionWorker writes into — never the raw QDRANT_COLLECTION_NAME prefix directly, so query
embeddings and document embeddings can never silently drift onto different collections.
"""

from app.core.config import Settings, get_settings
from app.rag.embedding_config import get_active_embedding_config
from app.rag.providers.provider_factory import get_embedding_provider, get_vector_store
from app.rag.providers.vector_store import VectorSearchResult


class EmptyQueryError(ValueError):
    """Raised when retrieve() is called with an empty or whitespace-only query."""


class RetrievalService:
    """Embeds a user query and returns ranked relevant chunks from the active versioned collection."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def retrieve(self, query: str, limit: int | None = None) -> list[VectorSearchResult]:
        """Embed `query`, search the active collection, and return matching chunks ranked by score.

        Returns an empty list if nothing meets RETRIEVAL_SCORE_THRESHOLD (when set) — never
        fabricates context. Raises EmptyQueryError for an empty/whitespace-only query.
        """
        if not query or not query.strip():
            raise EmptyQueryError("query must not be empty")

        top_k = limit if limit is not None else self._settings.retrieval_top_k

        embedding_provider = get_embedding_provider(self._settings)
        query_vector = (await embedding_provider.embed([query]))[0]

        vector_store = get_vector_store(self._settings)
        collection_name = get_active_embedding_config(self._settings).collection_name
        results = await vector_store.search_similar(collection_name, query_vector, limit=top_k)

        threshold = self._settings.retrieval_score_threshold
        if threshold is not None:
            results = [result for result in results if result.score >= threshold]

        return results
