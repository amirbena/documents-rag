"""Abstract interface for vector storage/search, plus its shared data types.

See app/rag/providers/qdrant_vector_store.py for the Qdrant-backed implementation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class VectorPoint:
    """One embedding vector plus its id and payload metadata, ready to upsert."""

    id: str
    vector: list[float]
    document_id: str
    chunk_id: str
    text: str
    source: str
    page_number: int | None = None
    sheet_name: str | None = None


@dataclass
class VectorSearchResult:
    """One nearest-neighbor match: similarity score plus the point's payload metadata."""

    id: str
    score: float
    document_id: str
    chunk_id: str
    text: str
    source: str
    page_number: int | None = None
    sheet_name: str | None = None


class VectorStore(ABC):
    """Contract for collection creation, vector upsert, similarity search, and cleanup."""

    @abstractmethod
    async def create_collection_if_not_exists(self, collection_name: str, vector_size: int) -> None:
        """Create the collection with the given vector size if it doesn't already exist."""
        raise NotImplementedError

    @abstractmethod
    async def upsert_vectors(self, collection_name: str, points: list[VectorPoint]) -> None:
        """Insert or update the given vector points in a collection."""
        raise NotImplementedError

    @abstractmethod
    async def search_similar(
        self, collection_name: str, query_vector: list[float], limit: int = 5
    ) -> list[VectorSearchResult]:
        """Return the top `limit` nearest points to query_vector in a collection."""
        raise NotImplementedError

    @abstractmethod
    async def get_collection_vector_size(self, collection_name: str) -> int | None:
        """Return the existing collection's configured vector size, or None if it doesn't exist."""
        raise NotImplementedError

    @abstractmethod
    async def delete_by_document_id(self, collection_name: str, document_id: str) -> None:
        """Delete every point belonging to document_id from a collection, if the collection exists."""
        raise NotImplementedError

    @abstractmethod
    async def count_document_vectors(self, collection_name: str, document_id: str) -> int:
        """Return how many points belong to document_id in a collection (0 if the collection is missing).

        A read-only existence/count check — added for the document lifecycle audit (Phase 2.8.7),
        which needs to know "does at least one vector exist for this document" without retrieving
        any payload or performing a similarity search.
        """
        raise NotImplementedError
