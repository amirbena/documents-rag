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
    """Contract for collection creation, vector upsert, and similarity search."""

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
