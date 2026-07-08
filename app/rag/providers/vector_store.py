"""Abstract interface for vector storage/search.

No concrete implementation yet — a Qdrant-backed store will be added,
and no collections are created, in a later milestone.
"""

from abc import ABC, abstractmethod
from typing import Any


class VectorStore(ABC):
    """Contract for upserting and searching embedding vectors by collection."""

    @abstractmethod
    async def upsert(
        self, collection: str, vectors: list[list[float]], payloads: list[dict[str, Any]]
    ) -> None:
        """Insert or update vectors and their associated payloads in a collection."""
        raise NotImplementedError

    @abstractmethod
    async def search(
        self, collection: str, query_vector: list[float], top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Return the top_k nearest payloads to query_vector in a collection."""
        raise NotImplementedError
