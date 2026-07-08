"""Qdrant-backed implementation of VectorStore, using Qdrant's HTTP API directly (no SDK).

Calls only Qdrant's REST endpoints under QDRANT_URL — no ingestion, no document upload, no
chat/SSE endpoint, no full RAG flow. This is the storage/search half of the RAG pipeline in
isolation.
"""

from typing import Any

import httpx

from app.core.config import Settings, get_settings
from app.rag.providers.vector_store import VectorPoint, VectorSearchResult, VectorStore


class QdrantVectorStoreError(Exception):
    """Raised when Qdrant is unreachable, returns an error, or responds unexpectedly."""


def _point_payload(point: VectorPoint) -> dict[str, Any]:
    """Build the Qdrant payload dict (document_id, chunk_id, text, source, page_number)."""
    payload: dict[str, Any] = {
        "document_id": point.document_id,
        "chunk_id": point.chunk_id,
        "text": point.text,
        "source": point.source,
    }
    if point.page_number is not None:
        payload["page_number"] = point.page_number
    return payload


def _parse_search_result(item: dict[str, Any]) -> VectorSearchResult:
    """Build a VectorSearchResult from one Qdrant search-response item."""
    payload = item.get("payload") or {}
    try:
        return VectorSearchResult(
            id=str(item["id"]),
            score=item["score"],
            document_id=payload["document_id"],
            chunk_id=payload["chunk_id"],
            text=payload["text"],
            source=payload["source"],
            page_number=payload.get("page_number"),
        )
    except KeyError as exc:
        raise QdrantVectorStoreError("Malformed search response from Qdrant") from exc


class QdrantVectorStore(VectorStore):
    """VectorStore that talks to Qdrant's HTTP API under QDRANT_URL."""

    def __init__(
        self,
        settings: Settings | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._transport = transport

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._settings.qdrant_url, timeout=30.0, transport=self._transport
        )

    async def create_collection_if_not_exists(self, collection_name: str, vector_size: int) -> None:
        """Create the collection with the given vector size if it doesn't already exist."""
        async with self._client() as client:
            try:
                existing = await client.get(f"/collections/{collection_name}")
            except httpx.HTTPError as exc:
                raise QdrantVectorStoreError(f"Qdrant unreachable at /collections: {exc}") from exc

            if existing.status_code == 200:
                return

            try:
                response = await client.put(
                    f"/collections/{collection_name}",
                    json={"vectors": {"size": vector_size, "distance": "Cosine"}},
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise QdrantVectorStoreError(
                    f"Qdrant returned {exc.response.status_code} creating collection {collection_name!r}"
                ) from exc
            except httpx.HTTPError as exc:
                raise QdrantVectorStoreError(f"Qdrant unreachable creating collection: {exc}") from exc

    async def upsert_vectors(self, collection_name: str, points: list[VectorPoint]) -> None:
        """Insert or update the given vector points in a collection."""
        body = {
            "points": [
                {"id": point.id, "vector": point.vector, "payload": _point_payload(point)}
                for point in points
            ]
        }
        async with self._client() as client:
            try:
                response = await client.put(
                    f"/collections/{collection_name}/points",
                    params={"wait": "true"},
                    json=body,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise QdrantVectorStoreError(
                    f"Qdrant returned {exc.response.status_code} upserting into {collection_name!r}"
                ) from exc
            except httpx.HTTPError as exc:
                raise QdrantVectorStoreError(f"Qdrant unreachable upserting vectors: {exc}") from exc

    async def search_similar(
        self, collection_name: str, query_vector: list[float], limit: int = 5
    ) -> list[VectorSearchResult]:
        """Return the top `limit` nearest points to query_vector in a collection."""
        async with self._client() as client:
            try:
                response = await client.post(
                    f"/collections/{collection_name}/points/search",
                    json={"vector": query_vector, "limit": limit, "with_payload": True},
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise QdrantVectorStoreError(
                    f"Qdrant returned {exc.response.status_code} searching {collection_name!r}"
                ) from exc
            except httpx.HTTPError as exc:
                raise QdrantVectorStoreError(f"Qdrant unreachable searching vectors: {exc}") from exc

        try:
            results = response.json()["result"]
        except (ValueError, KeyError, TypeError) as exc:
            raise QdrantVectorStoreError("Malformed search response from Qdrant") from exc

        return [_parse_search_result(item) for item in results]
