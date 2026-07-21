"""Qdrant-backed implementation of VectorStore, using Qdrant's HTTP API directly (no SDK).

Calls only Qdrant's REST endpoints under QDRANT_URL — no ingestion, no document upload, no
chat/SSE endpoint, no full RAG flow. This is the storage/search half of the RAG pipeline in
isolation.
"""

from typing import Any

import httpx

from app.core.config import Settings, get_settings
from app.core.correlation import correlation_headers
from app.core.retry import retry_async
from app.rag.providers.http_retry_policy import TRANSIENT_STATUS_CODES, is_transient_httpx_error
from app.rag.providers.vector_store import VectorPoint, VectorSearchResult, VectorStore

# Category (Phase 2.10, see app/core/errors.py): ProviderError.


class QdrantVectorStoreError(Exception):
    """Raised when Qdrant is unreachable, returns an error, or responds unexpectedly."""


def _point_payload(point: VectorPoint) -> dict[str, Any]:
    """Build the Qdrant payload dict (document_id, chunk_id, text, source, page_number, sheet_name)."""
    payload: dict[str, Any] = {
        "document_id": point.document_id,
        "chunk_id": point.chunk_id,
        "text": point.text,
        "source": point.source,
    }
    if point.page_number is not None:
        payload["page_number"] = point.page_number
    if point.sheet_name is not None:
        payload["sheet_name"] = point.sheet_name
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
            sheet_name=payload.get("sheet_name"),
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
            base_url=self._settings.qdrant_url,
            timeout=self._settings.qdrant_timeout_seconds,
            transport=self._transport,
            headers=correlation_headers(),
        )

    async def _request(
        self, client: httpx.AsyncClient, method: str, url: str, **kwargs: Any
    ) -> httpx.Response:
        """One retried HTTP call through an already-open `client` (Phase 2.10).

        Classification happens on the raw httpx exception (see http_retry_policy.py) — only
        connection/timeout failures and 429/502/503/504 are retried; every other status
        (including 404, which several call sites — existence checks — need to inspect themselves
        rather than treat as an error) is returned to the caller as a normal response, never
        raised here. A transient status code (429/502/503/504) is turned into an
        `httpx.HTTPStatusError` internally so the retry loop can see and act on it — a response
        with any other status is returned as-is, unraised, exactly as before.
        """

        async def _call() -> httpx.Response:
            response = await client.request(method, url, **kwargs)
            if response.status_code in TRANSIENT_STATUS_CODES:
                response.raise_for_status()
            return response

        return await retry_async(
            _call,
            max_attempts=self._settings.provider_retry_max_attempts,
            base_delay=self._settings.provider_retry_base_delay_seconds,
            max_delay=self._settings.provider_retry_max_delay_seconds,
            is_transient=is_transient_httpx_error,
        )

    async def _collection_vector_size(self, client: httpx.AsyncClient, collection_name: str) -> int | None:
        """Return `collection_name`'s configured vector size, or None if it doesn't exist.

        Internal helper reusing an already-open `client` — see `get_collection_vector_size()` for
        the public, standalone-client version of this same lookup.
        """
        try:
            response = await self._request(client, "GET", f"/collections/{collection_name}")
        except httpx.HTTPError as exc:
            raise QdrantVectorStoreError(f"Qdrant unreachable at /collections: {exc}") from exc

        if response.status_code == 404:
            return None

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise QdrantVectorStoreError(
                f"Qdrant returned {exc.response.status_code} inspecting collection {collection_name!r}"
            ) from exc

        try:
            vectors_config = response.json()["result"]["config"]["params"]["vectors"]
            # Qdrant returns either {"size": N, "distance": ...} for a single unnamed vector, or
            # {"<name>": {"size": N, ...}, ...} for named vectors — this project only ever creates
            # the single unnamed form, so only that shape is supported.
            return int(vectors_config["size"])
        except (ValueError, KeyError, TypeError) as exc:
            raise QdrantVectorStoreError(
                f"Malformed collection-info response from Qdrant for {collection_name!r}"
            ) from exc

    async def create_collection_if_not_exists(self, collection_name: str, vector_size: int) -> None:
        """Create the collection with the given vector size if it doesn't already exist."""
        async with self._client() as client:
            try:
                existing = await self._request(client, "GET", f"/collections/{collection_name}")
            except httpx.HTTPError as exc:
                raise QdrantVectorStoreError(f"Qdrant unreachable at /collections: {exc}") from exc

            if existing.status_code == 200:
                return

            try:
                response = await self._request(
                    client,
                    "PUT",
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
                response = await self._request(
                    client,
                    "PUT",
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
                response = await self._request(
                    client,
                    "POST",
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

    async def get_collection_vector_size(self, collection_name: str) -> int | None:
        """Return the existing collection's configured vector size, or None if it doesn't exist."""
        async with self._client() as client:
            return await self._collection_vector_size(client, collection_name)

    async def count_document_vectors(self, collection_name: str, document_id: str) -> int:
        """Return how many points belong to document_id in a collection (0 if it doesn't exist).

        Phase 2.10: the existence pre-check and the count call now share one client lifecycle
        (previously two separate `async with self._client()` blocks per logical operation).
        """
        async with self._client() as client:
            exists = await self._collection_vector_size(client, collection_name)
            if exists is None:
                return 0

            try:
                response = await self._request(
                    client,
                    "POST",
                    f"/collections/{collection_name}/points/count",
                    json={
                        "filter": {"must": [{"key": "document_id", "match": {"value": document_id}}]},
                        "exact": True,
                    },
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise QdrantVectorStoreError(
                    f"Qdrant returned {exc.response.status_code} counting document {document_id!r} "
                    f"in {collection_name!r}"
                ) from exc
            except httpx.HTTPError as exc:
                raise QdrantVectorStoreError(f"Qdrant unreachable counting vectors: {exc}") from exc

        try:
            return int(response.json()["result"]["count"])
        except (ValueError, KeyError, TypeError) as exc:
            raise QdrantVectorStoreError(
                f"Malformed count response from Qdrant for {collection_name!r}"
            ) from exc

    async def count_collection_vectors(self, collection_name: str) -> int | None:
        """Return the total number of points in a collection, or None if it doesn't exist.

        Phase 2.10: the existence pre-check and the count call now share one client lifecycle.
        """
        async with self._client() as client:
            exists = await self._collection_vector_size(client, collection_name)
            if exists is None:
                return None

            try:
                response = await self._request(
                    client,
                    "POST",
                    f"/collections/{collection_name}/points/count",
                    json={"exact": True},
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise QdrantVectorStoreError(
                    f"Qdrant returned {exc.response.status_code} counting points in {collection_name!r}"
                ) from exc
            except httpx.HTTPError as exc:
                raise QdrantVectorStoreError(f"Qdrant unreachable counting vectors: {exc}") from exc

        try:
            return int(response.json()["result"]["count"])
        except (ValueError, KeyError, TypeError) as exc:
            raise QdrantVectorStoreError(
                f"Malformed count response from Qdrant for {collection_name!r}"
            ) from exc

    async def delete_by_document_id(self, collection_name: str, document_id: str) -> None:
        """Delete every point belonging to document_id from a collection, if it exists.

        Phase 2.10: the existence pre-check and the delete call now share one client lifecycle.
        """
        async with self._client() as client:
            exists = await self._collection_vector_size(client, collection_name)
            if exists is None:
                return

            try:
                response = await self._request(
                    client,
                    "POST",
                    f"/collections/{collection_name}/points/delete",
                    params={"wait": "true"},
                    json={"filter": {"must": [{"key": "document_id", "match": {"value": document_id}}]}},
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise QdrantVectorStoreError(
                    f"Qdrant returned {exc.response.status_code} deleting document {document_id!r} "
                    f"from {collection_name!r}"
                ) from exc
            except httpx.HTTPError as exc:
                raise QdrantVectorStoreError(f"Qdrant unreachable deleting vectors: {exc}") from exc
