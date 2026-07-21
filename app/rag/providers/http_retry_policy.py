"""Shared transient/permanent classification for the raw-httpx providers (Ollama, Qdrant).

Used with `app.core.retry.retry_async` — classification happens on the *raw* `httpx` exception,
before it is translated into each provider's own error type (`OllamaEmbeddingError`,
`QdrantVectorStoreError`, ...), so a caller catching that provider's own error type sees no
difference between a retried-then-succeeded call and one that never needed a retry.
"""

import httpx

TRANSIENT_STATUS_CODES = frozenset({429, 502, 503, 504})


def is_transient_httpx_error(exc: Exception) -> bool:
    """Return True if `exc` is a connection/timeout failure or a 429/502/503/504 HTTP status.

    Every other 4xx (validation, auth, not-found) and any malformed-response error
    (`ValueError`/`KeyError`, raised by response-parsing code, never by httpx itself) is
    permanent — retrying a request the server has already told us is invalid or unauthorized
    only wastes time and amplifies load for no chance of success.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in TRANSIENT_STATUS_CODES
    if isinstance(exc, httpx.HTTPError):
        return True
    return False
