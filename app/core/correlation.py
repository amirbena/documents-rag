"""Request correlation ID: a `ContextVar`, never mutable global state.

`CorrelationIdMiddleware` (app/core/middleware.py) is the only writer — it reads an incoming
`X-Correlation-ID` header or generates a new UUID4, sets it here for the duration of the request,
and echoes it on the response. Everything else (logging, outbound provider calls, exception
handlers) only ever reads it via `get_correlation_id()`.
"""

import uuid
from contextvars import ContextVar

CORRELATION_ID_HEADER = "X-Correlation-ID"

_NO_CORRELATION_ID = "-"

_correlation_id: ContextVar[str] = ContextVar("correlation_id", default=_NO_CORRELATION_ID)


def generate_correlation_id() -> str:
    """Return a freshly generated correlation ID (UUID4, string form)."""
    return str(uuid.uuid4())


def get_correlation_id() -> str:
    """Return the current request's correlation ID, or a fixed placeholder outside any request."""
    return _correlation_id.get()


def set_correlation_id(value: str) -> None:
    """Set the correlation ID for the current context (task/request)."""
    _correlation_id.set(value)


def correlation_headers() -> dict[str, str]:
    """Return a single-entry header dict carrying the current correlation ID, for outbound calls."""
    return {CORRELATION_ID_HEADER: get_correlation_id()}
