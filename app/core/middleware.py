"""ASGI middleware (Phase 2.10) — currently just correlation ID propagation.

Registered in `app/main.py`. Reads an incoming `X-Correlation-ID` header if present and
well-formed (non-empty), otherwise generates a fresh one; sets it on the request-scoped
`ContextVar` (see `app/core/correlation.py`) for the duration of the request, and echoes it on
every response — including error responses, via the fallback exception handlers.
"""

from collections.abc import Awaitable, Callable

from fastapi import Request, Response

from app.core.correlation import (
    CORRELATION_ID_HEADER,
    generate_correlation_id,
    set_correlation_id,
)


async def correlation_id_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Assign (or accept) this request's correlation ID and echo it on the response."""
    incoming = request.headers.get(CORRELATION_ID_HEADER, "").strip()
    correlation_id = incoming or generate_correlation_id()
    set_correlation_id(correlation_id)

    response = await call_next(request)
    response.headers[CORRELATION_ID_HEADER] = correlation_id
    return response
