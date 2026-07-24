"""ASGI middleware (Phase 2.10) — correlation ID propagation and the unhandled-exception/CORS
boundary.

Registered in `app/main.py`. `correlation_id_middleware` reads an incoming `X-Correlation-ID`
header if present and well-formed (non-empty), otherwise generates a fresh one; sets it on the
request-scoped `ContextVar` (see `app/core/correlation.py`) for the duration of the request, and
echoes it on every response — including error responses, via the fallback exception handlers.

`UnhandledExceptionCorsBoundary` exists because Starlette always pulls a handler registered for
the bare `Exception` (or `500`) key out to `ServerErrorMiddleware`, which wraps *everything else*
— including `CORSMiddleware`. A handler registered that way builds a correct, sanitized response,
but that response never passes back through `CORSMiddleware`, so a genuinely unhandled exception on
an otherwise-CORS-enabled request silently loses its CORS headers while every `AppError`/
`HTTPException` response (handled inside `ExceptionMiddleware`, further in) keeps them. Registering
this class as ordinary ASGI middleware *inside* `CORSMiddleware` (see the registration order and
comment in `app/main.py`) means its response passes back out through CORS like any other response.
"""

from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.correlation import (
    CORRELATION_ID_HEADER,
    generate_correlation_id,
    set_correlation_id,
)
from app.core.exception_handlers import unhandled_exception_handler


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


class UnhandledExceptionCorsBoundary:
    """Pure-ASGI middleware: catches an exception that escapes routing/`ExceptionMiddleware`
    unhandled, builds the same sanitized 500 response `unhandled_exception_handler` would, sends
    it, then re-raises — mirroring `ServerErrorMiddleware`'s own build-send-then-reraise contract,
    so the exception still reaches the ASGI server's own logging/a `raise_server_exceptions=True`
    test client unchanged from before. `Exception` only (never `BaseException`), so
    `asyncio.CancelledError`/`KeyboardInterrupt` are never caught or delayed here.

    `AppError`/`HTTPException`/`RequestValidationError` never reach this layer at all — FastAPI's
    `ExceptionMiddleware` (further in) handles all three and returns a normal response, so this
    class's `except Exception` never fires for them.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        response_started = False

        async def _send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, _send)
        except Exception as exc:
            request = Request(scope, receive=receive)
            response = await unhandled_exception_handler(request, exc)
            if not response_started:
                await response(scope, receive, send)
            raise
