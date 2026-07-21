"""Centralized FastAPI exception handlers (Phase 2.10) — a fallback net, not the primary path.

Every route's own outcome-table/try-except mapping (e.g. `documents.py`'s `_RETRY_OUTCOME_ERRORS`)
remains the first, most-specific, and authoritative mapping for its own domain — these handlers
only ever run for an exception that reaches FastAPI's boundary unhandled: a new `AppError` raised
by lifespan/config/retry code, or a genuinely unexpected exception. Both preserve the existing
`{"detail": "..."}` response shape and never return `str(exc)` or a stack trace.
"""

import logging

from fastapi import Request, status
from fastapi.responses import JSONResponse

from app.core.correlation import CORRELATION_ID_HEADER, get_correlation_id
from app.core.errors import AppError

logger = logging.getLogger(__name__)

_UNEXPECTED_ERROR_DETAIL = "Internal server error."


def _error_response(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail},
        headers={CORRELATION_ID_HEADER: get_correlation_id()},
    )


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Map an AppError to its declared status code; log once with category/correlation context."""
    logger.error(
        "Unhandled AppError reached the fallback exception handler.",
        extra={
            "event": "app_error_fallback",
            "path": request.url.path,
            "error_category": exc.code,
            "correlation_id": get_correlation_id(),
        },
    )
    return _error_response(exc.status_code, exc.message)


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Map any other unhandled exception to a fixed, safe 500 — never leak str(exc)."""
    logger.exception(
        "Unhandled exception reached the fallback exception handler.",
        extra={
            "event": "unhandled_exception_fallback",
            "path": request.url.path,
            "correlation_id": get_correlation_id(),
        },
    )
    return _error_response(status.HTTP_500_INTERNAL_SERVER_ERROR, _UNEXPECTED_ERROR_DETAIL)
