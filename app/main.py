"""FastAPI application entrypoint. Wires routers, cross-cutting middleware, and the fallback
exception handlers; no business logic here."""

from fastapi import FastAPI

from app.api.routes import health as platform_health
from app.api.v1.routes import chat, documents, providers, reconciliation, reindex
from app.core.config import get_settings
from app.core.errors import AppError
from app.core.exception_handlers import app_error_handler, unhandled_exception_handler
from app.core.version import SERVICE_NAME, SERVICE_VERSION

settings = get_settings()

app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION)

# Fallback net only — every route's own outcome-table/try-except mapping is checked first by
# FastAPI (more specific handlers, including the built-in HTTPException one, always win); these
# only run for an exception that reaches this boundary unhandled. See exception_handlers.py.
# Starlette's add_exception_handler stub types every handler's second parameter as the base
# Exception, regardless of the exception class registered against — a known typing limitation for
# this exact, common FastAPI pattern (a handler registered for one specific class is only ever
# invoked with an instance of that class). Narrower parameter type is intentional and safe here.
app.add_exception_handler(AppError, app_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(Exception, unhandled_exception_handler)

app.include_router(platform_health.router, tags=["platform-health"])
app.include_router(providers.router, prefix="/api/v1", tags=["providers"])
app.include_router(documents.router, prefix="/api/v1", tags=["documents"])
app.include_router(reindex.router, prefix="/api/v1", tags=["reindex"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(reconciliation.router, prefix="/api/v1", tags=["reconciliation"])
