"""FastAPI application entrypoint. Wires routers; no business logic here."""

from fastapi import FastAPI

from app.api.routes import health as platform_health
from app.api.v1.routes import chat, documents, providers, reindex
from app.core.config import get_settings
from app.core.version import SERVICE_NAME, SERVICE_VERSION

settings = get_settings()

app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION)

app.include_router(platform_health.router, tags=["platform-health"])
app.include_router(providers.router, prefix="/api/v1", tags=["providers"])
app.include_router(documents.router, prefix="/api/v1", tags=["documents"])
app.include_router(reindex.router, prefix="/api/v1", tags=["reindex"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
