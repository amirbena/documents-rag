"""FastAPI application entrypoint. Wires routers; no business logic here."""

from fastapi import FastAPI

from app.api.v1.routes import health, providers
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(title="documents-rag", version="0.1.0")

app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(providers.router, prefix="/api/v1", tags=["providers"])
