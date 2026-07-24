"""Deterministic FastAPI startup/shutdown lifecycle (Phase 2.10).

Startup never probes PostgreSQL/Qdrant/MinIO/Redis/Ollama reachability — `GET /health/ready`
(`app/services/platform_health.py`) remains the sole dependency-readiness mechanism, so a
temporarily unreachable remote dependency never fails application startup. This lifespan only
owns resources this process itself allocates: today, that is exactly the shared SQLAlchemy engine
(`app/db/session.py`). It never constructs a provider client — every provider client in this
codebase (Ollama, Qdrant, MinIO) is already created and closed per operation, so there is nothing
process-lifetime-scoped for a lifespan to hold open or close.

Resources are registered on an `AsyncExitStack` as they are acquired, so if a later startup step
were ever added and raised, everything registered before it still gets released — a startup
failure never leaks an already-initialized resource, even though there is only one such resource
today.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)


def build_lifespan(engine: AsyncEngine) -> Any:
    """Return a lifespan context manager that disposes `engine` on shutdown.

    Factory form (rather than a module-level function closing over the global engine) so tests
    can drive the lifecycle against a stand-in engine without touching the real one.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            logger.info("Application startup beginning.", extra={"event": "app_startup_begin"})
            stack.push_async_callback(engine.dispose)
            logger.info("Application startup complete.", extra={"event": "app_startup_complete"})
            yield
            logger.info("Application shutdown beginning.", extra={"event": "app_shutdown_begin"})
        logger.info("Application shutdown complete.", extra={"event": "app_shutdown_complete"})

    return lifespan
