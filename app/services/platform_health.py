"""Async dependency checks for the unversioned platform health/readiness endpoints.

Each check is a small, timeout-bounded probe (SELECT 1, PING, a lightweight HTTP call) that
never mutates or restarts the dependency it checks and never retries beyond its own timeout.
Every result is safe to return to a client: no credentials, connection strings, stack traces,
or raw provider response bodies — only a fixed, generic detail message per failure mode.
"""

import asyncio

import httpx
import redis.asyncio as redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import Settings
from app.schemas.health import DependencyCheckResult
from app.services.ollama_client import OllamaClient

CHECK_TIMEOUT_SECONDS = 3.0


async def check_postgres(settings: Settings) -> DependencyCheckResult:
    """Run a lightweight `SELECT 1` against Postgres with a short timeout."""
    engine = create_async_engine(settings.database_url)
    try:
        async with asyncio.timeout(CHECK_TIMEOUT_SECONDS):
            async with engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
        return DependencyCheckResult(name="postgres", status="ok", required=True)
    except Exception:
        return DependencyCheckResult(
            name="postgres", status="error", required=True, detail="Postgres is unreachable."
        )
    finally:
        await engine.dispose()


async def check_redis(settings: Settings) -> DependencyCheckResult:
    """PING Redis with a short timeout.

    Not required for readiness today — no application code path reads or writes Redis yet
    (see ARCHITECTURE.md's environment variable table) — but still reported for visibility.
    """
    client = redis.from_url(
        settings.redis_url,
        socket_timeout=CHECK_TIMEOUT_SECONDS,
        socket_connect_timeout=CHECK_TIMEOUT_SECONDS,
    )
    try:
        async with asyncio.timeout(CHECK_TIMEOUT_SECONDS):
            await client.ping()
        return DependencyCheckResult(name="redis", status="ok", required=False)
    except Exception:
        return DependencyCheckResult(
            name="redis", status="error", required=False, detail="Redis is unreachable."
        )
    finally:
        await client.aclose()


async def check_qdrant(settings: Settings) -> DependencyCheckResult:
    """Call Qdrant's lightweight /collections endpoint with a short timeout."""
    try:
        async with httpx.AsyncClient(
            base_url=settings.qdrant_url, timeout=CHECK_TIMEOUT_SECONDS
        ) as client:
            response = await client.get("/collections")
            response.raise_for_status()
        return DependencyCheckResult(name="qdrant", status="ok", required=True)
    except httpx.HTTPError:
        return DependencyCheckResult(
            name="qdrant", status="error", required=True, detail="Qdrant is unreachable."
        )


async def check_ollama(settings: Settings) -> list[DependencyCheckResult]:
    """Reuse OllamaClient.check_health() for reachability and configured-model availability."""
    result = await OllamaClient(settings=settings).check_health()

    ollama_check = DependencyCheckResult(
        name="ollama",
        status="ok" if result.reachable else "error",
        required=True,
        detail=None if result.reachable else "Ollama is unreachable.",
    )
    chat_model_check = DependencyCheckResult(
        name="ollama_chat_model",
        status="ok" if result.chat_model_available else "error",
        required=True,
        detail=(
            None
            if result.chat_model_available
            else f"Configured chat model {settings.ollama_chat_model!r} is not available."
        ),
    )
    embedding_model_check = DependencyCheckResult(
        name="ollama_embedding_model",
        status="ok" if result.embedding_model_available else "error",
        required=True,
        detail=(
            None
            if result.embedding_model_available
            else f"Configured embedding model {settings.ollama_embedding_model!r} is not available."
        ),
    )
    return [ollama_check, chat_model_check, embedding_model_check]


async def run_all_checks(settings: Settings) -> list[DependencyCheckResult]:
    """Run every dependency check concurrently and return their results.

    Never raises: each check function already turns its own failure into an "error" result.
    """
    postgres_result, redis_result, qdrant_result, ollama_results = await asyncio.gather(
        check_postgres(settings),
        check_redis(settings),
        check_qdrant(settings),
        check_ollama(settings),
    )
    return [postgres_result, redis_result, qdrant_result, *ollama_results]
