"""Async dependency checks and readiness/dependencies aggregation for the unversioned platform
health endpoints.

Each check is a small, timeout-bounded probe (SELECT 1, PING, a lightweight HTTP call) that
never mutates or restarts the dependency it checks and never retries beyond its own timeout.
Every result is safe to return to a client: no credentials, connection strings, stack traces,
or raw provider response bodies — only a fixed, generic detail message per failure mode.

This module also owns all readiness/dependencies aggregation (required-check filtering,
failed-check calculation, overall status, safe error-summary, response construction) so the
route module (app/api/routes/health.py) stays a thin controller: dependency injection, one call
into this module, apply the returned HTTP status, return the returned response body.
"""

import asyncio
from dataclasses import dataclass

import httpx
import redis.asyncio as redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import Settings
from app.core.version import SERVICE_NAME, SERVICE_VERSION
from app.schemas.health import DependenciesResponse, DependencyCheckResult, ReadinessResponse
from app.services.ollama_client import OllamaClient

CHECK_TIMEOUT_SECONDS = 3.0

_HTTP_OK = 200
_HTTP_SERVICE_UNAVAILABLE = 503


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


@dataclass
class ReadinessResult:
    """Typed service result for GET /health/ready: the response body plus the HTTP status to apply."""

    response: ReadinessResponse
    status_code: int


def build_readiness_result(checks: list[DependencyCheckResult]) -> ReadinessResult:
    """Aggregate dependency checks into a ReadinessResponse + HTTP status.

    Only required checks gate readiness — a failing non-required check (e.g. redis) is dropped
    before this point and never appears in the readiness response. Pure and synchronous: no I/O,
    so it can be tested directly against a fabricated list of checks.
    """
    required_checks = [check for check in checks if check.required]
    failed_required = [check for check in required_checks if check.status == "error"]

    if failed_required:
        failed_names = ", ".join(check.name for check in failed_required)
        response = ReadinessResponse(
            status="unavailable",
            service=SERVICE_NAME,
            version=SERVICE_VERSION,
            checks=required_checks,
            error=f"Required dependencies not ready: {failed_names}.",
        )
        return ReadinessResult(response=response, status_code=_HTTP_SERVICE_UNAVAILABLE)

    response = ReadinessResponse(
        status="ok", service=SERVICE_NAME, version=SERVICE_VERSION, checks=required_checks
    )
    return ReadinessResult(response=response, status_code=_HTTP_OK)


def build_dependencies_response(checks: list[DependencyCheckResult]) -> DependenciesResponse:
    """Aggregate every dependency check (required and not) into a DependenciesResponse.

    Always maps to HTTP 200 at the route — this is a diagnostics response, not a gating probe.
    Pure and synchronous: no I/O, so it can be tested directly against a fabricated check list.
    """
    failed = [check for check in checks if check.status == "error"]
    return DependenciesResponse(
        status="degraded" if failed else "ok",
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        checks=checks,
        error=(
            f"{len(failed)} of {len(checks)} dependency checks failed: "
            f"{', '.join(check.name for check in failed)}."
            if failed
            else None
        ),
    )


async def get_readiness_result(settings: Settings) -> ReadinessResult:
    """Run every dependency check and aggregate them into a ReadinessResult."""
    checks = await run_all_checks(settings)
    return build_readiness_result(checks)


async def get_dependencies_response(settings: Settings) -> DependenciesResponse:
    """Run every dependency check and aggregate them into a DependenciesResponse."""
    checks = await run_all_checks(settings)
    return build_dependencies_response(checks)
