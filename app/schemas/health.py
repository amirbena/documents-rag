"""Response schemas for the legacy /api/v1/health endpoint and the unversioned platform
health/readiness endpoints (see app/api/routes/health.py).
"""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Shape returned by GET /api/v1/health (legacy; prefer GET /health)."""

    status: str
    environment: str


class DependencyCheckResult(BaseModel):
    """One dependency's health check outcome — never includes secrets or connection strings."""

    name: str
    status: Literal["ok", "error"]
    required: bool
    detail: str | None = None


class PlatformHealthResponse(BaseModel):
    """Shape returned by GET /health — a lightweight summary, no dependency calls."""

    status: Literal["ok"]
    service: str
    version: str


class LivenessResponse(BaseModel):
    """Shape returned by GET /health/live — process-alive only, no dependency calls."""

    status: Literal["ok"]
    service: str
    version: str


class ReadinessResponse(BaseModel):
    """Shape returned by GET /health/ready — 200 iff every required dependency check passes."""

    status: Literal["ok", "unavailable"]
    service: str
    version: str
    checks: list[DependencyCheckResult]
    error: str | None = None


class DependenciesResponse(BaseModel):
    """Shape returned by GET /health/dependencies — full diagnostic detail, always HTTP 200."""

    status: Literal["ok", "degraded"]
    service: str
    version: str
    checks: list[DependencyCheckResult]
    error: str | None = None
