"""Unversioned platform health/liveness/readiness endpoints.

Stable, version-independent contract for Kubernetes probes, load balancers, monitoring, and
deployment automation — see "Operational Health Contract" in ARCHITECTURE.md. Registered without
an /api/v1 prefix and must never move under one: business API versioning is orthogonal to
operational health, and moving these would break every external prober pointed at them.

Thin-controller routes only: dependency injection, one call into app/services/platform_health.py,
apply the HTTP status the service already computed, return the response body the service already
built. All required-check filtering, failed-check calculation, overall-status calculation, safe
error-summary construction, and readiness/dependencies response construction live in the service
module — see CLAUDE.md's "Operational Endpoints" / route-layer standing rule.
"""

from fastapi import APIRouter, Depends, Response

from app.core.config import Settings, get_settings
from app.core.version import SERVICE_NAME, SERVICE_VERSION
from app.schemas.health import (
    DependenciesResponse,
    LivenessResponse,
    PlatformHealthResponse,
    ReadinessResponse,
)
from app.services.platform_health import get_dependencies_response, get_readiness_result

router = APIRouter()


@router.get("/health", response_model=PlatformHealthResponse)
async def platform_health() -> PlatformHealthResponse:
    """Lightweight platform summary — 200 while the process is running, no dependency calls."""
    return PlatformHealthResponse(status="ok", service=SERVICE_NAME, version=SERVICE_VERSION)


@router.get("/health/live", response_model=LivenessResponse)
async def liveness() -> LivenessResponse:
    """Liveness probe — 200 while the process is alive; never calls any external service."""
    return LivenessResponse(status="ok", service=SERVICE_NAME, version=SERVICE_VERSION)


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness(
    response: Response, settings: Settings = Depends(get_settings)
) -> ReadinessResponse:
    """Readiness probe — delegates check aggregation to the service, only applies the HTTP status."""
    result = await get_readiness_result(settings)
    response.status_code = result.status_code
    return result.response


@router.get("/health/dependencies", response_model=DependenciesResponse)
async def dependencies(settings: Settings = Depends(get_settings)) -> DependenciesResponse:
    """Detailed dependency status for every checked dependency — always returns HTTP 200.

    A diagnostics/monitoring endpoint, not a gating probe — use GET /health/ready for that.
    """
    return await get_dependencies_response(settings)
