"""Unversioned platform health/liveness/readiness endpoints.

Stable, version-independent contract for Kubernetes probes, load balancers, monitoring, and
deployment automation — see "Operational Health Contract" in ARCHITECTURE.md. Registered without
an /api/v1 prefix and must never move under one: business API versioning is orthogonal to
operational health, and moving these would break every external prober pointed at them.
"""

from fastapi import APIRouter, Depends, Response, status

from app.core.config import Settings, get_settings
from app.core.version import SERVICE_NAME, SERVICE_VERSION
from app.schemas.health import (
    DependenciesResponse,
    LivenessResponse,
    PlatformHealthResponse,
    ReadinessResponse,
)
from app.services.platform_health import run_all_checks

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
    """Readiness probe — 200 only if every required dependency check passes, else 503.

    Never mutates or restarts a dependency, and never retries beyond each check's own timeout.
    """
    checks = await run_all_checks(settings)
    required_checks = [check for check in checks if check.required]
    failed_required = [check for check in required_checks if check.status == "error"]

    if failed_required:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        failed_names = ", ".join(check.name for check in failed_required)
        return ReadinessResponse(
            status="unavailable",
            service=SERVICE_NAME,
            version=SERVICE_VERSION,
            checks=required_checks,
            error=f"Required dependencies not ready: {failed_names}.",
        )

    return ReadinessResponse(
        status="ok", service=SERVICE_NAME, version=SERVICE_VERSION, checks=required_checks
    )


@router.get("/health/dependencies", response_model=DependenciesResponse)
async def dependencies(settings: Settings = Depends(get_settings)) -> DependenciesResponse:
    """Detailed dependency status for every checked dependency — always returns HTTP 200.

    A diagnostics/monitoring endpoint, not a gating probe — use GET /health/ready for that.
    """
    checks = await run_all_checks(settings)
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
