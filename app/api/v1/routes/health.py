"""Liveness/readiness endpoint. Reports app status only, no dependency checks yet."""

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.schemas.health import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Return a static ok status plus the running environment name."""
    return HealthResponse(status="ok", environment=settings.app_env)
