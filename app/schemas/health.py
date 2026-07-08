"""Response schemas for the /health endpoint."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Shape returned by GET /api/v1/health."""

    status: str
    environment: str
