"""Response schemas for provider health/status endpoints."""

from pydantic import BaseModel


class OllamaHealthResponse(BaseModel):
    """Shape returned by GET /api/v1/providers/ollama/health."""

    reachable: bool
    chat_model: str
    chat_model_available: bool
    embedding_model: str
    embedding_model_available: bool
    error: str | None = None
