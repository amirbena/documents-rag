"""Provider status endpoints. Only Ollama health/model-availability exists so far."""

from fastapi import APIRouter, Depends, Response, status

from app.core.config import Settings, get_settings
from app.schemas.providers import OllamaHealthResponse
from app.services.ollama_client import OllamaClient

router = APIRouter()


def get_ollama_client(settings: Settings = Depends(get_settings)) -> OllamaClient:
    """Build an OllamaClient wired to the current app settings."""
    return OllamaClient(settings=settings)


@router.get("/providers/ollama/health", response_model=OllamaHealthResponse)
async def ollama_health(
    response: Response,
    settings: Settings = Depends(get_settings),
    client: OllamaClient = Depends(get_ollama_client),
) -> OllamaHealthResponse:
    """Report Ollama reachability and whether the configured chat/embedding models exist."""
    result = await client.check_health()

    if not (result.reachable and result.chat_model_available and result.embedding_model_available):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return OllamaHealthResponse(
        reachable=result.reachable,
        chat_model=settings.ollama_chat_model,
        chat_model_available=result.chat_model_available,
        embedding_model=settings.ollama_embedding_model,
        embedding_model_available=result.embedding_model_available,
        error=result.error,
    )
