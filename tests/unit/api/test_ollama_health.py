"""Tests for GET /api/v1/providers/ollama/health with a mocked Ollama HTTP transport."""

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.v1.routes.providers import get_ollama_client
from app.core.config import get_settings
from app.main import app
from app.services.ollama_client import OllamaClient

client = TestClient(app)


def _tags_transport(model_names: list[str]) -> httpx.MockTransport:
    """Build a mock transport whose /api/tags response lists the given model names."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": name} for name in model_names]})

    return httpx.MockTransport(handler)


def _failing_transport() -> httpx.MockTransport:
    """Build a mock transport that simulates an unreachable Ollama server."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    return httpx.MockTransport(handler)


@pytest.fixture(autouse=True)
def _clear_overrides():
    """Reset FastAPI dependency overrides after each test."""
    yield
    app.dependency_overrides.clear()


def test_ollama_health_reachable_with_both_models_available() -> None:
    """Both models present should report 200 with all flags true."""
    transport = _tags_transport(["llama3.1:latest", "bge-m3:latest"])
    app.dependency_overrides[get_ollama_client] = lambda: OllamaClient(
        settings=get_settings(), transport=transport
    )

    response = client.get("/api/v1/providers/ollama/health")

    assert response.status_code == 200
    body = response.json()
    assert body["reachable"] is True
    assert body["chat_model_available"] is True
    assert body["embedding_model_available"] is True
    assert body["error"] is None


def test_ollama_health_reachable_with_one_model_missing() -> None:
    """Missing embedding model should report 503 with only chat_model_available true."""
    transport = _tags_transport(["llama3.1:latest"])
    app.dependency_overrides[get_ollama_client] = lambda: OllamaClient(
        settings=get_settings(), transport=transport
    )

    response = client.get("/api/v1/providers/ollama/health")

    assert response.status_code == 503
    body = response.json()
    assert body["reachable"] is True
    assert body["chat_model_available"] is True
    assert body["embedding_model_available"] is False


def test_ollama_health_unreachable() -> None:
    """Connection failure should report 503 with reachable=False and an error message."""
    app.dependency_overrides[get_ollama_client] = lambda: OllamaClient(
        settings=get_settings(), transport=_failing_transport()
    )

    response = client.get("/api/v1/providers/ollama/health")

    assert response.status_code == 503
    body = response.json()
    assert body["reachable"] is False
    assert body["chat_model_available"] is False
    assert body["embedding_model_available"] is False
    assert body["error"] is not None
