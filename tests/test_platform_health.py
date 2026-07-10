"""Route-level tests for the platform health endpoints: HTTP mapping only, no aggregation logic.

Aggregation (required-check filtering, failed-check calculation, overall status, error summary,
response construction) lives in app/services/platform_health.py and is tested directly in
tests/test_platform_health_service.py — these tests only confirm the route wires dependency
injection, calls the service, and maps the service's result onto the HTTP response correctly.
"""

import inspect

import pytest
from fastapi.testclient import TestClient

import app.api.routes.health as platform_health_routes
from app.main import app
from app.schemas.health import DependenciesResponse, ReadinessResponse
from app.services.platform_health import ReadinessResult

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides():
    """Reset FastAPI dependency overrides after each test."""
    yield
    app.dependency_overrides.clear()


def test_health_returns_ok_without_dependency_calls(monkeypatch) -> None:
    """GET /health must return 200 and must never call the service layer (no dependency I/O)."""

    async def _fail_if_called(settings=None):
        raise AssertionError("GET /health must not call any dependency check")

    monkeypatch.setattr(platform_health_routes, "get_readiness_result", _fail_if_called)
    monkeypatch.setattr(platform_health_routes, "get_dependencies_response", _fail_if_called)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "documents-rag"
    assert "version" in body


def test_liveness_returns_ok_without_dependency_calls(monkeypatch) -> None:
    """GET /health/live must return 200 and must never call the service layer (no dependency I/O)."""

    async def _fail_if_called(settings=None):
        raise AssertionError("GET /health/live must not call any dependency check")

    monkeypatch.setattr(platform_health_routes, "get_readiness_result", _fail_if_called)
    monkeypatch.setattr(platform_health_routes, "get_dependencies_response", _fail_if_called)

    response = client.get("/health/live")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "documents-rag"


def test_readiness_maps_service_ok_result_to_200(monkeypatch) -> None:
    """The route should return exactly the body the service builds, with its status code applied."""
    fake_response = ReadinessResponse(
        status="ok", service="documents-rag", version="0.1.0", checks=[]
    )

    async def _fake_get_readiness_result(settings):
        return ReadinessResult(response=fake_response, status_code=200)

    monkeypatch.setattr(
        platform_health_routes, "get_readiness_result", _fake_get_readiness_result
    )

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == fake_response.model_dump()


def test_readiness_maps_service_unavailable_result_to_503(monkeypatch) -> None:
    """The route must apply whatever status code the service computed — including 503."""
    fake_response = ReadinessResponse(
        status="unavailable",
        service="documents-rag",
        version="0.1.0",
        checks=[],
        error="Required dependencies not ready: postgres.",
    )

    async def _fake_get_readiness_result(settings):
        return ReadinessResult(response=fake_response, status_code=503)

    monkeypatch.setattr(
        platform_health_routes, "get_readiness_result", _fake_get_readiness_result
    )

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == fake_response.model_dump()


def test_dependencies_returns_service_response_verbatim_and_always_200(monkeypatch) -> None:
    """GET /health/dependencies should return exactly the service's response, always as HTTP 200."""
    fake_response = DependenciesResponse(
        status="degraded",
        service="documents-rag",
        version="0.1.0",
        checks=[],
        error="1 of 6 dependency checks failed: redis.",
    )

    async def _fake_get_dependencies_response(settings):
        return fake_response

    monkeypatch.setattr(
        platform_health_routes, "get_dependencies_response", _fake_get_dependencies_response
    )

    response = client.get("/health/dependencies")

    assert response.status_code == 200
    assert response.json() == fake_response.model_dump()


def test_route_module_does_not_perform_check_aggregation_itself() -> None:
    """The route module must delegate aggregation entirely — no filtering/status logic inline."""
    source = inspect.getsource(platform_health_routes)

    # The route must not import the low-level check runner or check-result type directly —
    # only the already-aggregated service entry points and response schemas.
    assert "run_all_checks" not in source
    assert "DependencyCheckResult" not in source
    # No inline required-check filtering, failure counting, or status/error-summary construction.
    assert ".required" not in source
    assert "status == \"error\"" not in source
    assert "degraded" not in source
    assert "unavailable" not in source


def test_business_routes_under_api_v1_remain_unchanged() -> None:
    """The legacy /api/v1/health endpoint must keep working exactly as before."""
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["environment"] == "local"


def test_platform_health_endpoints_are_unversioned() -> None:
    """The three new platform endpoints (unlike legacy /api/v1/health) must not exist under /api/v1."""
    for path in ("/health/live", "/health/ready", "/health/dependencies"):
        assert client.get(f"/api/v1{path}").status_code == 404

    for path in ("/health", "/health/live"):
        assert client.get(path).status_code == 200
