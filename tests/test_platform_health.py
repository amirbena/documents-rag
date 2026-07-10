"""Tests for the unversioned platform health/liveness/readiness endpoints, no real dependencies."""

import pytest
from fastapi.testclient import TestClient

import app.api.routes.health as platform_health_routes
from app.main import app
from app.schemas.health import DependencyCheckResult

client = TestClient(app)


def _check(name: str, healthy: bool, required: bool, detail: str | None = None) -> DependencyCheckResult:
    return DependencyCheckResult(
        name=name,
        status="ok" if healthy else "error",
        required=required,
        detail=detail,
    )


@pytest.fixture(autouse=True)
def _clear_overrides():
    """Reset FastAPI dependency overrides after each test."""
    yield
    app.dependency_overrides.clear()


def test_health_returns_ok_without_dependency_calls(monkeypatch) -> None:
    """GET /health must return 200 and must never call run_all_checks (no dependency I/O)."""

    async def _fail_if_called(settings=None):
        raise AssertionError("GET /health must not call any dependency check")

    monkeypatch.setattr(platform_health_routes, "run_all_checks", _fail_if_called)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "documents-rag"
    assert "version" in body


def test_liveness_returns_ok_without_dependency_calls(monkeypatch) -> None:
    """GET /health/live must return 200 and must never call run_all_checks (no dependency I/O)."""

    async def _fail_if_called(settings=None):
        raise AssertionError("GET /health/live must not call any dependency check")

    monkeypatch.setattr(platform_health_routes, "run_all_checks", _fail_if_called)

    response = client.get("/health/live")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "documents-rag"


def test_readiness_returns_200_when_all_required_checks_pass(monkeypatch) -> None:
    """All required dependencies healthy (redis optionally down) should still return 200."""

    async def _fake_checks(settings):
        return [
            _check("postgres", True, required=True),
            _check("redis", False, required=False, detail="Redis is unreachable."),
            _check("qdrant", True, required=True),
            _check("ollama", True, required=True),
            _check("ollama_chat_model", True, required=True),
            _check("ollama_embedding_model", True, required=True),
        ]

    monkeypatch.setattr(platform_health_routes, "run_all_checks", _fake_checks)

    response = client.get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["error"] is None
    assert {check["name"] for check in body["checks"]} == {
        "postgres",
        "qdrant",
        "ollama",
        "ollama_chat_model",
        "ollama_embedding_model",
    }


def test_readiness_returns_503_when_a_required_check_fails(monkeypatch) -> None:
    """A single failed required dependency should make readiness 503."""

    async def _fake_checks(settings):
        return [
            _check("postgres", False, required=True, detail="Postgres is unreachable."),
            _check("redis", True, required=False),
            _check("qdrant", True, required=True),
            _check("ollama", True, required=True),
            _check("ollama_chat_model", True, required=True),
            _check("ollama_embedding_model", True, required=True),
        ]

    monkeypatch.setattr(platform_health_routes, "run_all_checks", _fake_checks)

    response = client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unavailable"
    assert "postgres" in body["error"]
    failed = [check for check in body["checks"] if check["status"] == "error"]
    assert [check["name"] for check in failed] == ["postgres"]


def test_readiness_ignores_a_failing_non_required_dependency(monkeypatch) -> None:
    """Redis being down (not required today) must not make readiness 503."""

    async def _fake_checks(settings):
        return [
            _check("postgres", True, required=True),
            _check("redis", False, required=False, detail="Redis is unreachable."),
            _check("qdrant", True, required=True),
            _check("ollama", True, required=True),
            _check("ollama_chat_model", True, required=True),
            _check("ollama_embedding_model", True, required=True),
        ]

    monkeypatch.setattr(platform_health_routes, "run_all_checks", _fake_checks)

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_dependencies_returns_structured_statuses_for_all_six_checks(monkeypatch) -> None:
    """GET /health/dependencies should report all six checks, including non-required redis."""

    async def _fake_checks(settings):
        return [
            _check("postgres", True, required=True),
            _check("redis", False, required=False, detail="Redis is unreachable."),
            _check("qdrant", True, required=True),
            _check("ollama", True, required=True),
            _check("ollama_chat_model", True, required=True),
            _check("ollama_embedding_model", True, required=True),
        ]

    monkeypatch.setattr(platform_health_routes, "run_all_checks", _fake_checks)

    response = client.get("/health/dependencies")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    names = {check["name"] for check in body["checks"]}
    assert names == {
        "postgres",
        "redis",
        "qdrant",
        "ollama",
        "ollama_chat_model",
        "ollama_embedding_model",
    }
    redis_check = next(check for check in body["checks"] if check["name"] == "redis")
    assert redis_check["status"] == "error"
    assert redis_check["required"] is False


def test_dependencies_always_returns_200_even_when_everything_is_down(monkeypatch) -> None:
    """GET /health/dependencies is a diagnostics endpoint — it never returns non-200."""

    async def _fake_checks(settings):
        return [
            _check("postgres", False, required=True, detail="Postgres is unreachable."),
            _check("redis", False, required=False, detail="Redis is unreachable."),
            _check("qdrant", False, required=True, detail="Qdrant is unreachable."),
            _check("ollama", False, required=True, detail="Ollama is unreachable."),
            _check("ollama_chat_model", False, required=True),
            _check("ollama_embedding_model", False, required=True),
        ]

    monkeypatch.setattr(platform_health_routes, "run_all_checks", _fake_checks)

    response = client.get("/health/dependencies")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


def test_no_secrets_or_internal_urls_in_dependency_responses(monkeypatch) -> None:
    """Dependency check details must never leak connection strings, hosts, or credentials."""

    async def _fake_checks(settings):
        return [
            _check("postgres", False, required=True, detail="Postgres is unreachable."),
            _check("redis", False, required=False, detail="Redis is unreachable."),
            _check("qdrant", False, required=True, detail="Qdrant is unreachable."),
            _check("ollama", False, required=True, detail="Ollama is unreachable."),
            _check("ollama_chat_model", False, required=True),
            _check("ollama_embedding_model", False, required=True),
        ]

    monkeypatch.setattr(platform_health_routes, "run_all_checks", _fake_checks)

    response = client.get("/health/dependencies")

    body_text = response.text.lower()
    for leaky_fragment in ("password", "postgres:postgres", "5432", "6379", "6333", "11434", "traceback"):
        assert leaky_fragment not in body_text


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
