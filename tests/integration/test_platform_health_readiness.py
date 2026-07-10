"""Integration tests for platform readiness against real ephemeral Postgres and Qdrant.

Postgres and Qdrant checks run for real, against the same Testcontainers-managed services the
rest of the integration suite uses. Redis and Ollama checks are replaced with small deterministic
fakes — spinning up real Redis/Ollama containers (and pulling a real model) for every readiness
run would make this suite far heavier for no extra confidence, since those checks are already
covered structurally by the unit suite.
"""

from fastapi.testclient import TestClient

import app.services.platform_health as platform_health_service
from app.core.config import Settings, get_settings
from app.main import app
from app.schemas.health import DependencyCheckResult

client = TestClient(app)


async def _fake_healthy_redis(settings: Settings) -> DependencyCheckResult:
    return DependencyCheckResult(name="redis", status="ok", required=False)


async def _fake_healthy_ollama(settings: Settings) -> list[DependencyCheckResult]:
    return [
        DependencyCheckResult(name="ollama", status="ok", required=True),
        DependencyCheckResult(name="ollama_chat_model", status="ok", required=True),
        DependencyCheckResult(name="ollama_embedding_model", status="ok", required=True),
    ]


def _patch_redis_and_ollama(monkeypatch) -> None:
    monkeypatch.setattr(platform_health_service, "check_redis", _fake_healthy_redis)
    monkeypatch.setattr(platform_health_service, "check_ollama", _fake_healthy_ollama)


def test_readiness_is_200_when_real_postgres_and_qdrant_are_healthy(
    migrated_schema: None, monkeypatch
) -> None:
    """With real Postgres/Qdrant reachable and Redis/Ollama faked healthy, readiness is 200."""
    _patch_redis_and_ollama(monkeypatch)

    response = client.get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    postgres_check = next(check for check in body["checks"] if check["name"] == "postgres")
    qdrant_check = next(check for check in body["checks"] if check["name"] == "qdrant")
    assert postgres_check["status"] == "ok"
    assert qdrant_check["status"] == "ok"


def test_readiness_is_503_when_qdrant_is_unreachable(
    migrated_schema: None, qdrant_url: str, monkeypatch
) -> None:
    """A real, genuinely unreachable Qdrant URL should make readiness 503."""
    _patch_redis_and_ollama(monkeypatch)
    broken_settings = get_settings().model_copy(update={"qdrant_url": "http://127.0.0.1:1"})
    app.dependency_overrides[get_settings] = lambda: broken_settings

    try:
        response = client.get("/health/ready")
    finally:
        app.dependency_overrides.pop(get_settings, None)

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unavailable"
    qdrant_check = next(check for check in body["checks"] if check["name"] == "qdrant")
    assert qdrant_check["status"] == "error"
    postgres_check = next(check for check in body["checks"] if check["name"] == "postgres")
    assert postgres_check["status"] == "ok", "postgres should stay healthy independent of qdrant"


def test_liveness_stays_200_even_when_readiness_would_fail(
    migrated_schema: None, monkeypatch
) -> None:
    """GET /health/live must report 200 regardless of any dependency's real state."""
    broken_settings = get_settings().model_copy(update={"qdrant_url": "http://127.0.0.1:1"})
    app.dependency_overrides[get_settings] = lambda: broken_settings

    try:
        readiness_response = client.get("/health/ready")
        liveness_response = client.get("/health/live")
    finally:
        app.dependency_overrides.pop(get_settings, None)

    assert readiness_response.status_code == 503
    assert liveness_response.status_code == 200
    assert liveness_response.json()["status"] == "ok"


def test_dependencies_response_identifies_the_failed_dependency(
    migrated_schema: None, monkeypatch
) -> None:
    """GET /health/dependencies should name the specific dependency that's actually down."""
    _patch_redis_and_ollama(monkeypatch)
    broken_settings = get_settings().model_copy(update={"qdrant_url": "http://127.0.0.1:1"})
    app.dependency_overrides[get_settings] = lambda: broken_settings

    try:
        response = client.get("/health/dependencies")
    finally:
        app.dependency_overrides.pop(get_settings, None)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert "qdrant" in body["error"]
    failed_names = {check["name"] for check in body["checks"] if check["status"] == "error"}
    assert failed_names == {"qdrant"}


def test_no_secrets_or_internal_urls_leak_in_readiness_or_dependencies(
    migrated_schema: None, postgres_url: str, qdrant_url: str, monkeypatch
) -> None:
    """Even with real connection URLs configured, responses must never echo them back."""
    _patch_redis_and_ollama(monkeypatch)

    ready_text = client.get("/health/ready").text
    dependencies_text = client.get("/health/dependencies").text

    for response_text in (ready_text, dependencies_text):
        assert postgres_url not in response_text
        assert "test:test" not in response_text, "postgres credentials must never appear"
        for fragment in ("asyncpg://", "password", "traceback", "Traceback"):
            assert fragment not in response_text
