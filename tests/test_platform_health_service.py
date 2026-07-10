"""Service-level tests for readiness/dependency aggregation — pure, no I/O, no FastAPI.

Covers app/services/platform_health.py's build_readiness_result/build_dependencies_response
(required-check filtering, failed-check calculation, overall status, safe error-summary) plus a
thin check on the async orchestration wrappers that call run_all_checks() and delegate to them.
"""

import app.services.platform_health as platform_health_service
from app.schemas.health import DependencyCheckResult
from app.services.platform_health import (
    build_dependencies_response,
    build_readiness_result,
    get_dependencies_response,
    get_readiness_result,
)


def _check(name: str, healthy: bool, required: bool, detail: str | None = None) -> DependencyCheckResult:
    return DependencyCheckResult(
        name=name, status="ok" if healthy else "error", required=required, detail=detail
    )


def test_build_readiness_result_is_ok_when_all_required_checks_pass() -> None:
    """All required checks passing (redis optionally down) should yield status 200."""
    checks = [
        _check("postgres", True, required=True),
        _check("redis", False, required=False, detail="Redis is unreachable."),
        _check("qdrant", True, required=True),
        _check("ollama", True, required=True),
        _check("ollama_chat_model", True, required=True),
        _check("ollama_embedding_model", True, required=True),
    ]

    result = build_readiness_result(checks)

    assert result.status_code == 200
    assert result.response.status == "ok"
    assert result.response.error is None
    assert {check.name for check in result.response.checks} == {
        "postgres",
        "qdrant",
        "ollama",
        "ollama_chat_model",
        "ollama_embedding_model",
    }


def test_build_readiness_result_is_unavailable_when_a_required_check_fails() -> None:
    """A single failed required dependency should yield status 503 and name it in the error."""
    checks = [
        _check("postgres", False, required=True, detail="Postgres is unreachable."),
        _check("redis", True, required=False),
        _check("qdrant", True, required=True),
        _check("ollama", True, required=True),
        _check("ollama_chat_model", True, required=True),
        _check("ollama_embedding_model", True, required=True),
    ]

    result = build_readiness_result(checks)

    assert result.status_code == 503
    assert result.response.status == "unavailable"
    assert "postgres" in result.response.error
    failed = [check for check in result.response.checks if check.status == "error"]
    assert [check.name for check in failed] == ["postgres"]


def test_build_readiness_result_ignores_a_failing_non_required_dependency() -> None:
    """Redis being down (not required today) must not make readiness unavailable."""
    checks = [
        _check("postgres", True, required=True),
        _check("redis", False, required=False, detail="Redis is unreachable."),
        _check("qdrant", True, required=True),
        _check("ollama", True, required=True),
        _check("ollama_chat_model", True, required=True),
        _check("ollama_embedding_model", True, required=True),
    ]

    result = build_readiness_result(checks)

    assert result.status_code == 200
    assert result.response.status == "ok"


def test_build_readiness_result_excludes_non_required_checks_from_response() -> None:
    """Non-required checks (redis) must never appear in the readiness response's checks list."""
    checks = [_check("postgres", True, required=True), _check("redis", True, required=False)]

    result = build_readiness_result(checks)

    assert {check.name for check in result.response.checks} == {"postgres"}


def test_build_dependencies_response_is_ok_when_everything_passes() -> None:
    """All checks passing should yield overall status ok and no error summary."""
    checks = [_check("postgres", True, required=True), _check("redis", True, required=False)]

    response = build_dependencies_response(checks)

    assert response.status == "ok"
    assert response.error is None
    assert len(response.checks) == 2


def test_build_dependencies_response_is_degraded_when_any_check_fails() -> None:
    """Even a non-required failure (redis) should mark the overall status degraded."""
    checks = [
        _check("postgres", True, required=True),
        _check("redis", False, required=False, detail="Redis is unreachable."),
    ]

    response = build_dependencies_response(checks)

    assert response.status == "degraded"
    assert "redis" in response.error
    assert "1 of 2" in response.error


def test_build_dependencies_response_includes_all_checks_including_non_required() -> None:
    """Unlike readiness, the dependencies response must include every check, required or not."""
    checks = [
        _check("postgres", True, required=True),
        _check("redis", False, required=False, detail="Redis is unreachable."),
        _check("qdrant", True, required=True),
    ]

    response = build_dependencies_response(checks)

    assert {check.name for check in response.checks} == {"postgres", "redis", "qdrant"}


async def test_get_readiness_result_runs_checks_and_delegates_to_build_readiness_result(
    monkeypatch,
) -> None:
    """The async orchestration wrapper should call run_all_checks() then aggregate the result."""
    checks = [_check("postgres", True, required=True)]

    async def _fake_run_all_checks(settings):
        return checks

    monkeypatch.setattr(platform_health_service, "run_all_checks", _fake_run_all_checks)

    result = await get_readiness_result(settings=None)

    assert result.status_code == 200
    assert result.response.status == "ok"


async def test_get_dependencies_response_runs_checks_and_delegates_to_build_dependencies_response(
    monkeypatch,
) -> None:
    """The async orchestration wrapper should call run_all_checks() then aggregate the result."""
    checks = [_check("postgres", False, required=True, detail="Postgres is unreachable.")]

    async def _fake_run_all_checks(settings):
        return checks

    monkeypatch.setattr(platform_health_service, "run_all_checks", _fake_run_all_checks)

    response = await get_dependencies_response(settings=None)

    assert response.status == "degraded"
    assert "postgres" in response.error
