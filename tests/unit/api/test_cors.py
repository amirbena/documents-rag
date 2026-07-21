"""Tests for CORS wiring (Phase 2.10, app/core/cors.py + its registration in app/main.py).

Builds a small standalone FastAPI app per test, wired with the exact same
`cors_middleware_kwargs()`/`correlation_id_middleware` registration order as app/main.py, so these
tests exercise the real policy and real Starlette CORSMiddleware behavior without depending on
whatever CORS_ALLOW_ORIGINS happens to be set in the ambient test environment (the module-level
`app.main.app` singleton is built once at import time from real settings, so it can't be
reconfigured per test).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.correlation import CORRELATION_ID_HEADER
from app.core.cors import cors_middleware_kwargs
from app.core.middleware import correlation_id_middleware

_ALLOWED_ORIGIN = "http://localhost:3000"
_DISALLOWED_ORIGIN = "http://evil.example.com"


def _build_test_app(*, cors_allow_origins: str) -> FastAPI:
    settings = Settings(CORS_ALLOW_ORIGINS=cors_allow_origins)
    app = FastAPI()
    # Same registration order as app/main.py: CORS added first, correlation ID added last (and
    # therefore outermost) — see app/main.py's own comment for why the order matters.
    app.add_middleware(CORSMiddleware, **cors_middleware_kwargs(settings))
    app.middleware("http")(correlation_id_middleware)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.delete("/documents/{document_id}")
    async def delete_document(document_id: str) -> dict[str, str]:
        return {"status": "deleted"}

    return app


def test_configured_allowed_origin_receives_cors_headers() -> None:
    client = TestClient(_build_test_app(cors_allow_origins=_ALLOWED_ORIGIN))

    response = client.get("/health", headers={"Origin": _ALLOWED_ORIGIN})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == _ALLOWED_ORIGIN


def test_disallowed_origin_receives_no_allow_origin_header() -> None:
    client = TestClient(_build_test_app(cors_allow_origins=_ALLOWED_ORIGIN))

    response = client.get("/health", headers={"Origin": _DISALLOWED_ORIGIN})

    assert response.status_code == 200  # the server still answers — enforcement is browser-side
    assert "access-control-allow-origin" not in response.headers


def test_preflight_succeeds_for_an_allowed_origin_and_allowed_method() -> None:
    client = TestClient(_build_test_app(cors_allow_origins=_ALLOWED_ORIGIN))

    response = client.options(
        "/documents/doc-1",
        headers={
            "Origin": _ALLOWED_ORIGIN,
            "Access-Control-Request-Method": "DELETE",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == _ALLOWED_ORIGIN
    assert "DELETE" in response.headers["access-control-allow-methods"]


def test_preflight_from_a_disallowed_origin_is_rejected() -> None:
    client = TestClient(_build_test_app(cors_allow_origins=_ALLOWED_ORIGIN))

    response = client.options(
        "/documents/doc-1",
        headers={
            "Origin": _DISALLOWED_ORIGIN,
            "Access-Control-Request-Method": "DELETE",
        },
    )

    # Starlette answers preflight itself (400, "Disallowed CORS origin") rather than forwarding to
    # the route — either way, the disallowed origin must never be reflected back as allowed.
    assert response.status_code == 400
    assert response.headers.get("access-control-allow-origin") != _DISALLOWED_ORIGIN


def test_requests_without_an_origin_header_are_unaffected() -> None:
    client = TestClient(_build_test_app(cors_allow_origins=_ALLOWED_ORIGIN))

    response = client.get("/health")

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_correlation_id_remains_present_and_unchanged_on_a_cors_enabled_response() -> None:
    client = TestClient(_build_test_app(cors_allow_origins=_ALLOWED_ORIGIN))
    incoming_correlation_id = "operator-supplied-correlation-id-123"

    response = client.get(
        "/health",
        headers={"Origin": _ALLOWED_ORIGIN, CORRELATION_ID_HEADER: incoming_correlation_id},
    )

    assert response.headers[CORRELATION_ID_HEADER] == incoming_correlation_id
    assert response.headers["access-control-allow-origin"] == _ALLOWED_ORIGIN


def test_correlation_id_header_is_exposed_to_cross_origin_javascript() -> None:
    client = TestClient(_build_test_app(cors_allow_origins=_ALLOWED_ORIGIN))

    response = client.get("/health", headers={"Origin": _ALLOWED_ORIGIN})

    assert CORRELATION_ID_HEADER in response.headers["access-control-expose-headers"]


def test_wildcard_origin_is_safe_with_credentials_disabled() -> None:
    """If CORS_ALLOW_ORIGINS is ever set to "*", allow_credentials=False keeps the combination
    spec-safe: the response reflects a literal "*", never Access-Control-Allow-Credentials."""
    client = TestClient(_build_test_app(cors_allow_origins="*"))

    response = client.get("/health", headers={"Origin": _DISALLOWED_ORIGIN})

    assert response.headers["access-control-allow-origin"] == "*"
    assert "access-control-allow-credentials" not in response.headers


def test_empty_cors_allow_origins_permits_no_cross_origin_requests() -> None:
    """The secure default: CORS_ALLOW_ORIGINS unset must not accidentally allow anything."""
    client = TestClient(_build_test_app(cors_allow_origins=""))

    response = client.get("/health", headers={"Origin": _ALLOWED_ORIGIN})

    assert "access-control-allow-origin" not in response.headers
