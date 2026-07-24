"""Tests for CORS wiring (Phase 2.10, app/core/cors.py + its registration in app/main.py).

Builds a small standalone FastAPI app per test, wired with the exact same three-middleware
registration order as app/main.py (`UnhandledExceptionCorsBoundary` innermost, `CORSMiddleware`,
then `correlation_id_middleware` outermost), so these tests exercise the real policy and real
Starlette CORSMiddleware behavior without depending on whatever CORS_ALLOW_ORIGINS happens to be
set in the ambient test environment (the module-level `app.main.app` singleton is built once at
import time from real settings, so it can't be reconfigured per test).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.correlation import CORRELATION_ID_HEADER
from app.core.cors import cors_middleware_kwargs
from app.core.errors import NotFoundError
from app.core.exception_handlers import app_error_handler
from app.core.middleware import UnhandledExceptionCorsBoundary, correlation_id_middleware

_ALLOWED_ORIGIN = "http://localhost:3000"
_DISALLOWED_ORIGIN = "http://evil.example.com"


def _build_test_app(*, cors_allow_origins: str) -> FastAPI:
    settings = Settings(CORS_ALLOW_ORIGINS=cors_allow_origins)
    app = FastAPI()
    # Same registration order as app/main.py — see that module's comment for why the order matters:
    # boundary added first (innermost), CORS second, correlation last (outermost).
    app.add_middleware(UnhandledExceptionCorsBoundary)
    app.add_middleware(CORSMiddleware, **cors_middleware_kwargs(settings))
    app.middleware("http")(correlation_id_middleware)
    app.add_exception_handler(NotFoundError, app_error_handler)  # type: ignore[arg-type]

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.delete("/documents/{document_id}")
    async def delete_document(document_id: str) -> dict[str, str]:
        return {"status": "deleted"}

    @app.get("/boom")
    async def boom() -> dict[str, str]:
        raise RuntimeError("db password is hunter2")

    @app.get("/boom-app-error")
    async def boom_app_error() -> dict[str, str]:
        raise NotFoundError("some internal detail that must not leak")

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


def test_unhandled_exception_from_an_allowed_origin_still_receives_cors_headers() -> None:
    """A truly unhandled exception (not AppError, not HTTPException) must still get the same CORS
    headers as any other response for an allowed Origin — `UnhandledExceptionCorsBoundary` exists
    precisely so a bare-Exception 500 doesn't lose them (Starlette otherwise routes a bare-Exception
    handler to ServerErrorMiddleware, which sits outside CORSMiddleware)."""
    client = TestClient(_build_test_app(cors_allow_origins=_ALLOWED_ORIGIN), raise_server_exceptions=False)

    response = client.get("/boom", headers={"Origin": _ALLOWED_ORIGIN})

    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "Internal server error."
    assert "hunter2" not in body["detail"]
    assert "RuntimeError" not in body["detail"]
    assert response.headers["access-control-allow-origin"] == _ALLOWED_ORIGIN
    assert response.headers["vary"] == "Origin"
    assert CORRELATION_ID_HEADER in response.headers


def test_unhandled_exception_from_a_disallowed_origin_receives_no_allow_origin_header() -> None:
    """A disallowed Origin must never receive a permissive CORS header, even on a 500 caused by a
    genuinely unhandled exception."""
    client = TestClient(_build_test_app(cors_allow_origins=_ALLOWED_ORIGIN), raise_server_exceptions=False)

    response = client.get("/boom", headers={"Origin": _DISALLOWED_ORIGIN})

    assert response.status_code == 500
    assert "access-control-allow-origin" not in response.headers


def test_app_error_response_with_an_allowed_origin_is_unchanged_by_the_exception_boundary() -> None:
    """AppError handling (inside ExceptionMiddleware) is untouched by the new boundary — it already
    received correct CORS headers before this fix and must continue to."""
    client = TestClient(_build_test_app(cors_allow_origins=_ALLOWED_ORIGIN), raise_server_exceptions=False)

    response = client.get("/boom-app-error", headers={"Origin": _ALLOWED_ORIGIN})

    assert response.status_code == 404
    assert response.json()["detail"] == "some internal detail that must not leak"
    assert response.headers["access-control-allow-origin"] == _ALLOWED_ORIGIN
    assert CORRELATION_ID_HEADER in response.headers


def test_ordinary_successful_cors_response_is_unaffected_by_the_exception_boundary() -> None:
    """A normal 200 response from an allowed origin is unaffected by the new innermost middleware
    layer — same headers as before this fix."""
    client = TestClient(_build_test_app(cors_allow_origins=_ALLOWED_ORIGIN))

    response = client.get("/health", headers={"Origin": _ALLOWED_ORIGIN})

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["access-control-allow-origin"] == _ALLOWED_ORIGIN
    assert CORRELATION_ID_HEADER in response.headers
