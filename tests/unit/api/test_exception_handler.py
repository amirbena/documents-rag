"""Tests for the centralized fallback exception handlers (Phase 2.10, app/core/exception_handlers.py).

Both handlers are a fallback net only — every route's own outcome-table/try-except mapping is
checked first by FastAPI, and remains unchanged. These tests prove the fallback itself: an
`AppError` maps to its declared status code with a fixed `{"detail": ...}` body, and any other
unhandled exception maps to a fixed 500 — neither ever leaks `str(exc)` or a stack trace, and
existing HTTPException-based routes are completely unaffected.
"""

import uuid

import pytest
from fastapi.testclient import TestClient

import app.api.v1.routes.reconciliation as reconciliation_route_module
from app.core.correlation import CORRELATION_ID_HEADER
from app.core.errors import ConflictError, NotFoundError
from app.db.session import get_db_session
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def _install_fake_db_session() -> None:
    async def _fake_db_session():
        yield object()

    app.dependency_overrides[get_db_session] = _fake_db_session


def _raise(monkeypatch: pytest.MonkeyPatch, exc: Exception) -> None:
    async def _fake(session, document_id, settings, file_storage, vector_store):
        raise exc

    monkeypatch.setattr(reconciliation_route_module, "audit_document_lifecycle", _fake)


def test_app_error_subclass_maps_to_its_declared_status_code(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    _raise(monkeypatch, NotFoundError("some internal detail that must not leak"))

    response = client.get(f"/api/v1/reconciliation/documents/{uuid.uuid4()}/audit")

    assert response.status_code == 404
    assert response.json()["detail"] == "some internal detail that must not leak"


def test_app_error_conflict_maps_to_409(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    _raise(monkeypatch, ConflictError("conflict detail"))

    response = client.get(f"/api/v1/reconciliation/documents/{uuid.uuid4()}/audit")

    assert response.status_code == 409


def test_app_error_response_includes_correlation_id_header(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    _raise(monkeypatch, NotFoundError("x"))

    response = client.get(f"/api/v1/reconciliation/documents/{uuid.uuid4()}/audit")

    assert CORRELATION_ID_HEADER in response.headers


def test_generic_unhandled_exception_maps_to_fixed_500_never_leaking_str_exc() -> None:
    """A raw, unexpected exception (not an AppError, not an HTTPException) must still map to a
    fixed 500 body — never the raw exception text — via the generic fallback handler."""
    _install_fake_db_session()

    async def _fake(session, document_id, settings, file_storage, vector_store):
        raise RuntimeError("qdrant unreachable at internal-host:6333, credential=super-secret")

    import app.api.v1.routes.reconciliation as route_module

    original = route_module.audit_document_lifecycle
    route_module.audit_document_lifecycle = _fake
    try:
        with TestClient(app, raise_server_exceptions=False) as raising_client:
            response = raising_client.get(f"/api/v1/reconciliation/documents/{uuid.uuid4()}/audit")
    finally:
        route_module.audit_document_lifecycle = original

    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "Internal server error."
    assert "internal-host" not in body["detail"]
    assert "super-secret" not in body["detail"]
    assert "RuntimeError" not in body["detail"]


class _FakeSessionWithNoDocuments:
    async def get(self, model: object, key: object) -> None:
        return None


def test_existing_http_exception_routes_are_unaffected_by_the_fallback_handlers() -> None:
    """A regular HTTPException-based 404 (no AppError, no generic-Exception path involved) must
    still behave exactly as before — the fallback handlers must never intercept it."""

    async def _fake_db_session():
        yield _FakeSessionWithNoDocuments()

    app.dependency_overrides[get_db_session] = _fake_db_session

    response = client.get(f"/api/v1/documents/{uuid.uuid4()}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."
