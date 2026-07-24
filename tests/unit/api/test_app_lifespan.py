"""Tests that the real, fully-wired app's lifespan runs without probing external dependencies.

`TestClient(app)` used as a context manager triggers ASGI lifespan startup/shutdown; every other
route test in this suite uses a bare `TestClient(app)` (no dependency I/O required to import the
app), so this is the only test in the unit tier that actually exercises app/core/lifespan.py end
to end. No real Postgres/Qdrant/MinIO/Redis/Ollama is reachable in this tier — if startup ever
turned into a dependency-reachability gate (which it must not, per app/core/lifespan.py's
docstring), this test would fail or hang.
"""

from fastapi.testclient import TestClient

from app.main import app


def test_app_starts_and_stops_without_reaching_any_external_dependency() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200


def test_app_lifespan_can_run_more_than_once() -> None:
    """Guards against the engine.dispose() shutdown callback leaving the app unusable for reuse."""
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
