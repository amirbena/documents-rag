"""HTTP-boundary unit tests for the job-id-scoped operator activation endpoint
(Phase 2.8.7, subtask 4): POST /api/v1/reindex/jobs/{job_id}/activate.

Matches tests/unit/api/test_reindex_routes.py's exact style — monkeypatches the *service*
function the route module imports directly (`activate_reindexed_document`) rather than faking a
full AsyncSession. `activate_reindexed_document()` already has its own dedicated unit/integration
tests (test_reindex_activation.py / test_reindex_activation_postgres.py); this file only proves
the route layer's own job — delegation, outcome-to-status-code mapping, response-body shape, and
that no lifecycle mutation happens in the router itself.
"""

import uuid
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

import app.api.v1.routes.reindex as reindex_route_module
from app.db.session import get_db_session
from app.main import app
from app.models.reindex_job import ReindexJob, ReindexJobStatus
from app.services.indexing.reindex_activation import ReindexActivationOutcome, ReindexActivationResult

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def _install_fake_db_session() -> None:
    async def _fake_db_session():
        yield object()

    app.dependency_overrides[get_db_session] = _fake_db_session


def _job(**overrides: object) -> ReindexJob:
    fields: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        document_id=str(uuid.uuid4()),
        source_collection_name="documents__ollama__old__ev0__cv0__d768",
        target_collection_name="documents__ollama__model__ev1__cv1__d768",
        target_chunk_size=500,
        target_chunk_overlap=50,
        status=ReindexJobStatus.COMPLETED,
        error_message=None,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        completed_at=datetime(2026, 1, 1, tzinfo=UTC),
        activated_at=None,
    )
    fields.update(overrides)
    return ReindexJob(**fields)  # type: ignore[arg-type]


def _install_fake_activate(monkeypatch: pytest.MonkeyPatch, result: ReindexActivationResult, calls: list):
    async def _fake(session, job_id):
        calls.append(job_id)
        return result

    monkeypatch.setattr(reindex_route_module, "activate_reindexed_document", _fake)


# --- successful activation ----------------------------------------------------------------------


def test_valid_activation_request_returns_200(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    job = _job(activated_at=datetime.now(UTC))
    calls: list = []
    _install_fake_activate(
        monkeypatch,
        ReindexActivationResult(outcome=ReindexActivationOutcome.ACTIVATED, job=job, document=object()),
        calls,
    )

    response = client.post(f"/api/v1/reindex/jobs/{job.id}/activate")

    assert response.status_code == 200


def test_correct_job_id_is_forwarded_to_the_service(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    job = _job(activated_at=datetime.now(UTC))
    calls: list = []
    _install_fake_activate(
        monkeypatch,
        ReindexActivationResult(outcome=ReindexActivationOutcome.ACTIVATED, job=job, document=object()),
        calls,
    )

    client.post(f"/api/v1/reindex/jobs/{job.id}/activate")

    assert calls == [job.id]


def test_exactly_one_service_invocation(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    job = _job(activated_at=datetime.now(UTC))
    calls: list = []
    _install_fake_activate(
        monkeypatch,
        ReindexActivationResult(outcome=ReindexActivationOutcome.ACTIVATED, job=job, document=object()),
        calls,
    )

    client.post(f"/api/v1/reindex/jobs/{job.id}/activate")

    assert len(calls) == 1


def test_successful_response_maps_every_field(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    activated_timestamp = datetime(2026, 7, 19, 10, 0, 0, tzinfo=UTC)
    job = _job(activated_at=activated_timestamp)
    calls: list = []
    _install_fake_activate(
        monkeypatch,
        ReindexActivationResult(outcome=ReindexActivationOutcome.ACTIVATED, job=job, document=object()),
        calls,
    )

    response = client.post(f"/api/v1/reindex/jobs/{job.id}/activate")

    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == job.id
    assert body["document_id"] == job.document_id
    assert body["status"] == "completed"
    assert body["activated"] is True
    assert body["already_activated"] is False
    assert body["previous_collection_name"] == job.source_collection_name
    assert body["active_collection_name"] == job.target_collection_name
    assert "cleanup_job_id" not in body


def test_activated_at_is_serialized_as_iso_datetime(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    activated_timestamp = datetime(2026, 7, 19, 10, 0, 0, tzinfo=UTC)
    job = _job(activated_at=activated_timestamp)
    calls: list = []
    _install_fake_activate(
        monkeypatch,
        ReindexActivationResult(outcome=ReindexActivationOutcome.ACTIVATED, job=job, document=object()),
        calls,
    )

    response = client.post(f"/api/v1/reindex/jobs/{job.id}/activate")

    assert response.json()["activated_at"] == "2026-07-19T10:00:00Z"


# --- already activated (idempotent) ---------------------------------------------------------------


def test_already_activated_job_returns_200_with_already_activated_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_db_session()
    job = _job(activated_at=datetime(2026, 1, 2, tzinfo=UTC))
    calls: list = []
    _install_fake_activate(
        monkeypatch,
        ReindexActivationResult(
            outcome=ReindexActivationOutcome.ALREADY_ACTIVATED, job=job, document=None
        ),
        calls,
    )

    response = client.post(f"/api/v1/reindex/jobs/{job.id}/activate")

    assert response.status_code == 200
    body = response.json()
    assert body["activated"] is True
    assert body["already_activated"] is True


# --- ineligible / error outcomes ------------------------------------------------------------------


def test_job_not_found_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    calls: list = []
    _install_fake_activate(
        monkeypatch,
        ReindexActivationResult(outcome=ReindexActivationOutcome.JOB_NOT_FOUND, job=None, document=None),
        calls,
    )

    response = client.post(f"/api/v1/reindex/jobs/{uuid.uuid4()}/activate")

    assert response.status_code == 404


def test_queued_or_processing_or_failed_job_returns_409(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    job = _job(status=ReindexJobStatus.PENDING)
    calls: list = []
    _install_fake_activate(
        monkeypatch,
        ReindexActivationResult(outcome=ReindexActivationOutcome.NOT_READY, job=job, document=None),
        calls,
    )

    response = client.post(f"/api/v1/reindex/jobs/{job.id}/activate")

    assert response.status_code == 409


def test_source_changed_returns_409(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    job = _job(status=ReindexJobStatus.COMPLETED)
    calls: list = []
    _install_fake_activate(
        monkeypatch,
        ReindexActivationResult(outcome=ReindexActivationOutcome.SOURCE_CHANGED, job=job, document=object()),
        calls,
    )

    response = client.post(f"/api/v1/reindex/jobs/{job.id}/activate")

    assert response.status_code == 409


def test_blocked_by_deletion_returns_409(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    job = _job(status=ReindexJobStatus.COMPLETED)
    calls: list = []
    _install_fake_activate(
        monkeypatch,
        ReindexActivationResult(
            outcome=ReindexActivationOutcome.BLOCKED_BY_DELETION, job=job, document=object()
        ),
        calls,
    )

    response = client.post(f"/api/v1/reindex/jobs/{job.id}/activate")

    assert response.status_code == 409


def test_document_missing_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    job = _job(status=ReindexJobStatus.COMPLETED)
    calls: list = []
    _install_fake_activate(
        monkeypatch,
        ReindexActivationResult(outcome=ReindexActivationOutcome.DOCUMENT_MISSING, job=job, document=None),
        calls,
    )

    response = client.post(f"/api/v1/reindex/jobs/{job.id}/activate")

    assert response.status_code == 404


# --- unexpected failure propagation ---------------------------------------------------------------


def test_unexpected_service_exception_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()

    async def _fake_raises(session, job_id):
        raise RuntimeError("unexpected database failure")

    monkeypatch.setattr(reindex_route_module, "activate_reindexed_document", _fake_raises)

    with pytest.raises(RuntimeError):
        client.post(f"/api/v1/reindex/jobs/{uuid.uuid4()}/activate")


# --- router boundary -------------------------------------------------------------------------------


def test_router_performs_no_direct_database_mutation() -> None:
    """The route function's own source must never call session.add/commit/delete — every mutation
    belongs to activate_reindexed_document()."""
    import inspect

    source = inspect.getsource(reindex_route_module.activate_reindex_job_route)
    assert ".add(" not in source
    assert ".commit(" not in source
    assert ".delete(" not in source


def test_router_performs_no_direct_vector_store_call() -> None:
    """The activation route takes no vector_store dependency and calls no vector-store method —
    activation is metadata-only, exactly like the sibling document-scoped route."""
    import inspect

    source = inspect.getsource(reindex_route_module.activate_reindex_job_route)
    assert "vector_store" not in source
    assert ".upsert" not in source
    assert ".search_similar(" not in source


def test_router_does_not_duplicate_eligibility_or_outcome_mapping() -> None:
    """The route must reuse the same _ACTIVATION_OUTCOME_ERRORS mapping as the sibling
    document-scoped route, never define a second one."""
    import inspect

    source = inspect.getsource(reindex_route_module.activate_reindex_job_route)
    assert "_ACTIVATION_OUTCOME_ERRORS" in source
    assert "ReindexActivationOutcome.NOT_READY" not in source  # no ad-hoc outcome branching
