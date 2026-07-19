"""HTTP-boundary unit tests for the single-document re-index API (Phase 2.8.6, subtask 6).

Matches tests/unit/api/test_ingestion_retry_routes.py's dependency-override style, but — unlike
that file — monkeypatches the *service* functions the route module imports directly, rather than
faking a full AsyncSession. `inspect_document_reindex_state`/`schedule_reindex`/
`activate_reindexed_document` each have their own already-covered decision-table unit/PostgreSQL
tests (test_reindex_inspection_service.py, test_reindex_scheduling_service.py,
test_reindex_activation.py); this file only proves the route layer's own job — delegation,
outcome-to-status-code mapping, response-body shape, and sanitization — never re-deriving any of
those services' own business rules.
"""

import uuid
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

import app.api.v1.routes.reindex as reindex_route_module
from app.db.session import get_db_session
from app.main import app
from app.models.reindex_job import ReindexJob, ReindexJobStatus
from app.schemas.reindex import ReindexLifecycleState
from app.services.indexing.reindex_activation import ReindexActivationOutcome, ReindexActivationResult
from app.services.indexing.reindex_inspection_service import DocumentReindexState, IndexConfigSnapshot
from app.services.indexing.reindex_scheduling_service import ReindexSchedulingOutcome, ReindexSchedulingResult

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def _install_fake_db_session() -> None:
    """FastAPI still needs a get_db_session dependency to resolve — every test in this file
    monkeypatches the service functions the route calls, so the actual session value is never
    used for a real query; a bare sentinel object is sufficient."""

    async def _fake_db_session():
        yield object()

    app.dependency_overrides[get_db_session] = _fake_db_session


def _snapshot(**overrides: object) -> IndexConfigSnapshot:
    fields: dict[str, object] = dict(
        collection_name="documents__ollama__model__ev1__cv1__d768",
        provider="ollama",
        model="model",
        dimension=768,
        embedding_version="v1",
        chunking_version="v1",
        chunk_size=None,
        chunk_overlap=None,
    )
    fields.update(overrides)
    return IndexConfigSnapshot(**fields)  # type: ignore[arg-type]


def _job(document_id: str, **overrides: object) -> ReindexJob:
    fields: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        document_id=document_id,
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


# --- inspection (GET) -------------------------------------------------------------------------


def test_inspection_returns_404_for_missing_document(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()

    async def _fake_inspect(session, document_id, settings):
        return None

    monkeypatch.setattr(reindex_route_module, "inspect_document_reindex_state", _fake_inspect)

    response = client.get(f"/api/v1/documents/{uuid.uuid4()}/reindex")

    assert response.status_code == 404


def test_inspection_reports_up_to_date_document(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    document_id = str(uuid.uuid4())

    async def _fake_inspect(session, doc_id, settings):
        return DocumentReindexState(
            document_id=doc_id,
            state=ReindexLifecycleState.UP_TO_DATE,
            is_stale=False,
            active_index=_snapshot(),
            desired_index=_snapshot(),
            latest_job=None,
            can_schedule=False,
            can_activate=False,
        )

    monkeypatch.setattr(reindex_route_module, "inspect_document_reindex_state", _fake_inspect)

    response = client.get(f"/api/v1/documents/{document_id}/reindex")

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "up_to_date"
    assert body["is_stale"] is False
    assert body["latest_attempt"] is None
    assert body["can_schedule"] is False


def test_inspection_reports_stale_document(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    document_id = str(uuid.uuid4())

    async def _fake_inspect(session, doc_id, settings):
        return DocumentReindexState(
            document_id=doc_id,
            state=ReindexLifecycleState.STALE,
            is_stale=True,
            active_index=_snapshot(embedding_version="v0"),
            desired_index=_snapshot(embedding_version="v1"),
            latest_job=None,
            can_schedule=True,
            can_activate=False,
        )

    monkeypatch.setattr(reindex_route_module, "inspect_document_reindex_state", _fake_inspect)

    response = client.get(f"/api/v1/documents/{document_id}/reindex")

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "stale"
    assert body["is_stale"] is True
    assert body["can_schedule"] is True
    assert body["active_index"]["embedding_version"] == "v0"
    assert body["desired_index"]["embedding_version"] == "v1"


def test_inspection_reports_document_without_active_index(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    document_id = str(uuid.uuid4())

    async def _fake_inspect(session, doc_id, settings):
        return DocumentReindexState(
            document_id=doc_id,
            state=ReindexLifecycleState.NOT_INDEXED,
            is_stale=True,
            active_index=_snapshot(
                collection_name=None, provider=None, model=None, dimension=None,
                embedding_version=None, chunking_version=None,
            ),
            desired_index=_snapshot(),
            latest_job=None,
            can_schedule=False,
            can_activate=False,
        )

    monkeypatch.setattr(reindex_route_module, "inspect_document_reindex_state", _fake_inspect)

    response = client.get(f"/api/v1/documents/{document_id}/reindex")

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "not_indexed"
    assert body["active_index"]["collection_name"] is None
    assert body["can_schedule"] is False


def test_inspection_includes_the_latest_reindex_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    document_id = str(uuid.uuid4())
    job = _job(document_id, status=ReindexJobStatus.PROCESSING)

    async def _fake_inspect(session, doc_id, settings):
        return DocumentReindexState(
            document_id=doc_id,
            state=ReindexLifecycleState.REINDEX_PROCESSING,
            is_stale=True,
            active_index=_snapshot(),
            desired_index=_snapshot(),
            latest_job=job,
            can_schedule=False,
            can_activate=False,
        )

    monkeypatch.setattr(reindex_route_module, "inspect_document_reindex_state", _fake_inspect)

    response = client.get(f"/api/v1/documents/{document_id}/reindex")

    assert response.status_code == 200
    attempt = response.json()["latest_attempt"]
    assert attempt is not None
    assert attempt["job_id"] == job.id
    assert attempt["status"] == "processing"
    assert attempt["source_collection_name"] == job.source_collection_name
    assert attempt["target_collection_name"] == job.target_collection_name


def test_inspection_distinguishes_build_completion_from_activation(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    document_id = str(uuid.uuid4())
    built_job = _job(document_id, status=ReindexJobStatus.COMPLETED, activated_at=None)

    async def _fake_inspect_built(session, doc_id, settings):
        return DocumentReindexState(
            document_id=doc_id,
            state=ReindexLifecycleState.TARGET_BUILT,
            is_stale=True,
            active_index=_snapshot(),
            desired_index=_snapshot(),
            latest_job=built_job,
            can_schedule=False,
            can_activate=True,
        )

    monkeypatch.setattr(reindex_route_module, "inspect_document_reindex_state", _fake_inspect_built)
    built_response = client.get(f"/api/v1/documents/{document_id}/reindex")

    assert built_response.status_code == 200
    built_body = built_response.json()
    assert built_body["state"] == "target_built"
    assert built_body["latest_attempt"]["status"] == "completed"
    assert built_body["latest_attempt"]["activated_at"] is None
    assert built_body["can_activate"] is True

    activated_timestamp = datetime(2026, 1, 2, tzinfo=UTC)
    activated_job = _job(document_id, status=ReindexJobStatus.COMPLETED, activated_at=activated_timestamp)

    async def _fake_inspect_activated(session, doc_id, settings):
        return DocumentReindexState(
            document_id=doc_id,
            state=ReindexLifecycleState.ACTIVATED,
            is_stale=False,
            active_index=_snapshot(),
            desired_index=_snapshot(),
            latest_job=activated_job,
            can_schedule=False,
            can_activate=False,
        )

    monkeypatch.setattr(reindex_route_module, "inspect_document_reindex_state", _fake_inspect_activated)
    activated_response = client.get(f"/api/v1/documents/{document_id}/reindex")

    assert activated_response.status_code == 200
    activated_body = activated_response.json()
    assert activated_body["state"] == "activated"
    assert activated_body["latest_attempt"]["status"] == "completed"
    assert activated_body["latest_attempt"]["activated_at"] is not None


def test_inspection_sanitizes_internal_error_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    """API responses must sanitize internal errors — the raw ReindexJob.error_message never leaks."""
    _install_fake_db_session()
    document_id = str(uuid.uuid4())
    raw_secret = "connection refused at internal-host:6333 while writing vector"
    failed_job = _job(document_id, status=ReindexJobStatus.FAILED, error_message=raw_secret)

    async def _fake_inspect(session, doc_id, settings):
        return DocumentReindexState(
            document_id=doc_id,
            state=ReindexLifecycleState.FAILED,
            is_stale=True,
            active_index=_snapshot(),
            desired_index=_snapshot(),
            latest_job=failed_job,
            can_schedule=False,
            can_activate=False,
        )

    monkeypatch.setattr(reindex_route_module, "inspect_document_reindex_state", _fake_inspect)

    response = client.get(f"/api/v1/documents/{document_id}/reindex")

    body_text = response.text
    assert raw_secret not in body_text
    assert response.json()["latest_attempt"]["safe_error_message"] is not None


# --- schedule (POST) --------------------------------------------------------------------------


def test_schedule_delegates_once_to_the_scheduling_service(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    document_id = str(uuid.uuid4())
    calls: list[object] = []

    async def _fake_get_document(session, doc_id):
        return object()  # a truthy stand-in Document

    async def _fake_schedule_reindex(session, document, vector_store, target_config, **kwargs):
        calls.append((document, kwargs))
        return ReindexSchedulingResult(
            outcome=ReindexSchedulingOutcome.CREATED,
            document=document,  # type: ignore[arg-type]
            job=_job(document_id, status=ReindexJobStatus.PENDING),
            target_config=target_config,
        )

    monkeypatch.setattr(reindex_route_module, "get_document", _fake_get_document)
    monkeypatch.setattr(reindex_route_module, "schedule_reindex", _fake_schedule_reindex)
    monkeypatch.setattr(reindex_route_module, "get_vector_store", lambda: object())

    response = client.post(f"/api/v1/documents/{document_id}/reindex")

    assert response.status_code == 202
    assert len(calls) == 1


def test_schedule_returns_existing_active_attempt_idempotently(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    document_id = str(uuid.uuid4())
    existing_job = _job(document_id, status=ReindexJobStatus.PENDING)

    async def _fake_get_document(session, doc_id):
        return object()

    async def _fake_schedule_reindex(session, document, vector_store, target_config, **kwargs):
        return ReindexSchedulingResult(
            outcome=ReindexSchedulingOutcome.ALREADY_ACTIVE,
            document=document,  # type: ignore[arg-type]
            job=existing_job,
            target_config=target_config,
        )

    monkeypatch.setattr(reindex_route_module, "get_document", _fake_get_document)
    monkeypatch.setattr(reindex_route_module, "schedule_reindex", _fake_schedule_reindex)

    response = client.post(f"/api/v1/documents/{document_id}/reindex")

    assert response.status_code == 200
    body = response.json()
    assert body["created"] is False
    assert body["job_id"] == existing_job.id


def test_schedule_does_not_build_inline() -> None:
    """The route module must never import a build-execution symbol at all."""
    assert not hasattr(reindex_route_module, "build_reindex_target")
    assert not hasattr(reindex_route_module, "ReindexWorker")


def test_schedule_does_not_activate_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    document_id = str(uuid.uuid4())

    async def _fake_get_document(session, doc_id):
        return object()

    async def _fake_schedule_reindex(session, document, vector_store, target_config, **kwargs):
        return ReindexSchedulingResult(
            outcome=ReindexSchedulingOutcome.CREATED,
            document=document,  # type: ignore[arg-type]
            job=_job(document_id, status=ReindexJobStatus.PENDING),
            target_config=target_config,
        )

    async def _fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("schedule must never call activate_reindexed_document")

    monkeypatch.setattr(reindex_route_module, "get_document", _fake_get_document)
    monkeypatch.setattr(reindex_route_module, "schedule_reindex", _fake_schedule_reindex)
    monkeypatch.setattr(reindex_route_module, "activate_reindexed_document", _fail_if_called)

    response = client.post(f"/api/v1/documents/{document_id}/reindex")

    assert response.status_code == 202


def test_schedule_maps_deletion_conflict_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    document_id = str(uuid.uuid4())

    async def _fake_get_document(session, doc_id):
        return object()

    async def _fake_schedule_reindex(session, document, vector_store, target_config, **kwargs):
        return ReindexSchedulingResult(
            outcome=ReindexSchedulingOutcome.DELETION_ACTIVE,
            document=document,  # type: ignore[arg-type]
            job=None,
            target_config=target_config,
        )

    monkeypatch.setattr(reindex_route_module, "get_document", _fake_get_document)
    monkeypatch.setattr(reindex_route_module, "schedule_reindex", _fake_schedule_reindex)

    response = client.post(f"/api/v1/documents/{document_id}/reindex")

    assert response.status_code == 409


def test_schedule_maps_non_stale_behavior_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    document_id = str(uuid.uuid4())

    async def _fake_get_document(session, doc_id):
        return object()

    async def _fake_schedule_reindex(session, document, vector_store, target_config, **kwargs):
        return ReindexSchedulingResult(
            outcome=ReindexSchedulingOutcome.ALREADY_CURRENT,
            document=document,  # type: ignore[arg-type]
            job=None,
            target_config=target_config,
        )

    monkeypatch.setattr(reindex_route_module, "get_document", _fake_get_document)
    monkeypatch.setattr(reindex_route_module, "schedule_reindex", _fake_schedule_reindex)

    response = client.post(f"/api/v1/documents/{document_id}/reindex")

    assert response.status_code == 409


def test_schedule_returns_404_for_missing_document(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()

    async def _fake_get_document(session, doc_id):
        return None

    monkeypatch.setattr(reindex_route_module, "get_document", _fake_get_document)

    response = client.post(f"/api/v1/documents/{uuid.uuid4()}/reindex")

    assert response.status_code == 404


# --- activate (POST) --------------------------------------------------------------------------


def test_activation_delegates_once_to_the_activation_service(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    document_id = str(uuid.uuid4())
    job = _job(document_id, status=ReindexJobStatus.COMPLETED, activated_at=datetime.now(UTC))
    calls: list[str] = []

    async def _fake_get_latest_reindex_job(session, doc_id):
        return job

    async def _fake_activate(session, job_id):
        calls.append(job_id)
        return ReindexActivationResult(outcome=ReindexActivationOutcome.ACTIVATED, job=job, document=object())  # type: ignore[arg-type]

    monkeypatch.setattr(reindex_route_module, "get_latest_reindex_job", _fake_get_latest_reindex_job)
    monkeypatch.setattr(reindex_route_module, "activate_reindexed_document", _fake_activate)

    response = client.post(f"/api/v1/documents/{document_id}/reindex/activate")

    assert response.status_code == 200
    assert calls == [job.id]


def test_activation_maps_successful_activation_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    document_id = str(uuid.uuid4())
    activated_timestamp = datetime.now(UTC)
    job = _job(document_id, status=ReindexJobStatus.COMPLETED, activated_at=activated_timestamp)

    async def _fake_get_latest_reindex_job(session, doc_id):
        return job

    async def _fake_activate(session, job_id):
        return ReindexActivationResult(outcome=ReindexActivationOutcome.ACTIVATED, job=job, document=object())  # type: ignore[arg-type]

    monkeypatch.setattr(reindex_route_module, "get_latest_reindex_job", _fake_get_latest_reindex_job)
    monkeypatch.setattr(reindex_route_module, "activate_reindexed_document", _fake_activate)

    response = client.post(f"/api/v1/documents/{document_id}/reindex/activate")

    assert response.status_code == 200
    body = response.json()
    assert body["already_activated"] is False
    assert body["job_id"] == job.id


def test_activation_maps_already_activated_behavior_idempotently(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    document_id = str(uuid.uuid4())
    job = _job(document_id, status=ReindexJobStatus.COMPLETED, activated_at=datetime.now(UTC))

    async def _fake_get_latest_reindex_job(session, doc_id):
        return job

    async def _fake_activate(session, job_id):
        return ReindexActivationResult(
            outcome=ReindexActivationOutcome.ALREADY_ACTIVATED, job=job, document=None
        )

    monkeypatch.setattr(reindex_route_module, "get_latest_reindex_job", _fake_get_latest_reindex_job)
    monkeypatch.setattr(reindex_route_module, "activate_reindexed_document", _fake_activate)

    response = client.post(f"/api/v1/documents/{document_id}/reindex/activate")

    assert response.status_code == 200
    assert response.json()["already_activated"] is True


def test_activation_maps_not_ready_behavior_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    document_id = str(uuid.uuid4())
    job = _job(document_id, status=ReindexJobStatus.PENDING)

    async def _fake_get_latest_reindex_job(session, doc_id):
        return job

    async def _fake_activate(session, job_id):
        return ReindexActivationResult(outcome=ReindexActivationOutcome.NOT_READY, job=job, document=None)

    monkeypatch.setattr(reindex_route_module, "get_latest_reindex_job", _fake_get_latest_reindex_job)
    monkeypatch.setattr(reindex_route_module, "activate_reindexed_document", _fake_activate)

    response = client.post(f"/api/v1/documents/{document_id}/reindex/activate")

    assert response.status_code == 409


def test_activation_maps_source_changed_behavior_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    document_id = str(uuid.uuid4())
    job = _job(document_id, status=ReindexJobStatus.COMPLETED)

    async def _fake_get_latest_reindex_job(session, doc_id):
        return job

    async def _fake_activate(session, job_id):
        return ReindexActivationResult(
            outcome=ReindexActivationOutcome.SOURCE_CHANGED, job=job, document=object()  # type: ignore[arg-type]
        )

    monkeypatch.setattr(reindex_route_module, "get_latest_reindex_job", _fake_get_latest_reindex_job)
    monkeypatch.setattr(reindex_route_module, "activate_reindexed_document", _fake_activate)

    response = client.post(f"/api/v1/documents/{document_id}/reindex/activate")

    assert response.status_code == 409


def test_activation_maps_deletion_conflict_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    document_id = str(uuid.uuid4())
    job = _job(document_id, status=ReindexJobStatus.COMPLETED)

    async def _fake_get_latest_reindex_job(session, doc_id):
        return job

    async def _fake_activate(session, job_id):
        return ReindexActivationResult(
            outcome=ReindexActivationOutcome.BLOCKED_BY_DELETION, job=job, document=object()  # type: ignore[arg-type]
        )

    monkeypatch.setattr(reindex_route_module, "get_latest_reindex_job", _fake_get_latest_reindex_job)
    monkeypatch.setattr(reindex_route_module, "activate_reindexed_document", _fake_activate)

    response = client.post(f"/api/v1/documents/{document_id}/reindex/activate")

    assert response.status_code == 409


def test_activation_returns_404_when_no_reindex_attempt_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()

    async def _fake_get_latest_reindex_job(session, doc_id):
        return None

    monkeypatch.setattr(reindex_route_module, "get_latest_reindex_job", _fake_get_latest_reindex_job)

    response = client.post(f"/api/v1/documents/{uuid.uuid4()}/reindex/activate")

    assert response.status_code == 404


def test_activation_returns_404_for_job_id_belonging_to_another_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller-supplied job_id for a different document must never be silently activated."""
    _install_fake_db_session()
    document_id = str(uuid.uuid4())
    other_document_job = _job(str(uuid.uuid4()), status=ReindexJobStatus.COMPLETED)

    async def _fake_get_reindex_job(session, job_id):
        return other_document_job

    async def _fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("activation must never run for a job belonging to another document")

    monkeypatch.setattr(reindex_route_module, "get_reindex_job", _fake_get_reindex_job)
    monkeypatch.setattr(reindex_route_module, "activate_reindexed_document", _fail_if_called)

    response = client.post(
        f"/api/v1/documents/{document_id}/reindex/activate", params={"job_id": other_document_job.id}
    )

    assert response.status_code == 404


# --- controller boundary -----------------------------------------------------------------------


def test_controllers_perform_no_direct_qdrant_or_storage_operations() -> None:
    """Mirrors test_reindex_scheduling_service.py's own precedent for this exact concern."""
    import inspect

    source = inspect.getsource(reindex_route_module)
    assert "FileStorage" not in source
    assert ".upsert" not in source
    assert ".search_similar(" not in source
    assert "create_file_storage" not in source
