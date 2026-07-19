"""HTTP-boundary unit tests for the single-document lifecycle audit API (Phase 2.8.7, subtask 5):
GET /api/v1/reconciliation/documents/{document_id}/audit.

Matches tests/unit/api/test_reconciliation_routes.py's style: monkeypatches the *service*
function the route module imports directly (`audit_document_lifecycle`) rather than faking a full
AsyncSession — that service already has its own dedicated unit/integration tests
(test_document_audit_service.py / test_document_audit_postgres.py / test_document_audit_*_real.py);
this file only proves the route layer's own job: delegation, response-body mapping, and that no
batch/database/storage/vector-store call happens directly in the router.
"""

import inspect
import uuid
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

import app.api.v1.routes.reconciliation as reconciliation_route_module
from app.db.session import get_db_session
from app.main import app
from app.services.reconciliation.document_audit_service import (
    AuditOverallStatus,
    DocumentLifecycleAuditResult,
    DocumentLifecycleFinding,
    DocumentLifecycleFindingCode,
    FindingSeverity,
    PostgresLifecycleState,
    StorageLifecycleState,
    VectorLifecycleState,
)

client = TestClient(app)

_BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def _install_fake_db_session() -> None:
    async def _fake_db_session():
        yield object()

    app.dependency_overrides[get_db_session] = _fake_db_session


def _finding(**overrides: object) -> DocumentLifecycleFinding:
    fields: dict[str, object] = dict(
        code=DocumentLifecycleFindingCode.STALE_INGESTION_JOB,
        severity=FindingSeverity.WARNING,
        summary="A finding.",
        expected_state="expected",
        actual_state="actual",
        suggested_action="none",
        destructive_risk=False,
    )
    fields.update(overrides)
    return DocumentLifecycleFinding(**fields)  # type: ignore[arg-type]


def _postgres_state(**overrides: object) -> PostgresLifecycleState:
    fields: dict[str, object] = dict(
        collection_name="documents__ollama__model__ev1__cv1__d768",
        document_created_at=_BASE_TIME,
        latest_ingestion_status=None,
        latest_deletion_status=None,
        latest_reindex_status=None,
        latest_reindex_activated=False,
        pending_cleanup_collections=(),
    )
    fields.update(overrides)
    return PostgresLifecycleState(**fields)  # type: ignore[arg-type]


def _result(document_id: str, **overrides: object) -> DocumentLifecycleAuditResult:
    fields: dict[str, object] = dict(
        document_id=document_id,
        overall_status=AuditOverallStatus.CONSISTENT,
        findings=(),
        postgres_state=_postgres_state(),
        storage_state=StorageLifecycleState(inspected=True, exists=True),
        vector_state=VectorLifecycleState(
            inspected=True, collection_exists=True, has_vectors=True, vector_count=3
        ),
    )
    fields.update(overrides)
    return DocumentLifecycleAuditResult(**fields)  # type: ignore[arg-type]


def _install_fake_audit(monkeypatch: pytest.MonkeyPatch, result: DocumentLifecycleAuditResult, calls: list):
    async def _fake(session, document_id, settings, file_storage, vector_store):
        calls.append(document_id)
        return result

    monkeypatch.setattr(reconciliation_route_module, "audit_document_lifecycle", _fake)


# --- delegation -------------------------------------------------------------------------------


def test_valid_document_id_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    document_id = str(uuid.uuid4())
    calls: list = []
    _install_fake_audit(monkeypatch, _result(document_id), calls)

    response = client.get(f"/api/v1/reconciliation/documents/{document_id}/audit")

    assert response.status_code == 200
    assert calls == [document_id]


def test_exactly_one_call_to_audit_document_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    document_id = str(uuid.uuid4())
    calls: list = []
    _install_fake_audit(monkeypatch, _result(document_id), calls)

    client.get(f"/api/v1/reconciliation/documents/{document_id}/audit")

    assert len(calls) == 1


# --- response mapping ---------------------------------------------------------------------------


def test_successful_response_maps_every_top_level_field(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    document_id = str(uuid.uuid4())
    calls: list = []
    _install_fake_audit(monkeypatch, _result(document_id), calls)

    response = client.get(f"/api/v1/reconciliation/documents/{document_id}/audit")

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == document_id
    assert body["overall_status"] == "consistent"
    assert body["classification"] == "consistent"
    assert body["issues"] == []
    assert body["database"]["document_exists"] is True
    assert body["file_storage"]["source_file_exists"] is True
    assert body["vector_store"]["has_vectors"] is True
    assert body["vector_store"]["vector_count"] == 3


def test_findings_are_mapped_into_issues(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    document_id = str(uuid.uuid4())
    finding = _finding(
        code=DocumentLifecycleFindingCode.STALE_INGESTION_JOB, severity=FindingSeverity.WARNING
    )
    calls: list = []
    _install_fake_audit(monkeypatch, _result(document_id, findings=(finding,)), calls)

    response = client.get(f"/api/v1/reconciliation/documents/{document_id}/audit")

    issues = response.json()["issues"]
    assert len(issues) == 1
    assert issues[0]["code"] == "stale_ingestion_job"
    assert issues[0]["severity"] == "warning"


def test_classification_reflects_warning_findings(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    document_id = str(uuid.uuid4())
    finding = _finding(severity=FindingSeverity.WARNING)
    calls: list = []
    _install_fake_audit(monkeypatch, _result(document_id, findings=(finding,)), calls)

    response = client.get(f"/api/v1/reconciliation/documents/{document_id}/audit")

    assert response.json()["classification"] == "warning"


def test_classification_reflects_transitional_info_findings(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    document_id = str(uuid.uuid4())
    finding = _finding(
        code=DocumentLifecycleFindingCode.INGESTION_IN_PROGRESS, severity=FindingSeverity.INFO
    )
    calls: list = []
    _install_fake_audit(monkeypatch, _result(document_id, findings=(finding,)), calls)

    response = client.get(f"/api/v1/reconciliation/documents/{document_id}/audit")

    assert response.json()["classification"] == "transitional"


def test_created_at_is_serialized(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    document_id = str(uuid.uuid4())
    calls: list = []
    _install_fake_audit(
        monkeypatch,
        _result(document_id, postgres_state=_postgres_state(document_created_at=_BASE_TIME)),
        calls,
    )

    response = client.get(f"/api/v1/reconciliation/documents/{document_id}/audit")

    assert response.json()["database"]["document_created_at"] == "2026-01-01T00:00:00Z"


# --- missing document ----------------------------------------------------------------------------


def test_missing_document_returns_200_with_not_found_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    """The service returns a typed NOT_FOUND result, never an exception — the route must preserve
    that as data, never invent a 404."""
    _install_fake_db_session()
    document_id = str(uuid.uuid4())
    missing_finding = _finding(
        code=DocumentLifecycleFindingCode.DOCUMENT_MISSING, severity=FindingSeverity.ERROR
    )
    calls: list = []
    _install_fake_audit(
        monkeypatch,
        _result(
            document_id,
            overall_status=AuditOverallStatus.NOT_FOUND,
            findings=(missing_finding,),
            postgres_state=None,
            storage_state=None,
            vector_state=None,
        ),
        calls,
    )

    response = client.get(f"/api/v1/reconciliation/documents/{document_id}/audit")

    assert response.status_code == 200
    body = response.json()
    assert body["classification"] == "not_found"
    assert body["database"]["document_exists"] is False
    assert body["file_storage"] is None
    assert body["vector_store"] is None


# --- unexpected failure propagation ---------------------------------------------------------------


def test_unexpected_service_exception_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()

    async def _fake_raises(session, document_id, settings, file_storage, vector_store):
        raise RuntimeError("unexpected database failure")

    monkeypatch.setattr(reconciliation_route_module, "audit_document_lifecycle", _fake_raises)

    with pytest.raises(RuntimeError):
        client.get(f"/api/v1/reconciliation/documents/{uuid.uuid4()}/audit")


# --- router boundary -------------------------------------------------------------------------------


def test_route_never_calls_the_batch_auditor() -> None:
    source = inspect.getsource(reconciliation_route_module.audit_single_document_route)
    assert "audit_document_lifecycle_batch" not in source


def test_route_performs_no_direct_database_query() -> None:
    source = inspect.getsource(reconciliation_route_module.audit_single_document_route)
    assert "select(" not in source
    assert "session.get(" not in source
    assert "db.get(" not in source
    assert "db.execute(" not in source


def test_route_performs_no_direct_file_storage_call() -> None:
    source = inspect.getsource(reconciliation_route_module.audit_single_document_route)
    assert "file_storage.exists(" not in source


def test_route_performs_no_direct_vector_store_call() -> None:
    source = inspect.getsource(reconciliation_route_module.audit_single_document_route)
    assert "vector_store.get_collection_vector_size(" not in source
    assert "vector_store.count_document_vectors(" not in source


def test_route_performs_no_mutation_calls() -> None:
    source = inspect.getsource(reconciliation_route_module.audit_single_document_route)
    assert ".add(" not in source
    assert ".commit(" not in source
    assert ".delete(" not in source
