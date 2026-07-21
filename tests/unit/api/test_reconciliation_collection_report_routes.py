"""HTTP-boundary unit tests for the collection reconciliation report API (Phase 2.8.7, subtask 5):
GET /api/v1/reconciliation/collections/{collection_name}/report.

Matches tests/unit/api/test_reconciliation_routes.py's style: monkeypatches the *service*
function the route module imports directly (`build_collection_reconciliation_report`) rather than
faking a full AsyncSession/VectorStore — that service has its own dedicated unit tests
(test_collection_reconciliation_report_service.py); this file only proves the route layer's own
job: delegation, response-body mapping, and that no Qdrant/ORM count query or classification
recalculation happens directly in the router.
"""

import inspect
import uuid
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

import app.api.v1.routes.reconciliation as reconciliation_route_module
from app.db.session import get_db_session
from app.main import app
from app.models.index_collection import IndexCollectionStatus
from app.services.reconciliation.collection_reconciliation_report_service import (
    CollectionReconciliationReport,
    CollectionReportClassification,
    CollectionReportFinding,
    CollectionReportFindingCode,
    InvalidCollectionNameError,
)
from app.services.reconciliation.document_audit_service import FindingSeverity

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


def _finding(**overrides: object) -> CollectionReportFinding:
    fields: dict[str, object] = dict(
        code=CollectionReportFindingCode.VECTOR_COUNT_DEFICIT,
        severity=FindingSeverity.ERROR,
        summary="A finding.",
        expected_state="expected",
        actual_state="actual",
    )
    fields.update(overrides)
    return CollectionReportFinding(**fields)  # type: ignore[arg-type]


def _report(collection_name: str, **overrides: object) -> CollectionReconciliationReport:
    fields: dict[str, object] = dict(
        collection_name=collection_name,
        classification=CollectionReportClassification.HEALTHY,
        exists=True,
        is_active=True,
        index_collection_status=IndexCollectionStatus.ACTIVE,
        embedding_provider="ollama",
        embedding_model="model",
        embedding_dimension=768,
        embedding_version="v1",
        chunking_version="v1",
        document_count=10,
        expected_vector_count=10,
        actual_vector_count=10,
        difference=0,
        findings=(),
        generated_at=_BASE_TIME,
    )
    fields.update(overrides)
    return CollectionReconciliationReport(**fields)  # type: ignore[arg-type]


def _install_fake_report(
    monkeypatch: pytest.MonkeyPatch, report: CollectionReconciliationReport, calls: list
):
    async def _fake(session, collection_name, settings, vector_store):
        calls.append(collection_name)
        return report

    monkeypatch.setattr(reconciliation_route_module, "build_collection_reconciliation_report", _fake)


# --- delegation -------------------------------------------------------------------------------


def test_valid_collection_name_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    calls: list = []
    _install_fake_report(monkeypatch, _report("documents-v2"), calls)

    response = client.get("/api/v1/reconciliation/collections/documents-v2/report")

    assert response.status_code == 200
    assert calls == ["documents-v2"]


def test_exactly_one_report_service_call(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    calls: list = []
    _install_fake_report(monkeypatch, _report("documents-v2"), calls)

    client.get("/api/v1/reconciliation/collections/documents-v2/report")

    assert len(calls) == 1


# --- response mapping ---------------------------------------------------------------------------


def test_healthy_report_maps_every_field(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    calls: list = []
    _install_fake_report(monkeypatch, _report("documents-v2"), calls)

    response = client.get("/api/v1/reconciliation/collections/documents-v2/report")

    assert response.status_code == 200
    body = response.json()
    assert body["collection_name"] == "documents-v2"
    assert body["classification"] == "healthy"
    assert body["exists"] is True
    assert body["is_active"] is True
    assert body["index_collection_status"] == "active"
    assert body["embedding_provider"] == "ollama"
    assert body["document_count"] == 10
    assert body["expected_vector_count"] == 10
    assert body["actual_vector_count"] == 10
    assert body["difference"] == 0
    assert body["issues"] == []
    assert body["generated_at"] == "2026-01-01T00:00:00Z"


def test_mismatched_count_report_is_mapped(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    calls: list = []
    finding = _finding(code=CollectionReportFindingCode.VECTOR_COUNT_DEFICIT)
    report = _report(
        "documents-v2",
        classification=CollectionReportClassification.INCONSISTENT,
        document_count=10,
        expected_vector_count=10,
        actual_vector_count=4,
        difference=-6,
        findings=(finding,),
    )
    _install_fake_report(monkeypatch, report, calls)

    response = client.get("/api/v1/reconciliation/collections/documents-v2/report")

    body = response.json()
    assert body["classification"] == "inconsistent"
    assert body["expected_vector_count"] == 10
    assert body["actual_vector_count"] == 4
    assert body["difference"] == -6
    assert len(body["issues"]) == 1
    assert body["issues"][0]["code"] == "vector_count_deficit"


def test_missing_collection_report_is_mapped(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    calls: list = []
    finding = _finding(code=CollectionReportFindingCode.COLLECTION_MISSING)
    report = _report(
        "missing-collection",
        classification=CollectionReportClassification.MISSING,
        exists=False,
        is_active=False,
        index_collection_status=None,
        embedding_provider=None,
        embedding_model=None,
        embedding_dimension=None,
        embedding_version=None,
        chunking_version=None,
        actual_vector_count=0,
        findings=(finding,),
    )
    _install_fake_report(monkeypatch, report, calls)

    response = client.get("/api/v1/reconciliation/collections/missing-collection/report")

    assert response.status_code == 200
    body = response.json()
    assert body["classification"] == "missing"
    assert body["exists"] is False
    assert body["actual_vector_count"] == 0


def test_inactive_collection_report_is_mapped(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    calls: list = []
    report = _report(
        "documents-v1-old", is_active=False, index_collection_status=IndexCollectionStatus.RETIRED
    )
    _install_fake_report(monkeypatch, report, calls)

    response = client.get("/api/v1/reconciliation/collections/documents-v1-old/report")

    body = response.json()
    assert body["is_active"] is False
    assert body["index_collection_status"] == "retired"
    assert body["classification"] == "healthy"  # inactive != inconsistent


def test_unmanaged_collection_report_is_mapped(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    calls: list = []
    finding = _finding(
        code=CollectionReportFindingCode.COLLECTION_UNMANAGED, severity=FindingSeverity.WARNING
    )
    report = _report(
        "legacy-collection",
        classification=CollectionReportClassification.UNMANAGED,
        is_active=False,
        index_collection_status=None,
        embedding_provider=None,
        embedding_model=None,
        embedding_dimension=None,
        embedding_version=None,
        chunking_version=None,
        document_count=0,
        expected_vector_count=0,
        actual_vector_count=5,
        difference=5,
        findings=(finding,),
    )
    _install_fake_report(monkeypatch, report, calls)

    response = client.get("/api/v1/reconciliation/collections/legacy-collection/report")

    body = response.json()
    assert body["classification"] == "unmanaged"
    assert body["issues"][0]["code"] == "collection_unmanaged"


# --- invalid collection name -----------------------------------------------------------------------


def test_invalid_collection_name_returns_400(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()

    async def _fake_raises(session, collection_name, settings, vector_store):
        raise InvalidCollectionNameError("collection_name must be 1-255 characters...")

    monkeypatch.setattr(
        reconciliation_route_module, "build_collection_reconciliation_report", _fake_raises
    )

    response = client.get("/api/v1/reconciliation/collections/not..valid/report")

    assert response.status_code == 400
    assert "detail" in response.json()


def test_invalid_collection_name_error_never_leaks_the_raw_exception_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 2.10: detail must be a fixed constant, never str(exc)."""
    _install_fake_db_session()
    raw_message = "collection_name must be 1-255 characters of a-sensitive-internal-detail."

    async def _fake_raises(session, collection_name, settings, vector_store):
        raise InvalidCollectionNameError(raw_message)

    monkeypatch.setattr(
        reconciliation_route_module, "build_collection_reconciliation_report", _fake_raises
    )

    response = client.get("/api/v1/reconciliation/collections/not..valid/report")

    assert response.status_code == 400
    assert response.json()["detail"] != raw_message
    assert "a-sensitive-internal-detail" not in response.json()["detail"]


# --- unexpected failure propagation ---------------------------------------------------------------


def test_unexpected_service_exception_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()

    async def _fake_raises(session, collection_name, settings, vector_store):
        raise RuntimeError("unexpected qdrant failure")

    monkeypatch.setattr(
        reconciliation_route_module, "build_collection_reconciliation_report", _fake_raises
    )

    with pytest.raises(RuntimeError):
        client.get(f"/api/v1/reconciliation/collections/{uuid.uuid4().hex}/report")


# --- router boundary -------------------------------------------------------------------------------


def test_route_performs_no_direct_qdrant_count_call() -> None:
    source = inspect.getsource(reconciliation_route_module.collection_reconciliation_report_route)
    assert "count_collection_vectors(" not in source
    assert "get_collection_vector_size(" not in source


def test_route_performs_no_direct_orm_count_query() -> None:
    source = inspect.getsource(reconciliation_route_module.collection_reconciliation_report_route)
    assert "select(" not in source
    assert "func.count(" not in source


def test_route_does_not_recalculate_classification() -> None:
    source = inspect.getsource(reconciliation_route_module.collection_reconciliation_report_route)
    assert "CollectionReportClassification.HEALTHY" not in source
    assert "CollectionReportClassification.INCONSISTENT" not in source


def test_route_performs_no_mutation_calls() -> None:
    source = inspect.getsource(reconciliation_route_module.collection_reconciliation_report_route)
    assert ".add(" not in source
    assert ".commit(" not in source
    assert ".delete(" not in source
