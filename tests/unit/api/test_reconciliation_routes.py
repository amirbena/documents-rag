"""HTTP-boundary unit tests for the batch document lifecycle audit API (Phase 2.8.7, subtask 3).

Matches tests/unit/api/test_reindex_routes.py's style: monkeypatches the *service* function the
route module imports directly (`audit_document_lifecycle_batch`) rather than faking a full
AsyncSession — `audit_document_lifecycle_batch()` already has its own dedicated unit/integration
tests (test_document_audit_batch_service.py / test_document_audit_batch_postgres.py); this file
only proves the route layer's own job: query-param validation, delegation, exception-to-status
mapping, and response-body mapping — never re-deriving the service's own classification/summary
logic.
"""

import uuid
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

import app.api.v1.routes.reconciliation as reconciliation_route_module
from app.db.session import get_db_session
from app.main import app
from app.services.reconciliation.document_audit_batch_service import (
    DEFAULT_BATCH_LIMIT,
    MAX_BATCH_LIMIT,
    MIN_BATCH_LIMIT,
    DocumentAuditClassification,
    DocumentAuditSummary,
    DocumentLifecycleAuditBatchResult,
    InvalidAuditBatchLimitError,
    InvalidAuditCursorError,
)
from app.services.reconciliation.document_audit_service import (
    AuditOverallStatus,
    DocumentLifecycleFinding,
    DocumentLifecycleFindingCode,
    FindingSeverity,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def _install_fake_db_session() -> None:
    """The route still needs a get_db_session dependency to resolve — every test here monkeypatches
    `audit_document_lifecycle_batch` itself, so the actual session value is never used."""

    async def _fake_db_session():
        yield object()

    app.dependency_overrides[get_db_session] = _fake_db_session


def _summary(document_id: str, **overrides: object) -> DocumentAuditSummary:
    fields: dict[str, object] = dict(
        document_id=document_id,
        original_filename="report.pdf",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        overall_status=AuditOverallStatus.CONSISTENT,
        classification=DocumentAuditClassification.CONSISTENT,
        findings=(),
    )
    fields.update(overrides)
    return DocumentAuditSummary(**fields)  # type: ignore[arg-type]


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


def _batch_result(**overrides: object) -> DocumentLifecycleAuditBatchResult:
    fields: dict[str, object] = dict(
        scanned_count=0,
        consistent_count=0,
        transitional_count=0,
        warning_count=0,
        inconsistent_count=0,
        not_found_count=0,
        dependency_unavailable_count=0,
        finding_counts={},
        documents=(),
        next_cursor=None,
        has_more=False,
    )
    fields.update(overrides)
    return DocumentLifecycleAuditBatchResult(**fields)  # type: ignore[arg-type]


def _install_fake_audit(
    monkeypatch: pytest.MonkeyPatch, result: DocumentLifecycleAuditBatchResult, calls: list
) -> None:
    async def _fake(session, settings, file_storage, vector_store, *, limit, cursor=None):
        calls.append({"limit": limit, "cursor": cursor})
        return result

    monkeypatch.setattr(reconciliation_route_module, "audit_document_lifecycle_batch", _fake)


# --- limit handling --------------------------------------------------------------------------


def test_default_limit_is_used_when_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    calls: list = []
    _install_fake_audit(monkeypatch, _batch_result(), calls)

    response = client.get("/api/v1/reconciliation/documents/audit")

    assert response.status_code == 200
    assert calls == [{"limit": DEFAULT_BATCH_LIMIT, "cursor": None}]
    assert response.json()["limit"] == DEFAULT_BATCH_LIMIT


def test_explicit_valid_limit_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    calls: list = []
    _install_fake_audit(monkeypatch, _batch_result(), calls)

    response = client.get("/api/v1/reconciliation/documents/audit", params={"limit": 5})

    assert response.status_code == 200
    assert calls == [{"limit": 5, "cursor": None}]
    assert response.json()["limit"] == 5


def test_boundary_limits_are_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    calls: list = []
    _install_fake_audit(monkeypatch, _batch_result(), calls)

    for boundary in (MIN_BATCH_LIMIT, MAX_BATCH_LIMIT):
        response = client.get("/api/v1/reconciliation/documents/audit", params={"limit": boundary})
        assert response.status_code == 200


def test_limit_below_minimum_is_rejected_by_query_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    calls: list = []
    _install_fake_audit(monkeypatch, _batch_result(), calls)

    response = client.get(
        "/api/v1/reconciliation/documents/audit", params={"limit": MIN_BATCH_LIMIT - 1}
    )

    assert response.status_code == 422
    assert calls == []  # rejected before the service is ever called


def test_limit_above_maximum_is_rejected_by_query_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    calls: list = []
    _install_fake_audit(monkeypatch, _batch_result(), calls)

    response = client.get(
        "/api/v1/reconciliation/documents/audit", params={"limit": MAX_BATCH_LIMIT + 1}
    )

    assert response.status_code == 422
    assert calls == []


def test_service_limit_validation_error_is_translated_to_400(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defense in depth: even though FastAPI's Query() bounds already reject most bad limits, the
    service's own InvalidAuditBatchLimitError (should it ever be reached) still maps to a 400, not
    a 500."""
    _install_fake_db_session()

    async def _fake_raises(session, settings, file_storage, vector_store, *, limit, cursor=None):
        raise InvalidAuditBatchLimitError(f"limit must be between 1 and 50, got {limit}.")

    monkeypatch.setattr(reconciliation_route_module, "audit_document_lifecycle_batch", _fake_raises)

    response = client.get("/api/v1/reconciliation/documents/audit", params={"limit": MIN_BATCH_LIMIT})

    assert response.status_code == 400
    assert "detail" in response.json()


# --- cursor handling ---------------------------------------------------------------------------


def test_valid_cursor_is_forwarded_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    calls: list = []
    _install_fake_audit(monkeypatch, _batch_result(), calls)
    opaque_cursor = "some-opaque-cursor-value"

    response = client.get(
        "/api/v1/reconciliation/documents/audit", params={"cursor": opaque_cursor}
    )

    assert response.status_code == 200
    assert calls == [{"limit": DEFAULT_BATCH_LIMIT, "cursor": opaque_cursor}]


def test_malformed_cursor_is_translated_to_400_with_sanitized_body(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()

    async def _fake_raises(session, settings, file_storage, vector_store, *, limit, cursor=None):
        raise InvalidAuditCursorError("Cursor is not valid URL-safe Base64.")

    monkeypatch.setattr(reconciliation_route_module, "audit_document_lifecycle_batch", _fake_raises)

    response = client.get(
        "/api/v1/reconciliation/documents/audit", params={"cursor": "not-a-valid-cursor"}
    )

    assert response.status_code == 400
    body_text = response.text
    assert "Base64" not in body_text  # cursor implementation details never leak
    assert "JSON" not in body_text


# --- empty / populated result mapping -----------------------------------------------------------


def test_empty_result_maps_to_valid_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    calls: list = []
    _install_fake_audit(monkeypatch, _batch_result(), calls)

    response = client.get("/api/v1/reconciliation/documents/audit")

    assert response.status_code == 200
    body = response.json()
    assert body["items"] == []
    assert body["summary"] == {
        "total": 0,
        "consistent": 0,
        "transitional": 0,
        "warning": 0,
        "inconsistent": 0,
        "not_found": 0,
        "dependency_unavailable": 0,
        "finding_counts": {},
    }
    assert body["next_cursor"] is None


def test_populated_result_maps_items_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    calls: list = []
    document_id = str(uuid.uuid4())
    finding = _finding()
    summary = _summary(
        document_id,
        overall_status=AuditOverallStatus.CONSISTENT,
        classification=DocumentAuditClassification.WARNING,
        findings=(finding,),
    )
    result = _batch_result(scanned_count=1, warning_count=1, documents=(summary,))
    _install_fake_audit(monkeypatch, result, calls)

    response = client.get("/api/v1/reconciliation/documents/audit")

    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 1
    item = items[0]
    assert item["document_id"] == document_id
    assert item["original_filename"] == "report.pdf"
    assert item["classification"] == "warning"
    assert item["overall_status"] == "consistent"
    assert len(item["issues"]) == 1
    assert item["issues"][0]["code"] == "stale_ingestion_job"
    assert item["issues"][0]["severity"] == "warning"


def test_populated_result_maps_summary_counts_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    calls: list = []
    result = _batch_result(
        scanned_count=5,
        consistent_count=2,
        transitional_count=1,
        warning_count=1,
        inconsistent_count=1,
        not_found_count=0,
        dependency_unavailable_count=1,
        finding_counts={DocumentLifecycleFindingCode.STALE_INGESTION_JOB: 2},
    )
    _install_fake_audit(monkeypatch, result, calls)

    response = client.get("/api/v1/reconciliation/documents/audit")

    assert response.status_code == 200
    summary = response.json()["summary"]
    assert summary["total"] == 5
    assert summary["consistent"] == 2
    assert summary["transitional"] == 1
    assert summary["warning"] == 1
    assert summary["inconsistent"] == 1
    assert summary["not_found"] == 0
    assert summary["dependency_unavailable"] == 1
    assert summary["finding_counts"] == {"stale_ingestion_job": 2}


def test_next_cursor_present_when_more_documents_remain(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    calls: list = []
    result = _batch_result(next_cursor="opaque-next-cursor", has_more=True)
    _install_fake_audit(monkeypatch, result, calls)

    response = client.get("/api/v1/reconciliation/documents/audit")

    assert response.status_code == 200
    assert response.json()["next_cursor"] == "opaque-next-cursor"


def test_next_cursor_absent_on_final_page(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    calls: list = []
    result = _batch_result(next_cursor=None, has_more=False)
    _install_fake_audit(monkeypatch, result, calls)

    response = client.get("/api/v1/reconciliation/documents/audit")

    assert response.status_code == 200
    assert response.json()["next_cursor"] is None


# --- unexpected failure propagation --------------------------------------------------------------


def test_unexpected_service_exception_propagates_as_server_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()

    async def _fake_raises(session, settings, file_storage, vector_store, *, limit, cursor=None):
        raise RuntimeError("unexpected coding defect")

    monkeypatch.setattr(reconciliation_route_module, "audit_document_lifecycle_batch", _fake_raises)

    with pytest.raises(RuntimeError):
        client.get("/api/v1/reconciliation/documents/audit")


# --- delegation / no-duplicated-logic boundary --------------------------------------------------


def test_exactly_one_call_to_the_batch_service(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_db_session()
    calls: list = []
    _install_fake_audit(monkeypatch, _batch_result(), calls)

    client.get("/api/v1/reconciliation/documents/audit")

    assert len(calls) == 1


def test_batch_route_never_calls_the_single_document_auditor_directly() -> None:
    """The batch route must delegate to audit_document_lifecycle_batch() only — never call
    audit_document_lifecycle() itself, which would mean re-auditing outside the service's own
    sequential batch loop. (The module now also exposes a single-document audit route — Phase
    2.8.7, subtask 5 — which legitimately imports and calls audit_document_lifecycle() directly;
    this test scopes the assertion to the batch route's own source only.)"""
    import inspect

    source = inspect.getsource(reconciliation_route_module.audit_documents_batch_route)
    assert "audit_document_lifecycle(" not in source
    assert "audit_document_lifecycle_batch(" in source


def test_router_does_not_recalculate_classification_or_summary_counts() -> None:
    """The route module's own source must contain no bucket-classification logic of its own —
    every count/classification value returned to the client must trace directly to a field already
    present on the service's DocumentAuditSummary/DocumentLifecycleAuditBatchResult."""
    import inspect

    source = inspect.getsource(reconciliation_route_module)
    assert "FindingSeverity.INFO" not in source
    assert "AuditOverallStatus.INCONSISTENT" not in source
    assert "_classify" not in source
