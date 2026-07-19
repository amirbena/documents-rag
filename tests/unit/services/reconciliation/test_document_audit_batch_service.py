"""Unit tests for app.services.reconciliation.document_audit_batch_service (Phase 2.8.7,
subtask 2) against fake doubles.

Covers only the batch layer's own responsibilities — keyset pagination, cursor encode/decode,
limit validation, and aggregate classification/counting. Deliberately does not repeat
`test_document_audit_service.py`'s single-document finding matrix: pagination tests use plain
unindexed documents with no jobs (so `audit_document_lifecycle` always returns CONSISTENT with no
findings), and classification/aggregation tests replace `audit_document_lifecycle` with a
canned spy so the batch's own bucketing logic is exercised directly.
"""

import base64
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from app.core.config import get_settings
from app.models.document import Document
from app.services.reconciliation import document_audit_batch_service as batch_service
from app.services.reconciliation.document_audit_batch_service import (
    DEFAULT_BATCH_LIMIT,
    MAX_BATCH_LIMIT,
    MIN_BATCH_LIMIT,
    AuditCursor,
    InvalidAuditBatchLimitError,
    InvalidAuditCursorError,
    audit_document_lifecycle_batch,
    decode_audit_cursor,
    encode_audit_cursor,
)
from app.services.reconciliation.document_audit_service import (
    AuditOverallStatus,
    DocumentLifecycleAuditResult,
    DocumentLifecycleFinding,
    DocumentLifecycleFindingCode,
    FindingSeverity,
)

_SETTINGS = get_settings()
_BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


# --- fakes ----------------------------------------------------------------------------------


class _Scalars:
    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def all(self) -> list[Any]:
        return self._items

    def first(self) -> Any | None:
        return self._items[0] if self._items else None


class _ListResult:
    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def scalars(self) -> _Scalars:
        return _Scalars(self._items)


class _FakeBatchSession:
    """In-memory AsyncSession double supporting only what audit_document_lifecycle_batch()
    (plus a job-less audit_document_lifecycle()) needs: Document lookup/listing, and empty job
    tables so every seeded document is CONSISTENT with no findings."""

    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}

    async def get(self, model: type, instance_id: str) -> object | None:
        if model is Document:
            return self.documents.get(instance_id)
        return None

    async def execute(self, stmt: Any) -> _ListResult:
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))

        if any(
            table in compiled
            for table in (
                "ingestion_jobs",
                "document_deletion_jobs",
                "reindex_jobs",
                "vector_cleanup_jobs",
            )
        ):
            return _ListResult([])

        if "FROM documents" in compiled:
            params = stmt.compile().params
            limit = params.get("param_1")
            cursor_created_at = params.get("created_at_1")
            cursor_id = params.get("id_1")

            docs = sorted(self.documents.values(), key=lambda d: (d.created_at, d.id))
            if cursor_created_at is not None:
                docs = [d for d in docs if (d.created_at, d.id) > (cursor_created_at, cursor_id)]
            if limit is not None:
                docs = docs[:limit]
            return _ListResult(docs)

        return _ListResult([])


class _FakeFileStorage:
    async def exists(self, key: str) -> bool:
        return True


class _FakeVectorStore:
    async def get_collection_vector_size(self, collection_name: str) -> int | None:
        return None

    async def count_document_vectors(self, collection_name: str, document_id: str) -> int:
        return 0


def _document(*, created_at: datetime, document_id: str | None = None, **overrides: object) -> Document:
    fields: dict[str, object] = dict(
        id=document_id or str(uuid.uuid4()),
        original_filename="report.pdf",
        stored_filename=f"{uuid.uuid4().hex}.pdf",
        content_type="application/pdf",
        file_size=10,
        stored_path="storage/documents/report.pdf",
        storage_key=f"storage/documents/{uuid.uuid4().hex}.pdf",
        created_at=created_at,
        collection_name=None,
    )
    fields.update(overrides)
    return Document(**fields)  # type: ignore[arg-type]


def _seed(session: _FakeBatchSession, *documents: Document) -> None:
    for document in documents:
        session.documents[document.id] = document


async def _run_batch(
    session: _FakeBatchSession, *, limit: int = DEFAULT_BATCH_LIMIT, cursor: str | None = None
):
    return await audit_document_lifecycle_batch(
        session,
        _SETTINGS,
        _FakeFileStorage(),
        _FakeVectorStore(),
        limit=limit,
        cursor=cursor,
    )


def _canned_result(
    document_id: str,
    *,
    overall_status: AuditOverallStatus = AuditOverallStatus.CONSISTENT,
    findings: tuple[DocumentLifecycleFinding, ...] = (),
) -> DocumentLifecycleAuditResult:
    return DocumentLifecycleAuditResult(
        document_id=document_id,
        overall_status=overall_status,
        findings=findings,
        postgres_state=None,
        storage_state=None,
        vector_state=None,
    )


def _finding(code: DocumentLifecycleFindingCode, severity: FindingSeverity) -> DocumentLifecycleFinding:
    return DocumentLifecycleFinding(
        code=code,
        severity=severity,
        summary="test finding",
        expected_state="expected",
        actual_state="actual",
        suggested_action="none",
        destructive_risk=False,
    )


# --- 1-5: limit validation --------------------------------------------------------------------


async def test_default_limit_is_20() -> None:
    assert DEFAULT_BATCH_LIMIT == 20


async def test_minimum_valid_limit_is_accepted() -> None:
    session = _FakeBatchSession()
    result = await _run_batch(session, limit=MIN_BATCH_LIMIT)
    assert result.scanned_count == 0


async def test_maximum_valid_limit_is_accepted() -> None:
    session = _FakeBatchSession()
    result = await _run_batch(session, limit=MAX_BATCH_LIMIT)
    assert result.scanned_count == 0


async def test_limit_below_minimum_is_rejected() -> None:
    session = _FakeBatchSession()
    with pytest.raises(InvalidAuditBatchLimitError):
        await _run_batch(session, limit=MIN_BATCH_LIMIT - 1)


async def test_limit_above_maximum_is_rejected() -> None:
    session = _FakeBatchSession()
    with pytest.raises(InvalidAuditBatchLimitError):
        await _run_batch(session, limit=MAX_BATCH_LIMIT + 1)


# --- 6: empty result ---------------------------------------------------------------------------


async def test_empty_result_returns_zero_counts_and_no_cursor() -> None:
    session = _FakeBatchSession()
    result = await _run_batch(session)

    assert result.scanned_count == 0
    assert result.consistent_count == 0
    assert result.transitional_count == 0
    assert result.warning_count == 0
    assert result.inconsistent_count == 0
    assert result.not_found_count == 0
    assert result.dependency_unavailable_count == 0
    assert result.finding_counts == {}
    assert result.documents == ()
    assert result.has_more is False
    assert result.next_cursor is None


# --- 7-9: ordering + bounded page ---------------------------------------------------------------


async def test_documents_are_selected_in_created_at_then_id_order() -> None:
    session = _FakeBatchSession()
    doc_a = _document(created_at=_BASE_TIME, document_id="a")
    doc_b = _document(created_at=_BASE_TIME + timedelta(days=1), document_id="b")
    doc_c = _document(created_at=_BASE_TIME + timedelta(days=2), document_id="c")
    _seed(session, doc_c, doc_a, doc_b)  # seeded out of order

    result = await _run_batch(session, limit=10)

    assert [summary.document_id for summary in result.documents] == ["a", "b", "c"]


async def test_equal_timestamps_use_id_as_tiebreaker() -> None:
    session = _FakeBatchSession()
    doc_b = _document(created_at=_BASE_TIME, document_id="b")
    doc_a = _document(created_at=_BASE_TIME, document_id="a")
    _seed(session, doc_b, doc_a)

    result = await _run_batch(session, limit=10)

    assert [summary.document_id for summary in result.documents] == ["a", "b"]


async def test_at_most_limit_documents_are_returned() -> None:
    session = _FakeBatchSession()
    for i in range(5):
        _seed(session, _document(created_at=_BASE_TIME + timedelta(days=i), document_id=f"doc-{i}"))

    result = await _run_batch(session, limit=3)

    assert result.scanned_count == 3
    assert len(result.documents) == 3


# --- 10-12: lookahead + cursor semantics ---------------------------------------------------------


async def test_lookahead_row_determines_has_more() -> None:
    session = _FakeBatchSession()
    for i in range(4):
        _seed(session, _document(created_at=_BASE_TIME + timedelta(days=i), document_id=f"doc-{i}"))

    result = await _run_batch(session, limit=3)

    assert result.has_more is True
    assert result.scanned_count == 3


async def test_next_cursor_is_based_on_the_last_returned_document() -> None:
    session = _FakeBatchSession()
    ids = [str(uuid.uuid4()) for _ in range(4)]
    docs = [_document(created_at=_BASE_TIME + timedelta(days=i), document_id=ids[i]) for i in range(4)]
    _seed(session, *docs)

    result = await _run_batch(session, limit=3)

    assert result.next_cursor is not None
    decoded = decode_audit_cursor(result.next_cursor)
    assert decoded.document_id == ids[2]
    assert decoded.created_at == docs[2].created_at


async def test_final_page_returns_no_next_cursor() -> None:
    session = _FakeBatchSession()
    for i in range(2):
        _seed(session, _document(created_at=_BASE_TIME + timedelta(days=i), document_id=f"doc-{i}"))

    result = await _run_batch(session, limit=3)

    assert result.has_more is False
    assert result.next_cursor is None


# --- 13-19: cursor encode/decode -----------------------------------------------------------------


async def test_cursor_round_trips() -> None:
    document_id = str(uuid.uuid4())
    encoded = encode_audit_cursor(_BASE_TIME, document_id)
    decoded = decode_audit_cursor(encoded)

    assert decoded == AuditCursor(created_at=_BASE_TIME, document_id=document_id)


async def test_timezone_aware_datetime_survives_round_trip() -> None:
    document_id = str(uuid.uuid4())
    aware_time = datetime(2026, 6, 1, 12, 30, tzinfo=UTC)
    decoded = decode_audit_cursor(encode_audit_cursor(aware_time, document_id))

    assert decoded.created_at == aware_time
    assert decoded.created_at.tzinfo is not None


async def test_malformed_base64_cursor_is_rejected() -> None:
    with pytest.raises(InvalidAuditCursorError):
        decode_audit_cursor("not-valid-base64!!!")


async def test_malformed_json_cursor_is_rejected() -> None:
    garbage = base64.urlsafe_b64encode(b"not json").decode("ascii")
    with pytest.raises(InvalidAuditCursorError):
        decode_audit_cursor(garbage)


async def test_missing_cursor_fields_are_rejected() -> None:
    payload = base64.urlsafe_b64encode(b'{"created_at": "2026-01-01T00:00:00+00:00"}').decode("ascii")
    with pytest.raises(InvalidAuditCursorError):
        decode_audit_cursor(payload)


async def test_invalid_datetime_in_cursor_is_rejected() -> None:
    payload = base64.urlsafe_b64encode(b'{"created_at": "not-a-date", "id": "abc"}').decode("ascii")
    with pytest.raises(InvalidAuditCursorError):
        decode_audit_cursor(payload)


async def test_invalid_document_id_in_cursor_is_rejected() -> None:
    payload = base64.urlsafe_b64encode(
        b'{"created_at": "2026-01-01T00:00:00+00:00", "id": "not-a-uuid"}'
    ).decode("ascii")
    with pytest.raises(InvalidAuditCursorError):
        decode_audit_cursor(payload)


# --- 20-21: consecutive pages have no duplicates/gaps ---------------------------------------------


async def test_consecutive_pages_have_no_duplicates_or_gaps() -> None:
    session = _FakeBatchSession()
    ids = [str(uuid.uuid4()) for _ in range(7)]
    docs = [_document(created_at=_BASE_TIME + timedelta(days=i), document_id=ids[i]) for i in range(7)]
    _seed(session, *docs)

    first_page = await _run_batch(session, limit=3)
    assert [s.document_id for s in first_page.documents] == ids[0:3]
    assert first_page.has_more is True

    second_page = await _run_batch(session, limit=3, cursor=first_page.next_cursor)
    assert [s.document_id for s in second_page.documents] == ids[3:6]
    assert second_page.has_more is True

    third_page = await _run_batch(session, limit=3, cursor=second_page.next_cursor)
    assert [s.document_id for s in third_page.documents] == ids[6:7]
    assert third_page.has_more is False
    assert third_page.next_cursor is None


# --- 22-23/32-35: sequential execution, call count, exception propagation, no mutation -----------


async def test_single_document_auditor_is_called_exactly_once_per_returned_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeBatchSession()
    docs = [
        _document(created_at=_BASE_TIME + timedelta(days=i), document_id=f"doc-{i}") for i in range(3)
    ]
    _seed(session, *docs)

    calls: list[str] = []

    async def _spy(_session, document_id, _settings, _file_storage, _vector_store):
        calls.append(document_id)
        return _canned_result(document_id)

    monkeypatch.setattr(batch_service, "audit_document_lifecycle", _spy)

    await _run_batch(session, limit=10)

    assert calls == ["doc-0", "doc-1", "doc-2"]  # exactly once each, in ascending order


async def test_audits_execute_sequentially_not_concurrently(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeBatchSession()
    docs = [
        _document(created_at=_BASE_TIME + timedelta(days=i), document_id=f"doc-{i}") for i in range(3)
    ]
    _seed(session, *docs)

    in_flight = 0
    max_in_flight = 0

    async def _spy(_session, document_id, _settings, _file_storage, _vector_store):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        in_flight -= 1
        return _canned_result(document_id)

    monkeypatch.setattr(batch_service, "audit_document_lifecycle", _spy)

    await _run_batch(session, limit=10)

    assert max_in_flight == 1


async def test_unexpected_audit_exceptions_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeBatchSession()
    _seed(session, _document(created_at=_BASE_TIME, document_id="doc-0"))

    async def _spy(*_args, **_kwargs):
        raise AttributeError("unexpected coding defect")

    monkeypatch.setattr(batch_service, "audit_document_lifecycle", _spy)

    with pytest.raises(AttributeError):
        await _run_batch(session)


async def test_batch_service_has_no_commit_and_performs_no_mutation() -> None:
    session = _FakeBatchSession()
    assert not hasattr(session, "commit")
    assert not hasattr(session, "add")
    assert not hasattr(session, "delete")

    document = _document(created_at=_BASE_TIME, document_id="doc-0")
    _seed(session, document)
    original_collection_name = document.collection_name

    await _run_batch(session)

    assert document.collection_name == original_collection_name


async def test_batch_service_performs_no_external_orphan_scanning() -> None:
    """The fake FileStorage/VectorStore expose only the two lookup methods the single-document
    auditor calls — no bucket-listing or collection-scanning method exists to invoke."""
    assert not hasattr(_FakeFileStorage(), "list_objects")
    assert not hasattr(_FakeVectorStore(), "list_collections")


# --- 24-28: classification buckets (canned spy, one primary bucket each) -------------------------


@dataclass
class _ClassificationCase:
    name: str
    overall_status: AuditOverallStatus
    findings: tuple[DocumentLifecycleFinding, ...]
    expected_bucket: str


_CLASSIFICATION_CASES = [
    _ClassificationCase("consistent", AuditOverallStatus.CONSISTENT, (), "consistent_count"),
    _ClassificationCase(
        "transitional",
        AuditOverallStatus.CONSISTENT,
        (_finding(DocumentLifecycleFindingCode.INGESTION_IN_PROGRESS, FindingSeverity.INFO),),
        "transitional_count",
    ),
    _ClassificationCase(
        "warning",
        AuditOverallStatus.CONSISTENT,
        (_finding(DocumentLifecycleFindingCode.STALE_INGESTION_JOB, FindingSeverity.WARNING),),
        "warning_count",
    ),
    _ClassificationCase(
        "inconsistent",
        AuditOverallStatus.INCONSISTENT,
        (_finding(DocumentLifecycleFindingCode.OBJECT_MISSING, FindingSeverity.ERROR),),
        "inconsistent_count",
    ),
    _ClassificationCase(
        "not_found",
        AuditOverallStatus.NOT_FOUND,
        (_finding(DocumentLifecycleFindingCode.DOCUMENT_MISSING, FindingSeverity.ERROR),),
        "not_found_count",
    ),
]


@pytest.mark.parametrize("case", _CLASSIFICATION_CASES, ids=lambda c: c.name)
async def test_classification_buckets(case: _ClassificationCase, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeBatchSession()
    _seed(session, _document(created_at=_BASE_TIME, document_id="doc-0"))

    async def _spy(_session, document_id, _settings, _file_storage, _vector_store):
        return _canned_result(document_id, overall_status=case.overall_status, findings=case.findings)

    monkeypatch.setattr(batch_service, "audit_document_lifecycle", _spy)

    result = await _run_batch(session)

    all_buckets = {
        "consistent_count",
        "transitional_count",
        "warning_count",
        "inconsistent_count",
        "not_found_count",
    }
    for bucket in all_buckets:
        expected = 1 if bucket == case.expected_bucket else 0
        assert getattr(result, bucket) == expected, bucket


# --- 29-31: finding-code + dependency-unavailable aggregation -------------------------------------


async def test_finding_code_aggregation_is_correct(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeBatchSession()
    _seed(
        session,
        _document(created_at=_BASE_TIME, document_id="doc-0"),
        _document(created_at=_BASE_TIME + timedelta(days=1), document_id="doc-1"),
    )

    async def _spy(_session, document_id, _settings, _file_storage, _vector_store):
        if document_id == "doc-0":
            return _canned_result(
                document_id,
                findings=(
                    _finding(DocumentLifecycleFindingCode.STALE_INGESTION_JOB, FindingSeverity.WARNING),
                    _finding(DocumentLifecycleFindingCode.VECTOR_CLEANUP_INCOMPLETE, FindingSeverity.WARNING),
                ),
            )
        return _canned_result(
            document_id,
            findings=(_finding(DocumentLifecycleFindingCode.STALE_INGESTION_JOB, FindingSeverity.WARNING),),
        )

    monkeypatch.setattr(batch_service, "audit_document_lifecycle", _spy)

    result = await _run_batch(session)

    assert result.finding_counts[DocumentLifecycleFindingCode.STALE_INGESTION_JOB] == 2
    assert result.finding_counts[DocumentLifecycleFindingCode.VECTOR_CLEANUP_INCOMPLETE] == 1


async def test_dependency_unavailable_counts_documents_not_findings(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeBatchSession()
    _seed(session, _document(created_at=_BASE_TIME, document_id="doc-0"))

    async def _spy(_session, document_id, _settings, _file_storage, _vector_store):
        return _canned_result(
            document_id,
            findings=(
                _finding(
                    DocumentLifecycleFindingCode.STORAGE_INSPECTION_UNAVAILABLE, FindingSeverity.WARNING
                ),
                _finding(
                    DocumentLifecycleFindingCode.VECTOR_INSPECTION_UNAVAILABLE, FindingSeverity.WARNING
                ),
            ),
        )

    monkeypatch.setattr(batch_service, "audit_document_lifecycle", _spy)

    result = await _run_batch(session)

    assert result.dependency_unavailable_count == 1  # one document, despite two findings
    assert result.warning_count == 1


async def test_document_with_multiple_findings_belongs_to_one_primary_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeBatchSession()
    _seed(session, _document(created_at=_BASE_TIME, document_id="doc-0"))

    async def _spy(_session, document_id, _settings, _file_storage, _vector_store):
        return _canned_result(
            document_id,
            findings=(
                _finding(DocumentLifecycleFindingCode.STALE_INGESTION_JOB, FindingSeverity.WARNING),
                _finding(DocumentLifecycleFindingCode.VECTOR_CLEANUP_INCOMPLETE, FindingSeverity.WARNING),
                _finding(
                    DocumentLifecycleFindingCode.VECTOR_INSPECTION_UNAVAILABLE, FindingSeverity.WARNING
                ),
            ),
        )

    monkeypatch.setattr(batch_service, "audit_document_lifecycle", _spy)

    result = await _run_batch(session)

    assert result.warning_count == 1
    assert result.consistent_count == 0
    assert result.transitional_count == 0
    assert result.inconsistent_count == 0
    assert result.not_found_count == 0
