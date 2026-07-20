"""Unit tests for app.services.reconciliation.collection_reconciliation_report_service
(Phase 2.8.7, subtask 5) against fake doubles.

Covers report composition/classification only — narrow fakes for IndexCollection lookup, the
document-count aggregate query, and VectorStore.count_collection_vectors(). Does not repeat
QdrantVectorStore's own count_collection_vectors unit matrix (test_qdrant_vector_store.py).
"""

import re
from datetime import UTC, datetime

import pytest

from app.core.config import get_settings
from app.models.index_collection import IndexCollection, IndexCollectionStatus
from app.rag.providers.qdrant_vector_store import QdrantVectorStoreError
from app.services.reconciliation.collection_reconciliation_report_service import (
    CollectionReportClassification,
    CollectionReportFindingCode,
    InvalidCollectionNameError,
    build_collection_reconciliation_report,
    validate_collection_name,
)

_SETTINGS = get_settings()


class _Scalar:
    def __init__(self, value: int) -> None:
        self._value = value

    def scalar_one(self) -> int:
        return self._value


class _FakeReportSession:
    """In-memory AsyncSession double for build_collection_reconciliation_report()."""

    def __init__(self) -> None:
        self.index_collections: dict[str, IndexCollection] = {}
        self.document_counts: dict[str, int] = {}
        self.committed = False

    async def get(self, model: type, key: str) -> object | None:
        if model is IndexCollection:
            return self.index_collections.get(key)
        return None

    async def execute(self, stmt: object) -> _Scalar:
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))  # type: ignore[attr-defined]
        match = re.search(r"documents\.collection_name = '([^']*)'", compiled)
        collection_name = match.group(1) if match else None
        return _Scalar(self.document_counts.get(collection_name, 0))

    async def commit(self) -> None:  # pragma: no cover - must never be called
        self.committed = True
        raise AssertionError("report must never commit")

    def add(self, instance: object) -> None:  # pragma: no cover - must never be called
        raise AssertionError("report must never call session.add()")


class _FakeVectorStore:
    def __init__(
        self,
        *,
        counts: dict[str, int] | None = None,
        unavailable: set[str] | None = None,
    ) -> None:
        self._counts = counts or {}
        self._unavailable = unavailable or set()
        self.calls: list[str] = []

    async def count_collection_vectors(self, collection_name: str) -> int | None:
        self.calls.append(collection_name)
        if collection_name in self._unavailable:
            raise QdrantVectorStoreError("qdrant unreachable")
        return self._counts.get(collection_name)


def _index_collection(collection_name: str, **overrides: object) -> IndexCollection:
    fields: dict[str, object] = dict(
        collection_name=collection_name,
        embedding_provider="ollama",
        embedding_model="model",
        embedding_dimension=768,
        embedding_version="v1",
        chunking_version="v1",
        status=IndexCollectionStatus.ACTIVE,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    fields.update(overrides)
    return IndexCollection(**fields)  # type: ignore[arg-type]


async def _run(
    session: _FakeReportSession, collection_name: str, vector_store: _FakeVectorStore
):
    return await build_collection_reconciliation_report(session, collection_name, _SETTINGS, vector_store)


# --- collection name validation ------------------------------------------------------------------


def test_validate_collection_name_rejects_empty_value() -> None:
    with pytest.raises(InvalidCollectionNameError):
        validate_collection_name("")


def test_validate_collection_name_rejects_path_like_input() -> None:
    with pytest.raises(InvalidCollectionNameError):
        validate_collection_name("../etc/passwd")


def test_validate_collection_name_rejects_control_characters() -> None:
    with pytest.raises(InvalidCollectionNameError):
        validate_collection_name("docs\x00v2")


def test_validate_collection_name_accepts_platform_generated_charset() -> None:
    validate_collection_name("documents__ollama__model__ev1__cv1__d768")  # must not raise


async def test_invalid_collection_name_is_rejected_before_any_query() -> None:
    session = _FakeReportSession()
    vector_store = _FakeVectorStore()

    with pytest.raises(InvalidCollectionNameError):
        await _run(session, "not valid!", vector_store)

    assert vector_store.calls == []


# --- classification rules -------------------------------------------------------------------------


async def test_matching_expected_and_actual_counts_is_healthy() -> None:
    session = _FakeReportSession()
    session.index_collections["docs"] = _index_collection("docs")
    session.document_counts["docs"] = 10
    vector_store = _FakeVectorStore(counts={"docs": 10})

    report = await _run(session, "docs", vector_store)

    assert report.classification == CollectionReportClassification.HEALTHY
    assert report.difference == 0
    assert report.findings == ()


async def test_actual_lower_than_expected_is_inconsistent() -> None:
    session = _FakeReportSession()
    session.index_collections["docs"] = _index_collection("docs")
    session.document_counts["docs"] = 10
    vector_store = _FakeVectorStore(counts={"docs": 4})

    report = await _run(session, "docs", vector_store)

    assert report.classification == CollectionReportClassification.INCONSISTENT
    assert report.difference == -6
    assert len(report.findings) == 1
    assert report.findings[0].code == CollectionReportFindingCode.VECTOR_COUNT_DEFICIT


async def test_actual_higher_than_expected_is_healthy() -> None:
    """A surplus (multi-chunk documents) is the normal case, never flagged."""
    session = _FakeReportSession()
    session.index_collections["docs"] = _index_collection("docs")
    session.document_counts["docs"] = 3
    vector_store = _FakeVectorStore(counts={"docs": 40})

    report = await _run(session, "docs", vector_store)

    assert report.classification == CollectionReportClassification.HEALTHY
    assert report.difference == 37
    assert report.findings == ()


async def test_missing_collection_is_classified_missing() -> None:
    session = _FakeReportSession()
    session.index_collections["docs"] = _index_collection("docs")
    session.document_counts["docs"] = 10
    vector_store = _FakeVectorStore(counts={})  # absent -> None

    report = await _run(session, "docs", vector_store)

    assert report.classification == CollectionReportClassification.MISSING
    assert report.exists is False
    assert report.actual_vector_count == 0
    assert len(report.findings) == 1
    assert report.findings[0].code == CollectionReportFindingCode.COLLECTION_MISSING


async def test_active_collection_reports_is_active_true() -> None:
    session = _FakeReportSession()
    desired = _SETTINGS
    from app.rag.embedding_config import get_active_embedding_config

    desired_name = get_active_embedding_config(desired).collection_name
    session.index_collections[desired_name] = _index_collection(desired_name)
    session.document_counts[desired_name] = 0
    vector_store = _FakeVectorStore(counts={desired_name: 0})

    report = await _run(session, desired_name, vector_store)

    assert report.is_active is True


async def test_inactive_known_collection_reports_is_active_false_but_may_be_healthy() -> None:
    session = _FakeReportSession()
    session.index_collections["old-collection"] = _index_collection(
        "old-collection", status=IndexCollectionStatus.RETIRED
    )
    session.document_counts["old-collection"] = 5
    vector_store = _FakeVectorStore(counts={"old-collection": 5})

    report = await _run(session, "old-collection", vector_store)

    assert report.is_active is False
    assert report.index_collection_status == IndexCollectionStatus.RETIRED
    assert report.classification == CollectionReportClassification.HEALTHY


async def test_collection_without_index_metadata_is_unmanaged() -> None:
    session = _FakeReportSession()
    session.document_counts["orphan-collection"] = 0
    vector_store = _FakeVectorStore(counts={"orphan-collection": 3})

    report = await _run(session, "orphan-collection", vector_store)

    assert report.classification == CollectionReportClassification.UNMANAGED
    assert report.index_collection_status is None
    assert report.embedding_provider is None
    assert len(report.findings) == 1
    assert report.findings[0].code == CollectionReportFindingCode.COLLECTION_UNMANAGED


async def test_zero_expected_and_zero_actual_is_healthy() -> None:
    session = _FakeReportSession()
    session.index_collections["empty-collection"] = _index_collection("empty-collection")
    session.document_counts["empty-collection"] = 0
    vector_store = _FakeVectorStore(counts={"empty-collection": 0})

    report = await _run(session, "empty-collection", vector_store)

    assert report.classification == CollectionReportClassification.HEALTHY
    assert report.expected_vector_count == 0
    assert report.actual_vector_count == 0


async def test_zero_expected_with_nonzero_actual_is_healthy() -> None:
    session = _FakeReportSession()
    session.index_collections["docs"] = _index_collection("docs")
    session.document_counts["docs"] = 0
    vector_store = _FakeVectorStore(counts={"docs": 8})

    report = await _run(session, "docs", vector_store)

    assert report.classification == CollectionReportClassification.HEALTHY
    assert report.difference == 8


async def test_findings_ordering_is_deterministic() -> None:
    session = _FakeReportSession()
    session.index_collections["docs"] = _index_collection("docs")
    session.document_counts["docs"] = 10
    vector_store = _FakeVectorStore(counts={"docs": 2})

    first = await _run(session, "docs", vector_store)
    second = await _run(session, "docs", vector_store)

    assert [f.code for f in first.findings] == [f.code for f in second.findings]


# --- dependency failure / read-only guarantees ----------------------------------------------------


async def test_qdrant_failure_propagates() -> None:
    session = _FakeReportSession()
    session.index_collections["docs"] = _index_collection("docs")
    session.document_counts["docs"] = 10
    vector_store = _FakeVectorStore(unavailable={"docs"})

    with pytest.raises(QdrantVectorStoreError):
        await _run(session, "docs", vector_store)


async def test_report_performs_no_writes_or_commits() -> None:
    session = _FakeReportSession()
    session.index_collections["docs"] = _index_collection("docs")
    session.document_counts["docs"] = 10
    vector_store = _FakeVectorStore(counts={"docs": 10})

    await _run(session, "docs", vector_store)
    # No AssertionError raised means session.add()/commit() were never invoked.


async def test_exact_call_count_to_vector_count_service() -> None:
    session = _FakeReportSession()
    session.index_collections["docs"] = _index_collection("docs")
    session.document_counts["docs"] = 10
    vector_store = _FakeVectorStore(counts={"docs": 10})

    await _run(session, "docs", vector_store)

    assert vector_store.calls == ["docs"]
