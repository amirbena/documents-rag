"""Unit tests for app.services.indexing.reindex_inspection_service against a fake session double.

Covers inspect_document_reindex_state()'s state derivation, staleness, and can_schedule/
can_activate hints. Real PostgreSQL row/query behavior is covered separately by
tests/integration/indexing/test_reindex_inspection_postgres.py.
"""

import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from app.core.config import Settings, get_settings
from app.models.document import Document
from app.models.document_deletion_job import DocumentDeletionJob, DocumentDeletionStatus
from app.models.reindex_job import ReindexJob, ReindexJobStatus
from app.rag.embedding_config import get_active_embedding_config
from app.schemas.reindex import ReindexLifecycleState
from app.services.indexing.reindex_inspection_service import (
    inspect_document_reindex_state,
    sanitize_reindex_error,
)
from tests.support.indexing.builders import build_document, build_reindex_job

_BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)
_DESIRED_CONFIG = get_active_embedding_config(get_settings())
_DESIRED_COLLECTION = _DESIRED_CONFIG.collection_name


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


class _FakeInspectionSession:
    """In-memory AsyncSession double for inspect_document_reindex_state()."""

    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.reindex_jobs: dict[str, ReindexJob] = {}
        self.deletion_jobs: dict[str, DocumentDeletionJob] = {}

    async def get(self, model: type, instance_id: str) -> object | None:
        if model is Document:
            return self.documents.get(instance_id)
        return None

    async def execute(self, stmt: Any) -> _ListResult:
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))

        if "reindex_jobs" in compiled:
            jobs = list(self.reindex_jobs.values())
            eq_match = re.search(r"reindex_jobs\.document_id = '([^']*)'", compiled)
            if eq_match:
                jobs = [job for job in jobs if job.document_id == eq_match.group(1)]
            in_match = re.search(r"reindex_jobs\.status IN \(([^)]*)\)", compiled)
            if in_match:
                statuses = {token.strip().strip("'") for token in in_match.group(1).split(",")}
                jobs = [job for job in jobs if job.status.value in statuses]
            jobs.sort(key=lambda job: (job.created_at, job.id), reverse=True)
            limit_match = re.search(r"LIMIT (\d+)", compiled)
            if limit_match:
                jobs = jobs[: int(limit_match.group(1))]
            return _ListResult(jobs)

        if "document_deletion_jobs" in compiled:
            jobs = list(self.deletion_jobs.values())
            eq_match = re.search(r"document_deletion_jobs\.document_id = '([^']*)'", compiled)
            if eq_match:
                jobs = [job for job in jobs if job.document_id == eq_match.group(1)]
            jobs.sort(key=lambda job: (job.created_at, job.id), reverse=True)
            limit_match = re.search(r"LIMIT (\d+)", compiled)
            if limit_match:
                jobs = jobs[: int(limit_match.group(1))]
            return _ListResult(jobs)

        return _ListResult([])


def _settings() -> Settings:
    settings = get_settings()
    return settings


def _current_document(**overrides: object) -> Document:
    """A document already indexed under exactly the platform's current desired configuration."""
    fields: dict[str, object] = dict(
        collection_name=_DESIRED_COLLECTION,
        embedding_provider=_DESIRED_CONFIG.provider,
        embedding_model=_DESIRED_CONFIG.model,
        embedding_dimension=_DESIRED_CONFIG.dimension,
        embedding_version=_DESIRED_CONFIG.embedding_version,
        chunking_version=_DESIRED_CONFIG.chunking_version,
        indexed_at=_BASE_TIME,
    )
    fields.update(overrides)
    return build_document(**fields)


def _deletion_job(document_id: str, status: DocumentDeletionStatus) -> DocumentDeletionJob:
    return DocumentDeletionJob(
        id=str(uuid.uuid4()),
        document_id=document_id,
        status=status,
        vector_cleanup_completed=False,
        storage_cleanup_completed=False,
        created_at=_BASE_TIME,
    )


async def test_inspection_returns_none_for_missing_document() -> None:
    session = _FakeInspectionSession()

    result = await inspect_document_reindex_state(session, str(uuid.uuid4()), _settings())

    assert result is None


async def test_up_to_date_document_reports_correctly() -> None:
    session = _FakeInspectionSession()
    document = _current_document()
    session.documents[document.id] = document

    result = await inspect_document_reindex_state(session, document.id, _settings())

    assert result is not None
    assert result.is_stale is False
    assert result.state == ReindexLifecycleState.UP_TO_DATE
    assert result.can_schedule is False
    assert result.active_index.collection_name == document.collection_name
    assert result.desired_index.collection_name == _DESIRED_COLLECTION


async def test_stale_document_reports_correctly() -> None:
    session = _FakeInspectionSession()
    document = _current_document(
        collection_name="documents__ollama__old-model__ev0__cv0__d768",
        embedding_model="old-model",
        embedding_version="v0",
        chunking_version="v0",
    )
    session.documents[document.id] = document

    result = await inspect_document_reindex_state(session, document.id, _settings())

    assert result is not None
    assert result.is_stale is True
    assert result.state == ReindexLifecycleState.STALE
    assert result.can_schedule is True


async def test_document_without_active_index_reports_not_indexed() -> None:
    session = _FakeInspectionSession()
    document = build_document(collection_name=None)
    session.documents[document.id] = document

    result = await inspect_document_reindex_state(session, document.id, _settings())

    assert result is not None
    assert result.state == ReindexLifecycleState.NOT_INDEXED
    assert result.can_schedule is False
    assert result.active_index.collection_name is None


async def test_latest_job_is_included_when_one_exists() -> None:
    session = _FakeInspectionSession()
    document = _current_document(
        collection_name="documents__ollama__old-model__ev0__cv0__d768", embedding_version="v0"
    )
    session.documents[document.id] = document
    job = build_reindex_job(
        document.id,
        ReindexJobStatus.PROCESSING,
        source_collection_name=document.collection_name,
        target_collection_name=_DESIRED_COLLECTION,
    )
    session.reindex_jobs[job.id] = job

    result = await inspect_document_reindex_state(session, document.id, _settings())

    assert result is not None
    assert result.latest_job is job
    assert result.state == ReindexLifecycleState.REINDEX_PROCESSING


async def test_build_completed_but_unactivated_job_is_target_built_not_activated() -> None:
    session = _FakeInspectionSession()
    document = _current_document(
        collection_name="documents__ollama__old-model__ev0__cv0__d768", embedding_version="v0"
    )
    session.documents[document.id] = document
    job = build_reindex_job(
        document.id,
        ReindexJobStatus.COMPLETED,
        source_collection_name=document.collection_name,
        target_collection_name=_DESIRED_COLLECTION,
        completed_at=_BASE_TIME,
        activated_at=None,
    )
    session.reindex_jobs[job.id] = job

    result = await inspect_document_reindex_state(session, document.id, _settings())

    assert result is not None
    assert result.state == ReindexLifecycleState.TARGET_BUILT
    assert result.can_activate is True


async def test_activated_job_reports_activated() -> None:
    session = _FakeInspectionSession()
    document = _current_document()  # now serving the target, post-activation
    session.documents[document.id] = document
    job = build_reindex_job(
        document.id,
        ReindexJobStatus.COMPLETED,
        source_collection_name="documents__ollama__old-model__ev0__cv0__d768",
        target_collection_name=_DESIRED_COLLECTION,
        completed_at=_BASE_TIME,
        activated_at=_BASE_TIME + timedelta(minutes=5),
    )
    session.reindex_jobs[job.id] = job

    result = await inspect_document_reindex_state(session, document.id, _settings())

    assert result is not None
    assert result.state == ReindexLifecycleState.ACTIVATED
    assert result.can_activate is False  # already activated
    assert result.is_stale is False


async def test_failed_job_reports_failed() -> None:
    session = _FakeInspectionSession()
    document = _current_document(
        collection_name="documents__ollama__old-model__ev0__cv0__d768", embedding_version="v0"
    )
    session.documents[document.id] = document
    job = build_reindex_job(
        document.id,
        ReindexJobStatus.FAILED,
        source_collection_name=document.collection_name,
        target_collection_name=_DESIRED_COLLECTION,
        error_message="some internal failure",
    )
    session.reindex_jobs[job.id] = job

    result = await inspect_document_reindex_state(session, document.id, _settings())

    assert result is not None
    assert result.state == ReindexLifecycleState.FAILED
    assert result.can_activate is False


async def test_historical_job_for_a_superseded_target_does_not_mask_current_staleness() -> None:
    """An old activated job for a configuration nobody wants anymore must not report ACTIVATED
    forever once the desired configuration has moved on again."""
    session = _FakeInspectionSession()
    document = _current_document(
        collection_name="documents__ollama__superseded-model__ev0__cv0__d768",
        embedding_model="superseded-model",
        embedding_version="v0",
    )
    session.documents[document.id] = document
    stale_historical_job = build_reindex_job(
        document.id,
        ReindexJobStatus.COMPLETED,
        source_collection_name="documents__ollama__ancient-model__ev-1__cv-1__d768",
        target_collection_name="documents__ollama__superseded-model__ev0__cv0__d768",
        completed_at=_BASE_TIME,
        activated_at=_BASE_TIME,
    )
    session.reindex_jobs[stale_historical_job.id] = stale_historical_job

    result = await inspect_document_reindex_state(session, document.id, _settings())

    assert result is not None
    assert result.latest_job is stale_historical_job  # still reported, for visibility
    assert result.state == ReindexLifecycleState.STALE  # not ACTIVATED
    assert result.is_stale is True
    assert result.can_schedule is True


async def test_deletion_pending_blocks_schedule_and_activate_and_reports_deletion_blocked() -> None:
    session = _FakeInspectionSession()
    document = _current_document(
        collection_name="documents__ollama__old-model__ev0__cv0__d768", embedding_version="v0"
    )
    session.documents[document.id] = document
    job = build_reindex_job(
        document.id,
        ReindexJobStatus.COMPLETED,
        source_collection_name=document.collection_name,
        target_collection_name=_DESIRED_COLLECTION,
        activated_at=None,
    )
    session.reindex_jobs[job.id] = job
    deletion_job = _deletion_job(document.id, DocumentDeletionStatus.PENDING)
    session.deletion_jobs[deletion_job.id] = deletion_job

    result = await inspect_document_reindex_state(session, document.id, _settings())

    assert result is not None
    assert result.state == ReindexLifecycleState.DELETION_BLOCKED
    assert result.can_schedule is False
    assert result.can_activate is False


async def test_desired_index_includes_live_chunk_settings() -> None:
    session = _FakeInspectionSession()
    document = _current_document()
    session.documents[document.id] = document

    settings = _settings()
    result = await inspect_document_reindex_state(session, document.id, settings)

    assert result is not None
    assert result.desired_index.chunk_size == settings.chunk_size
    assert result.desired_index.chunk_overlap == settings.chunk_overlap
    assert result.active_index.chunk_size is None
    assert result.active_index.chunk_overlap is None


def test_sanitize_reindex_error_never_returns_raw_text() -> None:
    raw = "connection refused at internal-host:6333"
    safe = sanitize_reindex_error(raw)
    assert safe is not None
    assert raw not in safe
    assert sanitize_reindex_error(None) is None
