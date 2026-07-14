"""Unit tests for app/services/documents/upload_service.py — content-hash deduplication
integrated into the real upload flow, including constraint-specific concurrent-race recovery.

Real Postgres row-locking/constraint behavior (the actual concurrency guarantee) is covered
separately by tests/integration/documents/upload/test_concurrency.py. These tests use a fake
session that can simulate a commit()-time IntegrityError — with a controllable, asyncpg-shaped
`.orig.constraint_name` — to prove the loser/winner recovery path without a real database.
"""

import inspect
import re
from typing import Any

import asyncpg.exceptions
import pytest
from sqlalchemy.exc import IntegrityError

from app.models.document import Document
from app.models.document_deletion_job import DocumentDeletionJob
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.services.documents import upload_service
from app.services.documents.dedup_service import (
    CONTENT_HASH_CONSTRAINT_NAME,
    MissingWinnerAfterRaceError,
    UploadOutcome,
)
from app.services.documents.upload_service import UploadResult, upload_document
from app.storage.contract import StoredFile

_UNRELATED_CONSTRAINT = "documents_stored_filename_key"


class _ListResult:
    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def scalars(self) -> "_Scalars":
        return _Scalars(self._items)


class _Scalars:
    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def all(self) -> list[Any]:
        return self._items

    def first(self) -> Any | None:
        return self._items[0] if self._items else None


def _fake_orig_for(constraint_name: str | None) -> Exception:
    """Build an exception shaped like the real driver's — `.constraint_name` when a real
    PostgreSQL constraint is being simulated, absent entirely otherwise (mirroring a non-integrity
    DBAPI error, which has no such attribute at all).
    """
    if constraint_name is None:
        return RuntimeError("simulated unrelated DB failure with no constraint diagnostics")
    orig = asyncpg.exceptions.UniqueViolationError(
        f'duplicate key value violates unique constraint "{constraint_name}"'
    )
    orig.constraint_name = constraint_name
    return orig


class _FakeUploadSession:
    """In-memory AsyncSession double simulating the document/job queries + a controllable
    commit()-time IntegrityError, mirroring tests/support/ingestion/fake_session.py's
    `force_next_commit_integrity_error`/`concurrent_winner_*` pattern for the same kind of race.
    """

    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.jobs: dict[str, IngestionJob] = {}
        self.deletion_jobs: dict[str, DocumentDeletionJob] = {}
        self._pending_documents: list[Document] = []
        self._pending_jobs: list[IngestionJob] = []
        self.commit_count = 0
        self.rollback_count = 0
        self.force_next_commit_integrity_error: str | None = None
        self.concurrent_winner_document: Document | None = None
        self.concurrent_winner_job: IngestionJob | None = None

    def add(self, instance: object) -> None:
        if isinstance(instance, Document):
            self._pending_documents.append(instance)
        elif isinstance(instance, IngestionJob):
            self._pending_jobs.append(instance)
        elif isinstance(instance, DocumentDeletionJob):
            self.deletion_jobs[instance.id] = instance

    async def execute(self, stmt: Any) -> _ListResult:
        entity = stmt.column_descriptions[0].get("entity")
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))

        if entity is Document:
            docs = list(self.documents.values())
            match = re.search(r"content_hash = '([^']*)'", compiled)
            if match:
                docs = [doc for doc in docs if doc.content_hash == match.group(1)]
            return _ListResult(docs)

        if entity is IngestionJob:
            jobs = list(self.jobs.values())
            match = re.search(r"document_id = '([^']*)'", compiled)
            if match:
                jobs = [job for job in jobs if job.document_id == match.group(1)]
            jobs.sort(key=lambda job: (job.created_at, job.id), reverse=True)
            return _ListResult(jobs)

        if entity is DocumentDeletionJob:
            jobs = list(self.deletion_jobs.values())
            match = re.search(r"document_id = '([^']*)'", compiled)
            if match:
                jobs = [job for job in jobs if job.document_id == match.group(1)]
            jobs.sort(key=lambda job: (job.created_at, job.id), reverse=True)
            return _ListResult(jobs)

        raise NotImplementedError(f"Unhandled fake query shape: {compiled}")

    async def commit(self) -> None:
        if self.force_next_commit_integrity_error is not None:
            constraint = self.force_next_commit_integrity_error
            self.force_next_commit_integrity_error = None
            self._pending_documents.clear()
            self._pending_jobs.clear()
            # Simulate a concurrent winner's rows becoming visible only once this session's own
            # commit is attempted (mirroring the real "phantom insert" race).
            if self.concurrent_winner_document is not None:
                self.documents[self.concurrent_winner_document.id] = self.concurrent_winner_document
            if self.concurrent_winner_job is not None:
                self.jobs[self.concurrent_winner_job.id] = self.concurrent_winner_job
            raise IntegrityError("INSERT", {}, _fake_orig_for(constraint))

        for doc in self._pending_documents:
            self.documents[doc.id] = doc
        for job in self._pending_jobs:
            self.jobs[job.id] = job
        self._pending_documents.clear()
        self._pending_jobs.clear()
        self.commit_count += 1

    async def rollback(self) -> None:
        self._pending_documents.clear()
        self._pending_jobs.clear()
        self.rollback_count += 1


class _FakeStorage:
    """Minimal FileStorage double tracking save/delete calls; delete can be made to fail."""

    def __init__(self, *, raise_on_delete: Exception | None = None) -> None:
        self.saved_keys: list[str] = []
        self.deleted_keys: list[str] = []
        self._raise_on_delete = raise_on_delete

    async def save(
        self, key: str, content: bytes, *, content_type: str | None = None, metadata: object = None
    ) -> StoredFile:
        self.saved_keys.append(key)
        return StoredFile(key=key, size_bytes=len(content), content_type=content_type, etag="etag-value")

    async def delete(self, key: str) -> None:
        if self._raise_on_delete is not None:
            raise self._raise_on_delete
        self.deleted_keys.append(key)

    async def read(self, key: str) -> bytes:  # pragma: no cover - unused
        raise NotImplementedError

    async def exists(self, key: str) -> bool:  # pragma: no cover - unused
        raise NotImplementedError

    async def get_metadata(self, key: str) -> object:  # pragma: no cover - unused
        raise NotImplementedError

    async def generate_download_url(self, key: str, *, expiry_seconds: int | None = None) -> str:
        raise NotImplementedError  # pragma: no cover - unused


def _existing_document(content_hash: str, **overrides: object) -> Document:
    defaults: dict[str, object] = dict(
        id="existing-doc",
        original_filename="original.pdf",
        stored_filename="stored.pdf",
        content_type="application/pdf",
        file_size=10,
        stored_path="documents/existing-doc/stored.pdf",
        storage_provider="local",
        storage_key="documents/existing-doc/stored.pdf",
        content_hash=content_hash,
    )
    defaults.update(overrides)
    return Document(**defaults)  # type: ignore[arg-type]


def _existing_job(document_id: str, status: IngestionStatus) -> IngestionJob:
    return IngestionJob(id=f"{document_id}-job", document_id=document_id, status=status)


# --- sequential reuse: no storage write, no new rows ---------------------------------------------


async def test_sequential_matching_upload_returns_existing_document() -> None:
    session = _FakeUploadSession()
    content = b"identical bytes"
    content_hash = upload_service.compute_content_hash(content)
    existing = _existing_document(content_hash)
    session.documents[existing.id] = existing
    session.jobs[f"{existing.id}-job"] = _existing_job(existing.id, IngestionStatus.FAILED)
    storage = _FakeStorage()

    result = await upload_document(
        content=content, original_filename="new-name.pdf", content_type="application/pdf",
        storage=storage, session=session,
    )

    assert isinstance(result, UploadResult)
    assert result.document.id == existing.id
    assert result.outcome == UploadOutcome.REUSED_FAILED


async def test_sequential_reuse_does_not_call_storage_save() -> None:
    session = _FakeUploadSession()
    content = b"identical bytes"
    content_hash = upload_service.compute_content_hash(content)
    existing = _existing_document(content_hash)
    session.documents[existing.id] = existing
    session.jobs[f"{existing.id}-job"] = _existing_job(existing.id, IngestionStatus.COMPLETED)
    storage = _FakeStorage()

    await upload_document(
        content=content, original_filename="new-name.pdf", content_type="application/pdf",
        storage=storage, session=session,
    )

    assert storage.saved_keys == []


async def test_sequential_reuse_does_not_create_a_new_document() -> None:
    session = _FakeUploadSession()
    content = b"identical bytes"
    content_hash = upload_service.compute_content_hash(content)
    existing = _existing_document(content_hash)
    session.documents[existing.id] = existing
    session.jobs[f"{existing.id}-job"] = _existing_job(existing.id, IngestionStatus.COMPLETED)
    storage = _FakeStorage()

    await upload_document(
        content=content, original_filename="new-name.pdf", content_type="application/pdf",
        storage=storage, session=session,
    )

    assert list(session.documents.keys()) == [existing.id]


async def test_sequential_reuse_does_not_create_a_new_ingestion_job() -> None:
    session = _FakeUploadSession()
    content = b"identical bytes"
    content_hash = upload_service.compute_content_hash(content)
    existing = _existing_document(content_hash)
    session.documents[existing.id] = existing
    session.jobs[f"{existing.id}-job"] = _existing_job(existing.id, IngestionStatus.COMPLETED)
    storage = _FakeStorage()

    await upload_document(
        content=content, original_filename="new-name.pdf", content_type="application/pdf",
        storage=storage, session=session,
    )

    assert list(session.jobs.keys()) == [f"{existing.id}-job"]


# --- no-match: normal creation path unchanged ----------------------------------------------------


async def test_no_match_performs_one_storage_save() -> None:
    session = _FakeUploadSession()
    storage = _FakeStorage()

    await upload_document(
        content=b"brand new bytes", original_filename="report.pdf", content_type="application/pdf",
        storage=storage, session=session,
    )

    assert len(storage.saved_keys) == 1


async def test_no_match_returns_created() -> None:
    session = _FakeUploadSession()
    storage = _FakeStorage()

    result = await upload_document(
        content=b"brand new bytes", original_filename="report.pdf", content_type="application/pdf",
        storage=storage, session=session,
    )

    assert result.outcome == UploadOutcome.CREATED
    assert result.document.content_hash == upload_service.compute_content_hash(b"brand new bytes")


# --- constraint-specific IntegrityError handling -------------------------------------------------


async def test_content_hash_violation_is_recognized_by_exact_constraint_identity() -> None:
    session = _FakeUploadSession()
    content = b"racing bytes"
    content_hash = upload_service.compute_content_hash(content)
    winner = _existing_document(content_hash, id="winner-doc")
    winner_job = _existing_job(winner.id, IngestionStatus.PENDING)
    session.force_next_commit_integrity_error = CONTENT_HASH_CONSTRAINT_NAME
    session.concurrent_winner_document = winner
    session.concurrent_winner_job = winner_job
    storage = _FakeStorage()

    result = await upload_document(
        content=content, original_filename="report.pdf", content_type="application/pdf",
        storage=storage, session=session,
    )

    assert result.document.id == winner.id
    assert result.outcome == UploadOutcome.REUSED_ACTIVE


async def test_unrelated_integrity_error_is_reraised() -> None:
    session = _FakeUploadSession()
    session.force_next_commit_integrity_error = _UNRELATED_CONSTRAINT
    storage = _FakeStorage()

    with pytest.raises(IntegrityError):
        await upload_document(
            content=b"some bytes", original_filename="report.pdf", content_type="application/pdf",
            storage=storage, session=session,
        )


async def test_unrelated_integrity_error_still_attempts_cleanup_of_new_object() -> None:
    session = _FakeUploadSession()
    session.force_next_commit_integrity_error = _UNRELATED_CONSTRAINT
    storage = _FakeStorage()

    with pytest.raises(IntegrityError):
        await upload_document(
            content=b"some bytes", original_filename="report.pdf", content_type="application/pdf",
            storage=storage, session=session,
        )

    assert storage.deleted_keys == storage.saved_keys
    assert len(storage.deleted_keys) == 1


async def test_unrelated_db_failure_without_constraint_diagnostics_is_reraised() -> None:
    """A non-IntegrityError (or an IntegrityError with no constraint_name at all) must never be
    misclassified as a deduplication race.
    """
    session = _FakeUploadSession()
    session.force_next_commit_integrity_error = None

    class _BrokenSession(_FakeUploadSession):
        async def commit(self) -> None:
            raise RuntimeError("totally unrelated failure")

    broken_session = _BrokenSession()
    storage = _FakeStorage()

    with pytest.raises(RuntimeError):
        await upload_document(
            content=b"some bytes", original_filename="report.pdf", content_type="application/pdf",
            storage=storage, session=broken_session,
        )

    assert len(storage.deleted_keys) == 1


# --- losing a real race: rollback, cleanup own key only, reload winner --------------------------


async def test_losing_race_rolls_back_before_winner_reload() -> None:
    session = _FakeUploadSession()
    content = b"racing bytes"
    content_hash = upload_service.compute_content_hash(content)
    winner = _existing_document(content_hash, id="winner-doc")
    winner_job = _existing_job(winner.id, IngestionStatus.PENDING)
    session.force_next_commit_integrity_error = CONTENT_HASH_CONSTRAINT_NAME
    session.concurrent_winner_document = winner
    session.concurrent_winner_job = winner_job
    storage = _FakeStorage()

    await upload_document(
        content=content, original_filename="report.pdf", content_type="application/pdf",
        storage=storage, session=session,
    )

    assert session.rollback_count == 1
    assert list(session.documents.keys()) == [winner.id]


async def test_losing_race_deletes_only_its_own_object_key() -> None:
    session = _FakeUploadSession()
    content = b"racing bytes"
    content_hash = upload_service.compute_content_hash(content)
    winner = _existing_document(content_hash, id="winner-doc")
    winner_job = _existing_job(winner.id, IngestionStatus.PENDING)
    session.force_next_commit_integrity_error = CONTENT_HASH_CONSTRAINT_NAME
    session.concurrent_winner_document = winner
    session.concurrent_winner_job = winner_job
    storage = _FakeStorage()

    await upload_document(
        content=content, original_filename="report.pdf", content_type="application/pdf",
        storage=storage, session=session,
    )

    assert len(storage.saved_keys) == 1
    assert storage.deleted_keys == storage.saved_keys
    assert winner.storage_key not in storage.deleted_keys


async def test_losing_race_reloads_and_returns_the_winner() -> None:
    session = _FakeUploadSession()
    content = b"racing bytes"
    content_hash = upload_service.compute_content_hash(content)
    winner = _existing_document(content_hash, id="winner-doc")
    winner_job = _existing_job(winner.id, IngestionStatus.COMPLETED)
    session.force_next_commit_integrity_error = CONTENT_HASH_CONSTRAINT_NAME
    session.concurrent_winner_document = winner
    session.concurrent_winner_job = winner_job
    storage = _FakeStorage()

    result = await upload_document(
        content=content, original_filename="report.pdf", content_type="application/pdf",
        storage=storage, session=session,
    )

    assert result.document.id == winner.id
    assert result.ingestion_job.id == winner_job.id
    assert result.outcome == UploadOutcome.REUSED_INDEXED


async def test_missing_winner_after_hash_violation_raises_consistency_error() -> None:
    session = _FakeUploadSession()
    session.force_next_commit_integrity_error = CONTENT_HASH_CONSTRAINT_NAME
    # Deliberately no concurrent_winner_document set — nothing to reload.
    storage = _FakeStorage()

    with pytest.raises(MissingWinnerAfterRaceError):
        await upload_document(
            content=b"racing bytes", original_filename="report.pdf", content_type="application/pdf",
            storage=storage, session=session,
        )


async def test_cleanup_failure_is_logged_but_does_not_hide_a_valid_winner(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = _FakeUploadSession()
    content = b"racing bytes"
    content_hash = upload_service.compute_content_hash(content)
    winner = _existing_document(content_hash, id="winner-doc")
    winner_job = _existing_job(winner.id, IngestionStatus.PENDING)
    session.force_next_commit_integrity_error = CONTENT_HASH_CONSTRAINT_NAME
    session.concurrent_winner_document = winner
    session.concurrent_winner_job = winner_job
    storage = _FakeStorage(raise_on_delete=RuntimeError("MinIO unreachable"))

    with caplog.at_level("WARNING"):
        result = await upload_document(
            content=content, original_filename="report.pdf", content_type="application/pdf",
            storage=storage, session=session,
        )

    assert result.document.id == winner.id
    assert any("Failed to clean up" in record.message for record in caplog.records)


async def test_winner_lifecycle_is_reevaluated_after_reload() -> None:
    """The winner's outcome must reflect its *actual* reloaded ingestion state, never be assumed
    CREATED or copied from the loser's own attempt.
    """
    session = _FakeUploadSession()
    content = b"racing bytes"
    content_hash = upload_service.compute_content_hash(content)
    winner = _existing_document(content_hash, id="winner-doc")
    winner_job = _existing_job(winner.id, IngestionStatus.FAILED)
    session.force_next_commit_integrity_error = CONTENT_HASH_CONSTRAINT_NAME
    session.concurrent_winner_document = winner
    session.concurrent_winner_job = winner_job
    storage = _FakeStorage()

    result = await upload_document(
        content=content, original_filename="report.pdf", content_type="application/pdf",
        storage=storage, session=session,
    )

    assert result.outcome == UploadOutcome.REUSED_FAILED


def test_no_advisory_lock_or_transaction_spanning_storage_dependency() -> None:
    """No `pg_advisory_*` call, and `storage.save()` is called with no open transaction/lock
    wrapped around it — `upload_document()`'s only concurrency guarantee is the database unique
    index, exercised purely through `session.commit()`'s IntegrityError.
    """
    source = inspect.getsource(upload_document)
    assert "pg_advisory" not in source
    assert "session.begin(" not in source
    assert "FOR UPDATE" not in source
