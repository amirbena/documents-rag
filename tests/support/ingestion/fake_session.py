"""A minimal in-memory AsyncSession double for the ingestion retry/stale-recovery unit tests.

Dispatches each SELECT by its compiled SQL text (literal binds), mirroring
tests/support/documents/read/fake_session.py's style — no SQLite or any other real database
engine is used, since `app.services.ingestion.retry_service`/`stale_recovery_service` depend on
Postgres-specific row-locking semantics that a fake session cannot faithfully execute; these
fakes only simulate the *filter/order/limit* logic in plain Python, never the locking itself
(that is covered separately by real Postgres integration tests).
"""

import re
from typing import Any

from sqlalchemy.exc import IntegrityError

from app.models.document import Document
from app.models.document_deletion_job import DocumentDeletionJob
from app.models.ingestion_job import IngestionJob


class _Scalars:
    """Stand-in for SQLAlchemy's `Result.scalars()`."""

    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def all(self) -> list[Any]:
        return self._items

    def first(self) -> Any | None:
        return self._items[0] if self._items else None


class _ListResult:
    """Stand-in for a SQLAlchemy `Result` wrapping a list of ORM rows."""

    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def scalars(self) -> _Scalars:
        return _Scalars(self._items)


class FakeIngestionRetrySession:
    """In-memory AsyncSession double simulating the three query shapes retry/recovery issue.

    `unique_active_index`: when True, `commit()` raises `IntegrityError` if two active
    (pending/processing) jobs would exist for the same document_id after the pending add() calls
    are flushed — simulating the real partial unique index's enforcement, so
    `retry_ingestion()`'s IntegrityError-catch path can be exercised without a real database.
    """

    def __init__(self, *, enforce_unique_active_index: bool = True) -> None:
        self.documents: dict[str, Document] = {}
        self.jobs: dict[str, IngestionJob] = {}
        self.deletion_jobs: dict[str, DocumentDeletionJob] = {}
        self._pending_new_jobs: list[IngestionJob] = []
        self.commit_count = 0
        self.rollback_count = 0
        self.enforce_unique_active_index = enforce_unique_active_index
        self.force_next_commit_integrity_error = False
        # Simulates a concurrent transaction's row becoming visible only once this session's own
        # commit is attempted (mirroring the real "phantom insert" race retry_ingestion's
        # IntegrityError-catch path exists for) — set by a test, applied inside commit().
        self.concurrent_winner_job: IngestionJob | None = None

    def add(self, instance: object) -> None:
        if isinstance(instance, Document):
            self.documents[instance.id] = instance
        elif isinstance(instance, DocumentDeletionJob):
            self.deletion_jobs[instance.id] = instance
        elif isinstance(instance, IngestionJob):
            self._pending_new_jobs.append(instance)

    async def get(self, model: type, instance_id: str) -> object | None:
        if model is Document:
            return self.documents.get(instance_id)
        return None

    async def execute(self, stmt: Any) -> _ListResult:
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))

        if "document_deletion_jobs" in compiled:
            deletion_jobs = list(self.deletion_jobs.values())
            eq_match = re.search(r"document_deletion_jobs\.document_id = '([^']*)'", compiled)
            if eq_match:
                deletion_jobs = [job for job in deletion_jobs if job.document_id == eq_match.group(1)]
            deletion_jobs.sort(key=lambda job: (job.created_at, job.id), reverse=True)
            limit_match = re.search(r"LIMIT (\d+)", compiled)
            if limit_match:
                deletion_jobs = deletion_jobs[: int(limit_match.group(1))]
            return _ListResult(deletion_jobs)

        jobs = list(self.jobs.values())

        eq_match = re.search(r"document_id = '([^']*)'", compiled)
        if eq_match:
            jobs = [job for job in jobs if job.document_id == eq_match.group(1)]

        in_match = re.search(r"status IN \(([^)]*)\)", compiled)
        if in_match:
            statuses = {token.strip().strip("'") for token in in_match.group(1).split(",")}
            jobs = [job for job in jobs if job.status.value in statuses]

        eq_status_match = re.search(r"status = '([^']*)'", compiled)
        if eq_status_match:
            jobs = [job for job in jobs if job.status.value == eq_status_match.group(1)]

        if "ORDER BY ingestion_jobs.updated_at ASC" in compiled:
            jobs.sort(key=lambda job: (job.updated_at, job.id))
        else:
            jobs.sort(key=lambda job: (job.created_at, job.id), reverse=True)

        limit_match = re.search(r"LIMIT (\d+)", compiled)
        if limit_match:
            jobs = jobs[: int(limit_match.group(1))]

        return _ListResult(jobs)

    async def commit(self) -> None:
        if self.force_next_commit_integrity_error:
            self.force_next_commit_integrity_error = False
            self._pending_new_jobs.clear()
            if self.concurrent_winner_job is not None:
                self.jobs[self.concurrent_winner_job.id] = self.concurrent_winner_job
            raise IntegrityError("INSERT", {}, Exception("duplicate key value violates unique constraint"))

        if self.enforce_unique_active_index:
            by_document: dict[str, list[IngestionJob]] = {}
            for job in [*self.jobs.values(), *self._pending_new_jobs]:
                if job.status.value in ("pending", "processing"):
                    by_document.setdefault(job.document_id, []).append(job)
            for active_jobs in by_document.values():
                if len({job.id for job in active_jobs}) > 1:
                    self._pending_new_jobs.clear()
                    raise IntegrityError(
                        "INSERT", {}, Exception("duplicate key value violates unique constraint")
                    )

        for job in self._pending_new_jobs:
            self.jobs[job.id] = job
        self._pending_new_jobs.clear()
        self.commit_count += 1

    async def rollback(self) -> None:
        self._pending_new_jobs.clear()
        self.rollback_count += 1
