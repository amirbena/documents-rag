"""A minimal in-memory AsyncSession double for deletion_service/deletion_worker unit tests.

Dispatches each SELECT by inspecting the compiled SQL's target table plus its WHERE clause,
mirroring tests/support/fake_ingestion_retry_session.py's style — no SQLite or any other real
database engine is used, since deletion scheduling/claiming depends on Postgres-specific
row-locking semantics a fake session cannot faithfully execute. This fake only simulates the
filter/order/limit logic and the partial-unique-index/commit-time IntegrityError in plain Python;
real locking is covered separately by tests/integration/documents/deletion/test_postgres.py.

Supports the four models app.services.documents.deletion_service/deletion_worker and their
dependency (`index_registry.delete_all_tracked_document_vectors`) touch: Document, IngestionJob,
DocumentDeletionJob, VectorCleanupJob.
"""

import re
from typing import Any

from sqlalchemy.exc import IntegrityError

from app.models.document import Document
from app.models.document_deletion_job import DocumentDeletionJob
from app.models.ingestion_job import IngestionJob
from app.models.vector_cleanup_job import VectorCleanupJob, VectorCleanupStatus


class _Scalars:
    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def all(self) -> list[Any]:
        return self._items

    def first(self) -> Any | None:
        return self._items[0] if self._items else None

    def one_or_none(self) -> Any | None:
        return self._items[0] if self._items else None


class _ListResult:
    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def scalars(self) -> _Scalars:
        return _Scalars(self._items)

    def scalar_one_or_none(self) -> Any | None:
        return self._items[0] if self._items else None


class FakeDocumentDeletionSession:
    """In-memory AsyncSession double for request_document_deletion / DocumentDeletionWorker.

    `enforce_unique_active_index`: when True, `commit()` raises IntegrityError if two active
    (pending/processing) DocumentDeletionJob rows would exist for the same document_id — mirroring
    the real partial unique index (`ix_document_deletion_jobs_one_active_per_document`).
    """

    def __init__(self, *, enforce_unique_active_index: bool = True) -> None:
        self.documents: dict[str, Document] = {}
        self.ingestion_jobs: dict[str, IngestionJob] = {}
        self.deletion_jobs: dict[str, DocumentDeletionJob] = {}
        self.cleanup_jobs: dict[str, VectorCleanupJob] = {}
        self._pending_new: list[Any] = []
        self.commit_count = 0
        self.rollback_count = 0
        self.enforce_unique_active_index = enforce_unique_active_index
        self.force_next_commit_integrity_error = False
        self.concurrent_winner_job: DocumentDeletionJob | None = None

    def add(self, instance: object) -> None:
        if isinstance(instance, Document):
            self.documents[instance.id] = instance
        elif isinstance(instance, DocumentDeletionJob):
            self._pending_new.append(instance)
        elif isinstance(instance, IngestionJob):
            self.ingestion_jobs[instance.id] = instance
        elif isinstance(instance, VectorCleanupJob):
            self.cleanup_jobs[instance.id] = instance

    async def get(self, model: type, instance_id: str) -> object | None:
        if model is Document:
            return self.documents.get(instance_id)
        return None

    async def execute(self, stmt: Any) -> _ListResult:
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))

        if "document_deletion_jobs" in compiled:
            pending_deletion_jobs = [j for j in self._pending_new if isinstance(j, DocumentDeletionJob)]
            jobs = [*self.deletion_jobs.values(), *pending_deletion_jobs]
            eq_match = re.search(r"document_deletion_jobs\.document_id = '([^']*)'", compiled)
            if eq_match:
                jobs = [job for job in jobs if job.document_id == eq_match.group(1)]
            in_match = re.search(r"document_deletion_jobs\.status IN \(([^)]*)\)", compiled)
            if in_match:
                statuses = {token.strip().strip("'") for token in in_match.group(1).split(",")}
                jobs = [job for job in jobs if job.status.value in statuses]
            eq_status_match = re.search(r"document_deletion_jobs\.status = '([^']*)'", compiled)
            if eq_status_match:
                jobs = [job for job in jobs if job.status.value == eq_status_match.group(1)]
            in_ids_match = re.search(r"document_deletion_jobs\.document_id IN \(([^)]*)\)", compiled)
            if in_ids_match:
                ids = {token.strip().strip("'") for token in in_ids_match.group(1).split(",")}
                jobs = [job for job in jobs if job.document_id in ids]
            if "ORDER BY document_deletion_jobs.created_at DESC" in compiled:
                jobs.sort(key=lambda job: (job.created_at, job.id), reverse=True)
            elif "ORDER BY document_deletion_jobs.created_at" in compiled:
                jobs.sort(key=lambda job: (job.created_at, job.id))
            limit_match = re.search(r"LIMIT (\d+)", compiled)
            if limit_match:
                jobs = jobs[: int(limit_match.group(1))]
            return _ListResult(jobs)

        if "ingestion_jobs" in compiled:
            jobs = list(self.ingestion_jobs.values())
            eq_match = re.search(r"ingestion_jobs\.document_id = '([^']*)'", compiled)
            if eq_match:
                jobs = [job for job in jobs if job.document_id == eq_match.group(1)]
            jobs.sort(key=lambda job: (job.created_at, job.id), reverse=True)
            limit_match = re.search(r"LIMIT (\d+)", compiled)
            if limit_match:
                jobs = jobs[: int(limit_match.group(1))]
            return _ListResult(jobs)

        if "vector_cleanup_jobs" in compiled:
            jobs = [
                job
                for job in self.cleanup_jobs.values()
                if job.status in (VectorCleanupStatus.PENDING, VectorCleanupStatus.FAILED)
            ]
            return _ListResult(jobs)

        return _ListResult([])

    async def commit(self) -> None:
        if self.force_next_commit_integrity_error:
            self.force_next_commit_integrity_error = False
            self._pending_new.clear()
            if self.concurrent_winner_job is not None:
                self.deletion_jobs[self.concurrent_winner_job.id] = self.concurrent_winner_job
            raise IntegrityError("INSERT", {}, Exception("duplicate key value violates unique constraint"))

        if self.enforce_unique_active_index:
            by_document: dict[str, list[DocumentDeletionJob]] = {}
            for job in [*self.deletion_jobs.values(), *self._pending_new]:
                if not isinstance(job, DocumentDeletionJob):
                    continue
                if job.status.value in ("pending", "processing"):
                    by_document.setdefault(job.document_id, []).append(job)
            for active in by_document.values():
                if len({job.id for job in active}) > 1:
                    self._pending_new.clear()
                    raise IntegrityError(
                        "INSERT", {}, Exception("duplicate key value violates unique constraint")
                    )

        for job in self._pending_new:
            if isinstance(job, DocumentDeletionJob):
                self.deletion_jobs[job.id] = job
        self._pending_new.clear()
        self.commit_count += 1

    async def rollback(self) -> None:
        self._pending_new.clear()
        self.rollback_count += 1
