"""A minimal in-memory AsyncSession double for app.services.document_query_service unit tests.

Faithfully simulates the exact SELECT shapes document_query_service.py issues (count, list with
order/limit/offset, latest-job-by-document_id, latest-failed-job, batched latest-jobs-in) in
plain Python, by dispatching on each Select statement's mapped entity and compiled SQL text — no
SQLite or any other real database engine is used, per CLAUDE.md's database-testing-style rule.
"""

import re
from typing import Any

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


class _ScalarResult:
    """Stand-in for a SQLAlchemy `Result` wrapping a single scalar (e.g. COUNT)."""

    def __init__(self, value: Any) -> None:
        self._value = value

    def scalar_one(self) -> Any:
        return self._value


class FakeDocumentQuerySession:
    """Minimal AsyncSession double backing Document/IngestionJob rows in plain dicts.

    Tracks `execute_count` so tests can assert a fixed, page-size-independent number of queries
    (N+1 avoidance) rather than one query per row.
    """

    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.jobs: dict[str, IngestionJob] = {}
        self.deletion_jobs: dict[str, DocumentDeletionJob] = {}
        self.execute_count = 0

    def add(self, instance: object) -> None:
        if isinstance(instance, Document):
            self.documents[instance.id] = instance
        elif isinstance(instance, DocumentDeletionJob):
            self.deletion_jobs[instance.id] = instance
        elif isinstance(instance, IngestionJob):
            self.jobs[instance.id] = instance

    async def commit(self) -> None:
        return None

    async def get(self, model: type, instance_id: str) -> object | None:
        if model is Document:
            return self.documents.get(instance_id)
        return None

    async def execute(self, stmt: Any) -> Any:
        self.execute_count += 1
        entity = stmt.column_descriptions[0].get("entity")
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))

        if entity is None:
            return _ScalarResult(len(self.documents))
        if entity is Document:
            return _ListResult(self._select_documents(compiled))
        if entity is IngestionJob:
            return _ListResult(self._select_jobs(compiled))
        if entity is DocumentDeletionJob:
            return _ListResult(self._select_deletion_jobs(compiled))
        raise NotImplementedError(f"Unhandled fake query shape: {compiled}")

    def _select_documents(self, compiled: str) -> list[Document]:
        docs = sorted(self.documents.values(), key=lambda d: (d.created_at, d.id), reverse=True)
        offset_match = re.search(r"OFFSET (\d+)", compiled)
        if offset_match:
            docs = docs[int(offset_match.group(1)) :]
        limit_match = re.search(r"LIMIT (\d+)", compiled)
        if limit_match:
            docs = docs[: int(limit_match.group(1))]
        return docs

    def _select_jobs(self, compiled: str) -> list[IngestionJob]:
        jobs = list(self.jobs.values())

        in_match = re.search(r"document_id IN \(([^)]*)\)", compiled)
        eq_match = re.search(r"document_id = '([^']*)'", compiled)
        status_match = re.search(r"status = '([^']*)'", compiled)

        if in_match:
            ids = {token.strip().strip("'") for token in in_match.group(1).split(",")}
            jobs = [job for job in jobs if job.document_id in ids]
        elif eq_match:
            jobs = [job for job in jobs if job.document_id == eq_match.group(1)]

        if status_match:
            jobs = [job for job in jobs if job.status.value == status_match.group(1)]

        jobs.sort(key=lambda job: (job.created_at, job.id), reverse=True)

        limit_match = re.search(r"LIMIT (\d+)", compiled)
        if limit_match:
            jobs = jobs[: int(limit_match.group(1))]
        return jobs

    def _select_deletion_jobs(self, compiled: str) -> list[DocumentDeletionJob]:
        jobs = list(self.deletion_jobs.values())

        in_match = re.search(r"document_id IN \(([^)]*)\)", compiled)
        eq_match = re.search(r"document_id = '([^']*)'", compiled)

        if in_match:
            ids = {token.strip().strip("'") for token in in_match.group(1).split(",")}
            jobs = [job for job in jobs if job.document_id in ids]
        elif eq_match:
            jobs = [job for job in jobs if job.document_id == eq_match.group(1)]

        jobs.sort(key=lambda job: (job.created_at, job.id), reverse=True)

        limit_match = re.search(r"LIMIT (\d+)", compiled)
        if limit_match:
            jobs = jobs[: int(limit_match.group(1))]
        return jobs
