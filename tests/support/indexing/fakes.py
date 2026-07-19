"""Shared fake VectorStore/session doubles for the indexing package's unit tests.

Used by tests/unit/services/indexing/test_collection_registry.py,
test_vector_deletion_service.py, test_cleanup_job_service.py, and
test_reindex_scheduling_service.py — no real Qdrant/Postgres.
"""

import re
from typing import Any

from app.models.index_collection import IndexCollection
from app.models.reindex_job import ReindexJob
from app.models.vector_cleanup_job import VectorCleanupJob, VectorCleanupStatus


class FakeVectorStore:
    """Minimal VectorStore double tracking created collections and delete calls."""

    def __init__(
        self, existing_dimension: int | None = None, fail_delete_for: set[str] | None = None
    ) -> None:
        self.existing_dimension = existing_dimension
        self.created_collections: list[tuple[str, int]] = []
        self.deleted: list[tuple[str, str]] = []
        self._fail_delete_for = fail_delete_for or set()

    async def get_collection_vector_size(self, collection_name: str) -> int | None:
        return self.existing_dimension

    async def create_collection_if_not_exists(self, collection_name: str, vector_size: int) -> None:
        self.created_collections.append((collection_name, vector_size))

    async def delete_by_document_id(self, collection_name: str, document_id: str) -> None:
        if collection_name in self._fail_delete_for:
            raise RuntimeError(f"could not delete from {collection_name}")
        self.deleted.append((collection_name, document_id))


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


class FakeIndexSession:
    """Minimal AsyncSession double backing IndexCollection/VectorCleanupJob/ReindexJob rows in
    plain dicts, dispatching each SELECT by inspecting the compiled SQL's target table.
    """

    def __init__(self) -> None:
        self.index_collections: dict[str, IndexCollection] = {}
        self.cleanup_jobs: dict[str, VectorCleanupJob] = {}
        self.reindex_jobs: dict[str, ReindexJob] = {}
        self.commit_count = 0

    def add(self, instance: object) -> None:
        if isinstance(instance, IndexCollection):
            self.index_collections[instance.collection_name] = instance
        elif isinstance(instance, VectorCleanupJob):
            self.cleanup_jobs[instance.id] = instance
        elif isinstance(instance, ReindexJob):
            self.reindex_jobs[instance.id] = instance

    async def get(self, model: type, instance_id: str) -> object | None:
        if model is IndexCollection:
            return self.index_collections.get(instance_id)
        return None

    async def execute(self, stmt: Any) -> _ListResult:
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))

        if "reindex_jobs" in compiled:
            jobs = list(self.reindex_jobs.values())
            eq_match = re.search(r"reindex_jobs\.document_id = '([^']*)'", compiled)
            if eq_match:
                jobs = [job for job in jobs if job.document_id == eq_match.group(1)]
            eq_status_match = re.search(r"reindex_jobs\.status = '([^']*)'", compiled)
            if eq_status_match:
                jobs = [job for job in jobs if job.status.value == eq_status_match.group(1)]
            in_match = re.search(r"reindex_jobs\.status IN \(([^)]*)\)", compiled)
            if in_match:
                statuses = {token.strip().strip("'") for token in in_match.group(1).split(",")}
                jobs = [job for job in jobs if job.status.value in statuses]
            # A column-only SELECT (target_collection_name) still routes here — the fake always
            # returns full ReindexJob rows and lets `_Scalars` extract whatever the caller expects.
            if "reindex_jobs.target_collection_name" in compiled and "reindex_jobs.id" not in compiled:
                return _ListResult([job.target_collection_name for job in jobs])
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
        self.commit_count += 1
