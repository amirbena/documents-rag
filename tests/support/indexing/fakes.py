"""Shared fake VectorStore/session doubles for the indexing package's unit tests.

Used by tests/unit/services/indexing/test_collection_registry.py,
test_vector_deletion_service.py, and test_cleanup_job_service.py — no real Qdrant/Postgres.
"""

from typing import Any

from app.models.index_collection import IndexCollection
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


class FakeIndexSession:
    """Minimal AsyncSession double backing IndexCollection/VectorCleanupJob rows in plain dicts."""

    def __init__(self) -> None:
        self.index_collections: dict[str, IndexCollection] = {}
        self.cleanup_jobs: dict[str, VectorCleanupJob] = {}
        self.commit_count = 0

    def add(self, instance: object) -> None:
        if isinstance(instance, IndexCollection):
            self.index_collections[instance.collection_name] = instance
        elif isinstance(instance, VectorCleanupJob):
            self.cleanup_jobs[instance.id] = instance

    async def get(self, model: type, instance_id: str) -> object | None:
        if model is IndexCollection:
            return self.index_collections.get(instance_id)
        return None

    async def execute(self, stmt: Any):
        """Simulate: SELECT * FROM vector_cleanup_jobs WHERE status IN (pending, failed)."""
        matching = [
            job
            for job in self.cleanup_jobs.values()
            if job.status in (VectorCleanupStatus.PENDING, VectorCleanupStatus.FAILED)
        ]

        class _Scalars:
            def all(_self) -> list[VectorCleanupJob]:
                return matching

        class _Result:
            def scalars(_self) -> _Scalars:
                return _Scalars()

        return _Result()

    async def commit(self) -> None:
        self.commit_count += 1
