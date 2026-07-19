"""Backend E2E: the batch document lifecycle audit HTTP boundary, over the real public API,
real Testcontainers Postgres.

Documents are seeded directly via a raw DB session (like
tests/integration/reconciliation/test_document_audit_batch_postgres.py) rather than through the
full upload/ingestion HTTP pipeline — this file's purpose is the HTTP boundary and a representative
end-to-end pagination flow, not re-proving the batch service's own pagination-correctness matrix
(already covered by that Postgres-integration suite) or the full upload lifecycle (already covered
by tests/e2e/backend/documents/). Object Storage is faked (always "exists") so every seeded plain
document (no `collection_name`, no jobs) audits as CONSISTENT with no findings, keeping the
classification/summary assertions deterministic without writing real files to disk.
"""

import uuid
from datetime import UTC, datetime

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import app.api.v1.routes.reconciliation as reconciliation_route_module
from app.main import app
from app.models.document import Document

pytestmark = pytest.mark.e2e


class _AlwaysExistsFileStorage:
    async def exists(self, key: str) -> bool:
        return True


@pytest.fixture(autouse=True)
def _override_file_storage():
    app.dependency_overrides[reconciliation_route_module.get_file_storage] = _AlwaysExistsFileStorage
    yield
    app.dependency_overrides.pop(reconciliation_route_module.get_file_storage, None)


async def _seed_document(
    session_factory: async_sessionmaker[AsyncSession], *, created_at: datetime, **overrides: object
) -> Document:
    fields: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        original_filename="report.pdf",
        stored_filename=f"{uuid.uuid4().hex}.pdf",
        content_type="application/pdf",
        file_size=10,
        stored_path="storage/documents/report.pdf",
        storage_provider="local",
        storage_key=f"storage/documents/{uuid.uuid4().hex}.pdf",
        created_at=created_at,
    )
    fields.update(overrides)
    document = Document(**fields)  # type: ignore[arg-type]
    async with session_factory() as session:
        session.add(document)
        await session.commit()
    return document


async def _row_counts(session_factory: async_sessionmaker[AsyncSession]) -> dict[str, int]:
    async with session_factory() as session:
        counts = {}
        tables = (
            "documents",
            "ingestion_jobs",
            "document_deletion_jobs",
            "reindex_jobs",
            "vector_cleanup_jobs",
        )
        for table in tables:
            result = await session.execute(text(f"SELECT count(*) FROM {table}"))
            counts[table] = result.scalar_one()
        return counts


async def test_empty_database_returns_valid_empty_response(app_client: httpx.AsyncClient) -> None:
    response = await app_client.get("/api/v1/reconciliation/documents/audit")

    assert response.status_code == 200
    body = response.json()
    assert body["items"] == []
    assert body["summary"]["total"] == 0
    assert body["summary"]["consistent"] == 0
    assert body["next_cursor"] is None


async def test_multiple_documents_return_deterministic_ordering_and_summary(
    app_client: httpx.AsyncClient, e2e_session_factory: async_sessionmaker[AsyncSession]
) -> None:
    doc_1 = await _seed_document(e2e_session_factory, created_at=datetime(2026, 1, 1, tzinfo=UTC))
    doc_3 = await _seed_document(e2e_session_factory, created_at=datetime(2026, 1, 3, tzinfo=UTC))
    doc_2 = await _seed_document(e2e_session_factory, created_at=datetime(2026, 1, 2, tzinfo=UTC))

    response = await app_client.get("/api/v1/reconciliation/documents/audit", params={"limit": 10})

    assert response.status_code == 200
    body = response.json()
    assert [item["document_id"] for item in body["items"]] == [doc_1.id, doc_2.id, doc_3.id]
    assert all(item["classification"] == "consistent" for item in body["items"])
    assert body["summary"]["total"] == 3
    assert body["summary"]["consistent"] == 3
    assert body["next_cursor"] is None


async def test_pagination_covers_all_documents_with_no_duplicates_or_gaps(
    app_client: httpx.AsyncClient, e2e_session_factory: async_sessionmaker[AsyncSession]
) -> None:
    docs = [
        await _seed_document(e2e_session_factory, created_at=datetime(2026, 1, i + 1, tzinfo=UTC))
        for i in range(5)
    ]

    seen: list[str] = []
    cursor = None
    while True:
        params = {"limit": 2}
        if cursor is not None:
            params["cursor"] = cursor
        response = await app_client.get("/api/v1/reconciliation/documents/audit", params=params)
        assert response.status_code == 200
        body = response.json()
        seen.extend(item["document_id"] for item in body["items"])
        if body["next_cursor"] is None:
            break
        cursor = body["next_cursor"]

    assert seen == [d.id for d in docs]
    assert len(seen) == len(set(seen))


async def test_duplicate_timestamps_tie_break_deterministically_through_the_api(
    app_client: httpx.AsyncClient, e2e_session_factory: async_sessionmaker[AsyncSession]
) -> None:
    same_time = datetime(2026, 1, 1, tzinfo=UTC)
    doc_a = await _seed_document(e2e_session_factory, created_at=same_time)
    doc_b = await _seed_document(e2e_session_factory, created_at=same_time)
    expected_order = sorted([doc_a.id, doc_b.id])

    response = await app_client.get("/api/v1/reconciliation/documents/audit", params={"limit": 10})

    assert response.status_code == 200
    assert [item["document_id"] for item in response.json()["items"]] == expected_order


async def test_invalid_cursor_returns_repository_standard_client_error(app_client: httpx.AsyncClient) -> None:
    response = await app_client.get(
        "/api/v1/reconciliation/documents/audit", params={"cursor": "not-a-valid-cursor!!"}
    )

    assert response.status_code == 400
    body = response.json()
    assert "detail" in body
    assert "Base64" not in response.text


async def test_batch_audit_is_read_only(
    app_client: httpx.AsyncClient, e2e_session_factory: async_sessionmaker[AsyncSession]
) -> None:
    await _seed_document(e2e_session_factory, created_at=datetime(2026, 1, 1, tzinfo=UTC))
    before = await _row_counts(e2e_session_factory)

    response = await app_client.get("/api/v1/reconciliation/documents/audit", params={"limit": 10})
    assert response.status_code == 200

    after = await _row_counts(e2e_session_factory)
    assert after == before
