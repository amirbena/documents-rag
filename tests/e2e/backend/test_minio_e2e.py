"""Backend E2E: document upload -> real MinIO -> ingestion -> retrieval -> streaming chat.

Everything tests/e2e/backend/test_upload_to_streaming_chat.py already covers for
FILE_STORAGE_PROVIDER=local, this module covers for FILE_STORAGE_PROVIDER=minio — the real public
HTTP boundary (POST /api/v1/documents, POST /api/v1/chat) against a real, ephemeral MinIO
container, with the app's own get_file_storage()/create_file_storage() dependency chain doing the
provider selection (never a hand-substituted storage instance). Only the embedding/LLM providers
are faked; Postgres, Qdrant, and MinIO are all real Testcontainers-managed services. Runs under
both RAG_ENGINE settings, mirroring tests/e2e/backend/test_multilingual_matrix.py's parametrize
pattern.
"""

import uuid

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.models.document import Document
from app.models.ingestion_job import IngestionStatus
from app.storage.minio_storage import MinioFileStorage
from tests.e2e.backend.fakes import FakeEmbeddingProvider, FakeStreamingLLMProvider
from tests.e2e.backend.sse import iter_sse_events

pytestmark = pytest.mark.e2e

_ENGINES = ["custom", "langchain"]

_HEBREW_TECHNICAL_IDENTIFIER = "Kafka"
_UNIQUE_FACT = "MINIO-E2E-UNIQUE-FACT-97531"

_MINIO_DOCUMENT = (
    f"מדיניות אחסון: המערכת משתמשת בשירות {_HEBREW_TECHNICAL_IDENTIFIER} לניהול תורים של מסמכים.\n"
    f"Unique retrievable fact: {_UNIQUE_FACT}.\n"
).encode()

_ORIGINAL_FILENAME = "מדיניות-אחסון.txt"
_QUESTION = f"What does the uploaded document say about the fact {_UNIQUE_FACT}?"


@pytest.fixture
def minio_bucket_name() -> str:
    """A unique bucket name per test invocation, so parametrized/concurrent runs never collide."""
    return f"e2e-minio-{uuid.uuid4().hex}"


@pytest.fixture
def minio_storage_settings(
    minio_endpoint: str,
    minio_credentials: tuple[str, str],
    minio_bucket_name: str,
    monkeypatch: pytest.MonkeyPatch,
    isolated_test_state: None,
) -> None:
    """Point the app's real, cached Settings singleton at FILE_STORAGE_PROVIDER=minio.

    get_settings() is process-wide cached (functools.lru_cache); mutating attributes on the
    already-cached instance via monkeypatch — exactly how isolated_test_state overrides
    qdrant_collection_name per test — makes every dependency that calls get_settings() (including
    app.storage.factory.create_file_storage(), which the real get_file_storage() route dependency
    calls) resolve MinioFileStorage pointed at the ephemeral container for the duration of this
    test only; monkeypatch reverts it automatically afterward.
    """
    access_key, secret_key = minio_credentials
    settings = get_settings()
    monkeypatch.setattr(settings, "file_storage_provider", "minio")
    monkeypatch.setattr(settings, "minio_endpoint", minio_endpoint)
    monkeypatch.setattr(settings, "minio_access_key", access_key)
    monkeypatch.setattr(settings, "minio_secret_key", secret_key)
    monkeypatch.setattr(settings, "minio_bucket", minio_bucket_name)
    monkeypatch.setattr(settings, "minio_secure", False)
    monkeypatch.setattr(settings, "minio_create_bucket_if_missing", True)


async def _fetch_document_row(
    session_factory: async_sessionmaker[AsyncSession], document_id: str
) -> Document:
    """Load the persisted Document row for `document_id` from the real ephemeral Postgres."""
    async with session_factory() as session:
        result = await session.execute(select(Document).where(Document.id == document_id))
        document = result.scalar_one()
    return document


async def test_upload_to_streaming_chat_through_real_minio(
    minio_app_client: httpx.AsyncClient,
    process_pending_job_minio,
    e2e_session_factory: async_sessionmaker[AsyncSession],
    minio_storage_settings: None,
    minio_endpoint: str,
    minio_credentials: tuple[str, str],
    minio_bucket_name: str,
    fake_embedding_provider: FakeEmbeddingProvider,
    fake_llm_provider: FakeStreamingLLMProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Upload through MinIO, verify the object landed there, ingest it, and stream a grounded answer."""
    monkeypatch.setattr(get_settings(), "rag_engine", "custom")

    # A. Readiness check exercises the real create_file_storage()->MinioFileStorage.ensure_bucket()
    # path (see app/services/platform_health.py's check_file_storage), creating the bucket exactly
    # the way a real deployment's readiness probe would — never a test-only shortcut.
    ready = await minio_app_client.get("/health/ready")
    ready_body = ready.json()
    storage_check = next(check for check in ready_body["checks"] if check["name"] == "file_storage")
    assert storage_check["status"] == "ok"

    # B. Upload a Hebrew document (with an embedded English technical identifier and a unique fact
    # string) through the real HTTP boundary; storage is selected purely via Settings/DI.
    upload = await minio_app_client.post(
        "/api/v1/documents",
        files={"file": (_ORIGINAL_FILENAME, _MINIO_DOCUMENT, "text/plain; charset=utf-8")},
    )
    assert upload.status_code == 202
    document_id = upload.json()["document_id"]
    assert upload.json()["status"] == IngestionStatus.PENDING

    # C. Verify persistence directly against MinIO (never a local filesystem path): the stored
    # object exists under the Document row's real storage_key, with byte-identical content.
    document_row = await _fetch_document_row(e2e_session_factory, document_id)
    assert document_row.storage_provider == "minio"
    assert document_row.storage_bucket == minio_bucket_name
    assert document_row.storage_key

    access_key, secret_key = minio_credentials
    verification_settings = get_settings()
    verification_storage = MinioFileStorage(settings=verification_settings)
    assert await verification_storage.exists(document_row.storage_key) is True
    stored_bytes = await verification_storage.read(document_row.storage_key)
    assert stored_bytes == _MINIO_DOCUMENT

    # D. Drive ingestion to completion via the real IngestionWorker, reading content back from
    # MinIO through create_file_storage() (never a local-filesystem shortcut).
    processed = await process_pending_job_minio()
    assert processed is not None
    assert processed.status == IngestionStatus.COMPLETED

    # E. Ask a question targeting the unique fact and consume the real SSE stream incrementally.
    async with minio_app_client.stream("POST", "/api/v1/chat", json={"question": _QUESTION}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = [event async for event in iter_sse_events(response)]

    event_names = [name for name, _ in events]
    assert event_names[0] == "metadata"
    assert event_names[-1] == "done"
    assert event_names.count("done") == 1
    assert "error" not in event_names

    metadata = events[0][1]
    assert metadata["decision"] == "needs_retrieval"
    assert metadata["retrieval_used"] is True
    sources = metadata["sources"]
    assert sources, "expected at least one retrieved source"
    assert sources[0]["document_id"] == document_id
    assert sources[0]["source"] == _ORIGINAL_FILENAME  # original filename preserved, Unicode intact
    assert sources[0]["chunk_id"]
    assert isinstance(sources[0]["score"], int | float)

    tokens = [data["text"] for name, data in events if name == "token"]
    assert tokens == list(fake_llm_provider.chunks)

    # F. The public response must carry no MinIO implementation detail: no bucket name, endpoint,
    # or presigned URL leaks into the metadata event or the streamed answer.
    response_blob = str(metadata) + "".join(tokens)
    assert minio_bucket_name not in response_blob
    assert minio_endpoint not in response_blob
    assert access_key not in response_blob
    assert secret_key not in response_blob


@pytest.mark.parametrize("engine", _ENGINES)
async def test_both_engines_retrieve_through_real_minio(
    minio_app_client: httpx.AsyncClient,
    process_pending_job_minio,
    minio_storage_settings: None,
    fake_embedding_provider: FakeEmbeddingProvider,
    fake_llm_provider: FakeStreamingLLMProvider,
    monkeypatch: pytest.MonkeyPatch,
    engine: str,
) -> None:
    """Both RAG_ENGINE settings must retrieve/cite the same MinIO-backed document identically."""
    monkeypatch.setattr(get_settings(), "rag_engine", engine)

    # Exercise the real readiness check first — this is what actually creates the per-test bucket
    # via create_file_storage()->MinioFileStorage.ensure_bucket(), exactly as a real deployment's
    # readiness probe would (see app/services/platform_health.py's check_file_storage).
    ready = await minio_app_client.get("/health/ready")
    assert ready.json()["checks"]

    upload = await minio_app_client.post(
        "/api/v1/documents",
        files={"file": (_ORIGINAL_FILENAME, _MINIO_DOCUMENT, "text/plain; charset=utf-8")},
    )
    assert upload.status_code == 202
    document_id = upload.json()["document_id"]

    processed = await process_pending_job_minio()
    assert processed is not None
    assert processed.status == IngestionStatus.COMPLETED

    async with minio_app_client.stream("POST", "/api/v1/chat", json={"question": _QUESTION}) as response:
        assert response.status_code == 200
        events = [event async for event in iter_sse_events(response)]

    event_names = [name for name, _ in events]
    assert event_names[0] == "metadata"
    assert event_names[-1] == "done"
    assert event_names.count("done") == 1
    assert "error" not in event_names

    metadata = events[0][1]
    assert metadata["decision"] == "needs_retrieval"
    assert metadata["retrieval_used"] is True
    sources = metadata["sources"]
    assert sources
    assert sources[0]["document_id"] == document_id
    assert sources[0]["source"] == _ORIGINAL_FILENAME

    tokens = [data["text"] for name, data in events if name == "token"]
    assert tokens == list(fake_llm_provider.chunks)
