"""Shared fixtures for the backend E2E suite.

Drives the real FastAPI application through a real ASGI HTTP client, against real ephemeral
Postgres and Qdrant containers started via Testcontainers for Python — never the repository's
docker-compose.yml, never fixed host ports, never persistent volumes, never real Ollama. The only
parts of the stack that are faked are the embedding model and the chat LLM: swapped in by
monkeypatching the provider-factory functions each consuming module imports, never by branching
production code on APP_ENV. Every override is restored and every container is torn down at the
end of the E2E session.
"""

import os
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import httpx
import pytest
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import HttpWaitStrategy
from testcontainers.postgres import PostgresContainer

import app.rag.engines.langchain_engine as langchain_engine_module
import app.rag.orchestrator as orchestrator_module
import app.rag.retrieval_service as retrieval_service_module
import app.services.ingestion_worker as ingestion_worker_module
from alembic import command
from app.api.v1.routes.documents import get_local_file_storage
from app.core.config import get_settings
from app.db.session import get_db_session
from app.main import app
from app.services.ingestion_worker import IngestionWorker
from app.services.local_file_storage import LocalFileStorage
from tests.e2e.backend.fakes import FakeEmbeddingProvider, FakeStreamingLLMProvider

_PRODUCTION_ENV_NAMES = {"production", "prod"}
_ALEMBIC_INI = Path(__file__).resolve().parents[3] / "alembic.ini"
_OVERRIDDEN_ENV_KEYS = ("APP_ENV", "DATABASE_URL", "QDRANT_URL", "QDRANT_COLLECTION_NAME", "VECTOR_SIZE")

# Small, deterministic vector size for the E2E suite's fake embeddings/Qdrant collection — the
# real production default (768) buys nothing here, only slower hashing over a wider vector.
_E2E_VECTOR_SIZE = "32"


def _alembic_config() -> Config:
    """Build an Alembic Config pointed at this repo's alembic.ini."""
    return Config(str(_ALEMBIC_INI))


def run_alembic_upgrade(revision: str = "head") -> None:
    """Run `alembic upgrade <revision>` against whatever DATABASE_URL Settings resolves to."""
    command.upgrade(_alembic_config(), revision)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-mark every test collected under tests/e2e/backend/ with the `e2e` marker."""
    for item in items:
        if "tests/e2e/backend/" in str(item.fspath).replace(os.sep, "/"):
            item.add_marker(pytest.mark.e2e)


@pytest.fixture(scope="session", autouse=True)
def _guard_against_production_environment() -> None:
    """Refuse to run any E2E test if the ambient environment looks like production.

    Runs before any container starts or any settings are overridden — checks the environment
    exactly as the process inherited it, mirroring tests/integration/conftest.py's guard.
    """
    app_env = os.environ.get("APP_ENV", "").strip().lower()
    if app_env in _PRODUCTION_ENV_NAMES:
        pytest.fail(
            f"Refusing to run E2E tests: APP_ENV={app_env!r} looks like production. "
            "Backend E2E tests only run against ephemeral Testcontainers-managed services."
        )

    for var in ("DATABASE_URL", "QDRANT_URL"):
        value = os.environ.get(var, "")
        if any(marker in value.lower() for marker in _PRODUCTION_ENV_NAMES):
            pytest.fail(
                f"Refusing to run E2E tests: {var} ({value!r}) looks like a production URL. "
                "Backend E2E tests only run against ephemeral Testcontainers-managed services."
            )


@pytest.fixture(scope="session")
def postgres_container(_guard_against_production_environment: None) -> Iterator[PostgresContainer]:
    """Start one ephemeral Postgres container for the whole E2E session, on a dynamic port."""
    with PostgresContainer("postgres:16-alpine", driver="asyncpg") as container:
        yield container


@pytest.fixture(scope="session")
def qdrant_container(_guard_against_production_environment: None) -> Iterator[DockerContainer]:
    """Start one ephemeral Qdrant container for the whole E2E session, on a dynamic port."""
    container = DockerContainer("qdrant/qdrant:latest").with_exposed_ports(6333)
    container.waiting_for(HttpWaitStrategy(6333, "/"))
    container.start()
    try:
        yield container
    finally:
        container.stop()


@pytest.fixture(scope="session")
def postgres_url(postgres_container: PostgresContainer) -> str:
    """Dynamically generated asyncpg connection URL for the ephemeral Postgres container."""
    return postgres_container.get_connection_url()


@pytest.fixture(scope="session")
def qdrant_url(qdrant_container: DockerContainer) -> str:
    """Dynamically generated HTTP URL for the ephemeral Qdrant container."""
    host = qdrant_container.get_container_host_ip()
    port = qdrant_container.get_exposed_port(6333)
    return f"http://{host}:{port}"


@pytest.fixture(scope="session")
def e2e_collection_name() -> str:
    """A unique Qdrant collection name for this E2E session, never reused across runs."""
    return f"e2e-{uuid.uuid4().hex}"


@pytest.fixture(scope="session", autouse=True)
def e2e_environment(postgres_url: str, qdrant_url: str, e2e_collection_name: str) -> Iterator[None]:
    """Point Settings at the ephemeral containers, for the whole E2E session only.

    Overrides APP_ENV/DATABASE_URL/QDRANT_URL/QDRANT_COLLECTION_NAME/VECTOR_SIZE and clears the
    get_settings() cache so every call inside this session resolves to the ephemeral containers
    and an isolated collection. Restores the prior environment and cache on teardown.
    """
    original = {key: os.environ.get(key) for key in _OVERRIDDEN_ENV_KEYS}

    os.environ["APP_ENV"] = "e2e"
    os.environ["DATABASE_URL"] = postgres_url
    os.environ["QDRANT_URL"] = qdrant_url
    os.environ["QDRANT_COLLECTION_NAME"] = e2e_collection_name
    os.environ["VECTOR_SIZE"] = _E2E_VECTOR_SIZE
    get_settings.cache_clear()

    settings = get_settings()
    assert settings.app_env == "e2e"
    assert settings.database_url == postgres_url
    assert settings.qdrant_url == qdrant_url
    assert settings.qdrant_collection_name == e2e_collection_name
    assert settings.vector_size == int(_E2E_VECTOR_SIZE)

    try:
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()


@pytest.fixture(scope="session")
def migrated_schema(e2e_environment: None) -> None:
    """Run `alembic upgrade head` once against the ephemeral Postgres container."""
    run_alembic_upgrade("head")


@pytest.fixture
def e2e_session_factory(
    postgres_url: str, migrated_schema: None
) -> async_sessionmaker[AsyncSession]:
    """A session factory bound to the ephemeral Postgres container, fresh for each test.

    Function-scoped (not session-scoped) because asyncpg connections are bound to the event loop
    they were created under, and pytest-asyncio gives each test its own event loop — a
    session-scoped engine's pooled connections would be reused across event loops and fail.
    Independent of app.db.session's module-level engine — used both to override the app's
    get_db_session dependency and to drive IngestionWorker directly in test code.
    """
    engine = create_async_engine(postgres_url, future=True)
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


@pytest.fixture
async def isolated_test_state(
    migrated_schema: None,
    e2e_session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Give every test its own clean Postgres rows and its own Qdrant collection.

    The Postgres/Qdrant containers, and the get_settings() cache, are shared for the whole E2E
    session — starting fresh containers per test would be far too slow — so this fixture keeps
    one test's documents/vectors from leaking into another's regardless of run order.
    """
    async with e2e_session_factory() as session:
        await session.execute(text("TRUNCATE TABLE ingestion_jobs, documents RESTART IDENTITY CASCADE"))
        await session.commit()

    monkeypatch.setattr(get_settings(), "qdrant_collection_name", f"e2e-{uuid.uuid4().hex}")


@pytest.fixture
def fake_embedding_provider() -> FakeEmbeddingProvider:
    """A deterministic, hashing-based embedding provider sized to match the E2E Qdrant collection."""
    return FakeEmbeddingProvider(vector_size=get_settings().vector_size)


@pytest.fixture
def fake_llm_provider() -> FakeStreamingLLMProvider:
    """A deterministic streaming LLM provider yielding a fixed sequence of chunks."""
    return FakeStreamingLLMProvider()


@pytest.fixture
def e2e_provider_overrides(
    fake_embedding_provider: FakeEmbeddingProvider,
    fake_llm_provider: FakeStreamingLLMProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Swap the embedding/LLM provider factories for deterministic fakes, for one test.

    The vector store is never faked — QdrantVectorStore keeps talking to the real ephemeral
    Qdrant container. Only the AI-model-backed providers are replaced, and only by monkeypatching
    the provider-factory function each consuming module already imported — no production code
    branches on APP_ENV, and the orchestration/decision/retrieval/prompt-building code paths run
    exactly as they do in production. Patches both engines' LLM resolution
    (app.rag.orchestrator for CustomRagEngine, app.rag.engines.langchain_engine for
    LangChainRagEngine) so either RAG_ENGINE setting gets the same deterministic fake.
    """
    monkeypatch.setattr(
        retrieval_service_module, "get_embedding_provider", lambda settings=None: fake_embedding_provider
    )
    monkeypatch.setattr(orchestrator_module, "get_llm_provider", lambda settings=None: fake_llm_provider)
    monkeypatch.setattr(
        langchain_engine_module, "get_llm_provider", lambda settings=None: fake_llm_provider
    )
    monkeypatch.setattr(
        ingestion_worker_module, "get_embedding_provider", lambda settings=None: fake_embedding_provider
    )


def _db_session_override(session_factory: async_sessionmaker[AsyncSession]):
    """Build a get_db_session override yielding sessions from the ephemeral Postgres container."""

    async def _override() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    return _override


def _local_file_storage_override(root: Path):
    """Build a get_local_file_storage override rooted at a temporary directory."""

    def _override() -> LocalFileStorage:
        return LocalFileStorage(root=root)

    return _override


@pytest.fixture
async def app_client(
    e2e_session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
    e2e_provider_overrides: None,
    isolated_test_state: None,
) -> AsyncIterator[httpx.AsyncClient]:
    """A real ASGI HTTP client against the real FastAPI app, wired to the E2E overrides.

    Document storage and the DB session are overridden via FastAPI dependency injection
    (temporary directory, ephemeral Postgres); the embedding/LLM providers are overridden via
    e2e_provider_overrides. Uses httpx's ASGI transport so genuine streaming (SSE) semantics —
    incremental delivery, event order — remain observable, unlike a fully-buffered test client.
    """
    app.dependency_overrides[get_db_session] = _db_session_override(e2e_session_factory)
    app.dependency_overrides[get_local_file_storage] = _local_file_storage_override(tmp_path)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://e2e-testserver") as client:
        try:
            yield client
        finally:
            app.dependency_overrides.pop(get_db_session, None)
            app.dependency_overrides.pop(get_local_file_storage, None)


@pytest.fixture
def process_pending_job(
    e2e_session_factory: async_sessionmaker[AsyncSession],
    e2e_provider_overrides: None,
    isolated_test_state: None,
):
    """Run the real IngestionWorker against one pending job, using a fresh DB session.

    Real extraction/chunking/Qdrant upsert; embeddings come from the fake provider via
    e2e_provider_overrides.
    """

    async def _process():
        async with e2e_session_factory() as session:
            return await IngestionWorker().process_next_job(session)

    return _process
