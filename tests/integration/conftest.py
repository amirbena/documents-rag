"""Shared fixtures for the Testcontainers-based integration suite.

Spins up ephemeral Postgres and Qdrant containers via Testcontainers for Python — never the
repository's docker-compose.yml, never fixed host ports, never persistent volumes. Overrides
DATABASE_URL/QDRANT_URL/APP_ENV for the duration of the integration test session only, guarded so
tests refuse to run against anything that looks like a production environment.
"""

import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import HttpWaitStrategy
from testcontainers.postgres import PostgresContainer

from alembic import command
from app.core.config import get_settings

_PRODUCTION_ENV_NAMES = {"production", "prod"}
_ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"


def _alembic_config() -> Config:
    """Build an Alembic Config pointed at this repo's alembic.ini.

    Does not set sqlalchemy.url itself — alembic/env.py reads it from get_settings(), which
    the integration_environment fixture has already pointed at the ephemeral Postgres container.
    """
    return Config(str(_ALEMBIC_INI))


def run_alembic_upgrade(revision: str = "head") -> None:
    """Run `alembic upgrade <revision>` against whatever DATABASE_URL Settings resolves to."""
    command.upgrade(_alembic_config(), revision)


def run_alembic_downgrade(revision: str) -> None:
    """Run `alembic downgrade <revision>` against whatever DATABASE_URL Settings resolves to."""
    command.downgrade(_alembic_config(), revision)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-mark every test collected under tests/integration/ with the `integration` marker."""
    for item in items:
        if "tests/integration/" in str(item.fspath).replace(os.sep, "/"):
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session", autouse=True)
def _guard_against_production_environment() -> None:
    """Refuse to run any integration test if the ambient environment looks like production.

    Runs before any container starts or any settings are overridden — checks the environment
    exactly as the process inherited it, so a misconfigured CI/host environment fails loudly
    instead of an integration test silently touching a real deployment.
    """
    app_env = os.environ.get("APP_ENV", "").strip().lower()
    if app_env in _PRODUCTION_ENV_NAMES:
        pytest.fail(
            f"Refusing to run integration tests: APP_ENV={app_env!r} looks like production. "
            "Integration tests only run against ephemeral Testcontainers-managed services."
        )

    for var in ("DATABASE_URL", "QDRANT_URL"):
        value = os.environ.get(var, "")
        if any(marker in value for marker in _PRODUCTION_ENV_NAMES):
            pytest.fail(
                f"Refusing to run integration tests: {var} ({value!r}) looks like a production "
                "URL. Integration tests only run against ephemeral Testcontainers-managed "
                "services."
            )


@pytest.fixture(scope="session")
def postgres_container() -> Iterator[PostgresContainer]:
    """Start one ephemeral Postgres container for the whole integration session, dynamic port."""
    with PostgresContainer("postgres:16-alpine", driver="asyncpg") as container:
        yield container


@pytest.fixture(scope="session")
def qdrant_container() -> Iterator[DockerContainer]:
    """Start one ephemeral Qdrant container for the whole integration session, dynamic port."""
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


@pytest.fixture(scope="session", autouse=True)
def integration_environment(postgres_url: str, qdrant_url: str) -> Iterator[None]:
    """Point Settings at the ephemeral containers for the whole integration session.

    Overrides APP_ENV/DATABASE_URL/QDRANT_URL and clears the get_settings() cache so every
    call inside this session resolves to the ephemeral containers — never a production URL.
    Restores the prior environment and cache on teardown so nothing leaks past this session.
    """
    original = {
        key: os.environ.get(key) for key in ("APP_ENV", "DATABASE_URL", "QDRANT_URL")
    }

    os.environ["APP_ENV"] = "integration"
    os.environ["DATABASE_URL"] = postgres_url
    os.environ["QDRANT_URL"] = qdrant_url
    get_settings.cache_clear()

    settings = get_settings()
    assert settings.app_env == "integration"
    assert settings.database_url == postgres_url
    assert settings.qdrant_url == qdrant_url

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
def migrated_schema(integration_environment: None) -> None:
    """Run `alembic upgrade head` once against the ephemeral Postgres container.

    Depended on by any test/fixture that needs the `documents`/`ingestion_jobs` tables to exist.
    Depends explicitly on integration_environment so DATABASE_URL is guaranteed to already point
    at the ephemeral container before Alembic runs.
    """
    run_alembic_upgrade("head")


@pytest.fixture
async def integration_db_session(postgres_url: str) -> AsyncIterator[AsyncSession]:
    """A real AsyncSession against the ephemeral Postgres container, disposed after the test.

    Independent of app.db.session's module-level engine (built once at import time from
    whatever DATABASE_URL was set then) — this always targets the ephemeral container.
    """
    engine = create_async_engine(postgres_url, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with session_factory() as session:
        yield session
    await engine.dispose()
