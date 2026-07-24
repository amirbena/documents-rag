"""Tests for the shared SQLAlchemy engine's pool wiring (Phase 2.10, app/db/session.py)."""

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.core.config import Settings
from app.db.session import Base, _pool_kwargs, async_session_factory, engine, get_db_session


def _settings(**overrides: object) -> Settings:
    fields = {"DB_POOL_SIZE": 7, "DB_MAX_OVERFLOW": 11, "DB_POOL_RECYCLE": 900}
    fields.update(overrides)
    return Settings(**fields)


def test_postgresql_engine_receives_the_configured_pool_values() -> None:
    settings = _settings(DATABASE_URL="postgresql+asyncpg://postgres:postgres@postgres:5432/rag_db")

    test_engine = create_async_engine(
        settings.database_url, echo=False, future=True, **_pool_kwargs(settings)
    )

    assert test_engine.pool.size() == 7
    assert test_engine.pool._max_overflow == 11
    assert test_engine.pool._recycle == 900


def test_sqlite_backend_receives_no_pool_kwargs() -> None:
    """sqlite's default pool (NullPool/StaticPool) rejects pool_size/max_overflow/pool_recycle —
    _pool_kwargs must return {} rather than let create_async_engine raise a TypeError.

    Settings._validate_url_format requires a host, so a real Settings(DATABASE_URL=...) can never
    hold a hostless sqlite URL (e.g. "sqlite+aiosqlite:///:memory:") — the codebase's Settings
    validation already forecloses that path entirely. This test targets _pool_kwargs's own
    defense-in-depth directly via Settings.model_construct() (bypassing validators, same pattern
    used elsewhere for a downstream check that isn't Settings' own job to enforce), and aiosqlite
    isn't a project dependency, so no real sqlite engine is constructed here either.
    """
    settings = Settings.model_construct(
        database_url="sqlite+aiosqlite:///:memory:",
        db_pool_size=7,
        db_max_overflow=11,
        db_pool_recycle=900,
    )

    assert _pool_kwargs(settings) == {}


def test_module_level_engine_and_session_factory_are_unchanged_in_shape() -> None:
    """The pool-wiring change must not alter the module's existing public surface or behavior."""
    assert isinstance(engine, AsyncEngine)
    assert async_session_factory.kw["bind"] is engine
    assert async_session_factory.kw["expire_on_commit"] is False
    assert issubclass(Base, object) and hasattr(Base, "metadata")

    session_gen = get_db_session()
    assert hasattr(session_gen, "__anext__")
