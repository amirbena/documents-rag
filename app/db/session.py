"""Async SQLAlchemy engine, session factory, and declarative base.

Shared by app models (app/models) and Alembic (alembic/env.py imports Base.metadata).
"""

from collections.abc import AsyncIterator

from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import Settings, get_settings

settings = get_settings()

# Backends whose default pool (NullPool/StaticPool for sqlite) rejects pool_size/max_overflow/
# pool_recycle with a TypeError. Every backend this codebase actually points DATABASE_URL at
# (postgresql+asyncpg — production default, and the same driver Testcontainers-backed
# integration/E2E tests use) resolves to AsyncAdaptedQueuePool, which accepts all three. sqlite
# is listed defensively only: nothing here uses it today, but a future/local override must not
# crash engine creation.
_QUEUE_POOL_INCOMPATIBLE_BACKENDS = frozenset({"sqlite"})


def _pool_kwargs(settings: Settings) -> dict[str, int]:
    """Return the configured pool kwargs, or {} for a backend whose pool doesn't accept them."""
    if make_url(settings.database_url).get_backend_name() in _QUEUE_POOL_INCOMPATIBLE_BACKENDS:
        return {}
    return {
        "pool_size": settings.db_pool_size,
        "max_overflow": settings.db_max_overflow,
        "pool_recycle": settings.db_pool_recycle,
    }


engine = create_async_engine(
    settings.database_url, echo=False, future=True, **_pool_kwargs(settings)
)

async_session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    """Declarative base class for all ORM models."""


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency that yields a request-scoped async DB session."""
    async with async_session_factory() as session:
        yield session
