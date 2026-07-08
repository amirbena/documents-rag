# Alembic migrations

## What this is for

Alembic manages versioned schema migrations for the Postgres database used by this project. It
tracks the sequence of schema changes as Python scripts so the database schema can be created,
upgraded, and rolled back reproducibly across environments.

## Where migration files live

- `alembic/env.py` — runtime configuration; wires Alembic to the app's `Settings` and async
  SQLAlchemy `Base` metadata.
- `alembic/script.py.mako` — template used to generate new migration files.
- `alembic/versions/` — individual migration scripts, one per revision. Currently empty — no
  models or migrations exist yet.
- `alembic.ini` (repo root) — Alembic CLI configuration (script location, logging).

## How migrations connect to the async SQLAlchemy setup

`alembic/env.py` imports `get_settings()` from `app/core/config.py` and `Base` from
`app/db/session.py`:

- `settings.database_url` is written into Alembic's `sqlalchemy.url` at runtime, so migrations
  always target the same database the app connects to (no separate migration DB config to keep
  in sync).
- `Base.metadata` is passed as `target_metadata`, so `alembic revision --autogenerate` can diff
  ORM models declared under `app/models/` against the live schema.
- Because the app uses an async engine (`asyncpg`), `env.py` runs migrations through an async
  connection (`create_async_engine` + `run_sync`) rather than Alembic's default sync path.

## Creating a new migration

Run from the repository root (the app package must be importable, e.g. inside the `app`
container or a local venv with `pip install -e ".[dev]"`):

```bash
alembic revision --autogenerate -m "message"
```

Always review the generated script in `alembic/versions/` before applying it — autogenerate is a
starting point, not a guarantee of correctness.

## Applying migrations

```bash
alembic upgrade head
```

## Common commands

| Command | Purpose |
|---|---|
| `alembic revision --autogenerate -m "message"` | Generate a new migration from model changes |
| `alembic upgrade head` | Apply all pending migrations |
| `alembic downgrade -1` | Roll back the most recent migration |

## Current status

This is scaffold-only: `app/models/` has no ORM models yet, so `alembic/versions/` is empty and
there is nothing to autogenerate or apply. This will be populated once real models are added in a
later milestone.
