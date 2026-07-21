# Alembic migrations

## What this is for

Alembic manages versioned schema migrations for the Postgres database used by this project. It
tracks the sequence of schema changes as Python scripts so the database schema can be created,
upgraded, and rolled back reproducibly across environments.

## Where migration files live

- `alembic/env.py` — runtime configuration; wires Alembic to the app's `Settings` and async
  SQLAlchemy `Base` metadata.
- `alembic/script.py.mako` — template used to generate new migration files.
- `alembic/versions/` — individual migration scripts, one per revision.
- `alembic.ini` (repo root) — Alembic CLI configuration (script location, logging).

## How migrations connect to the async SQLAlchemy setup

`alembic/env.py` imports `get_settings()` from `app/core/config.py` and `Base` from
`app/db/session.py`:

- `settings.database_url` is written into Alembic's `sqlalchemy.url` at runtime, so migrations
  always target the same database the app connects to (no separate migration DB config to keep
  in sync).
- `Base.metadata` is passed as `target_metadata`, so `alembic revision --autogenerate` can diff
  ORM models declared under `app/models/` against the live schema. `env.py` also imports
  `app.models` directly (`import app.models  # noqa: F401`) so every model module actually runs
  and registers its table on `Base.metadata` before the diff happens — a model that exists as a
  file but is never imported anywhere is invisible to autogenerate.
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

Docker Compose starts Postgres but does not apply migrations automatically — run this once the
`app`/`postgres` containers are up (recommended, matches the containerized app environment):

```bash
docker compose exec app alembic upgrade head
```

Or, with an activated local virtual environment and Postgres reachable directly:

```bash
alembic upgrade head
```

See [docs/deployment/](../docs/deployment/README.md#migration-sequencing) for when this fits into
the onboarding/deployment flow, and [docs/development/](../docs/development/README.md) for local
setup in general.

## Common commands

| Command | Purpose |
|---|---|
| `alembic revision --autogenerate -m "message"` | Generate a new migration from model changes |
| `alembic upgrade head` | Apply all pending migrations (run locally; use `docker compose exec app alembic upgrade head` when running via Docker) |
| `alembic downgrade -1` | Roll back the most recent migration |

## Current status

`app/models/` now has multiple ORM models spanning documents, ingestion, deletion, re-indexing,
vector cleanup, and index-collection tracking — see
[docs/architecture/](../docs/architecture/README.md) for the module ownership map and
[docs/document-lifecycle/](../docs/document-lifecycle/README.md) for the lifecycle each model's
job table backs. Run `alembic heads` to confirm the current single migration head (there must
always be exactly one) — see [docs/troubleshooting/](../docs/troubleshooting/README.md) if it
ever reports more than one.
