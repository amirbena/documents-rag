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
| `alembic heads` | Confirm the current single migration head (there must always be exactly one) |
| `alembic current` | Show the revision the connected database is actually stamped at |

## Migration history reset (Phase 2.10)

This project's 9 incremental development migrations (`acf1b01d5a02` through `fb63f21089ca`) have
been **intentionally squashed** into a single baseline revision, **`a1a302e871c3`**
(`current schema baseline`), with `down_revision = None`. `alembic heads`/`alembic history` now
show exactly this one revision.

**Approved policy — no in-place upgrade path is supported from the deleted revisions.** This was a
development-stage repository with no deployed or persistent database that required preserving an
upgrade path; the decision was to reset history rather than keep an ever-growing chain or write a
migration bridge. If your local database is currently stamped at any of the 9 deleted revision
IDs, `alembic upgrade head` will fail (`Can't locate revision identified by '<old-id>'`) — you
must recreate it:

```bash
# Local venv, Postgres reachable directly:
dropdb rag_db && createdb rag_db
alembic upgrade head

# Or, via Docker Compose:
docker compose exec postgres dropdb -U postgres rag_db
docker compose exec postgres createdb -U postgres rag_db
docker compose exec app alembic upgrade head
```

Do **not** use `alembic stamp head` on an existing database as a substitute for the recreate step
above — stamping only rewrites Alembic's own bookkeeping row; it does not create or verify any
schema, and a database stamped this way with a schema from an old, still-incremental migration
chain would silently diverge from what `a1a302e871c3` actually creates.

**Verified before this reset landed** (see the commit that introduced `a1a302e871c3` for the full
methodology):

- `alembic heads` returns exactly `a1a302e871c3 (head)`.
- `alembic upgrade head` succeeds against a genuinely empty PostgreSQL database.
- `alembic current` reports `a1a302e871c3 (head)` afterward.
- `alembic downgrade base` followed by `alembic upgrade head` succeeds again (round-trip).
- `alembic revision --autogenerate` against a database built from `a1a302e871c3` reports the exact
  same 5-object diff as the same command run against a database built from the old 9-migration
  chain — see "Known autogenerate diff" below; the reset did not change this.
- The full `tests/integration/**` and `tests/e2e/backend/**` suites (real Testcontainers Postgres)
  pass against a database created from `a1a302e871c3` alone.

## Known autogenerate diff (pre-existing, not caused by the reset)

Running `alembic revision --autogenerate` against a database created from `a1a302e871c3` will
always report 5 objects as "removed" versus current ORM metadata — this is expected, not a defect
in the baseline:

- `ix_ingestion_jobs_one_active_per_document`
- `ix_document_deletion_jobs_one_active_per_document`
- `ix_reindex_jobs_one_active_per_document`
- `uq_documents_content_hash`
- `ix_vector_cleanup_jobs_document_id`

The first three are the partial unique "at most one active job per document" indexes described in
`CLAUDE.md`'s High-Risk Invariants ("enforced by a real Postgres partial unique index, never
application logic alone") — deliberately not declared via SQLAlchemy `Index`/`__table_args__` on
their models, so `Base.metadata` (what autogenerate compares against) has never known about them.
This predates the Phase 2.10 reset: the same 5-object diff exists identically against the old
9-migration chain. If a model is ever changed to declare one of these explicitly, regenerate and
review the resulting migration rather than assuming the diff is spurious.

## Current status

`app/models/` has ORM models spanning documents, ingestion, deletion, re-indexing, vector cleanup,
and index-collection tracking — see [docs/architecture/](../docs/architecture/README.md) for the
module ownership map and [docs/document-lifecycle/](../docs/document-lifecycle/README.md) for the
lifecycle each model's job table backs. Run `alembic heads` to confirm the current single
migration head (there must always be exactly one) — see
[docs/troubleshooting/](../docs/troubleshooting/README.md) if it ever reports more than one.
