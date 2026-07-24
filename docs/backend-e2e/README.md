# Backend E2E

Full-flow testing through the real HTTP boundary. See [docs/testing/](../testing/README.md) for
where this tier sits relative to unit/integration.

## Scope

Covers the complete backend user flow: document upload (`POST /documents`) → real
`IngestionWorker` processing (extraction, chunking, embedding, Qdrant upsert) →
retrieval/orchestration → the streaming chat SSE endpoint (`POST /chat`), consumed incrementally so
event order (`metadata` → `token`(s) → `done`) is genuinely exercised, not inspected as one
buffered string. Also covers: validation errors, all four decision paths, a mid-stream LLM
failure, an ingestion failure, document read/download/deletion/re-index/reconciliation flows, and
liveness staying up when readiness fails.

**Does not cover:** frontend behavior (no frontend exists), real AI model output quality (see
[docs/multilingual/README.md#smoke test](../multilingual/README.md)), or anything requiring
`docker-compose.yml`.

## Environment requirements

- **Docker is required.** [Testcontainers for Python](https://testcontainers-python.readthedocs.io/)
  starts real, isolated, temporary Postgres and Qdrant containers on ephemeral (dynamically
  assigned) ports, with an isolated database and Qdrant collection per test.
- **`docker-compose.yml` is never used** — no fixed ports, no shared/persistent Compose volumes.
  Every container is started and torn down by Testcontainers, entirely separate from your local
  dev stack.
- **Real Postgres and real Qdrant; deterministic fake AI providers.** The FastAPI app runs for
  real behind a real ASGI HTTP client (`httpx.AsyncClient` + `ASGITransport`), with real
  extraction/chunking/decision/prompt-building/Qdrant code paths — only the embedding model and
  chat LLM are swapped for fakes via monkeypatching the provider-factory function each consuming
  module already imports, never a production-code branch on `APP_ENV`. **No real Ollama container
  runs and no model is pulled.**
- **MinIO coverage** — a subset of tests run with `FILE_STORAGE_PROVIDER=minio` against a real,
  ephemeral MinIO container (Testcontainers, dynamic port, unique bucket per test), selected purely
  through the app's real `Settings`/`create_file_storage()` dependency chain.

## Database schema (Phase 2.10 baseline)

Each test session's ephemeral Postgres container is migrated via `run_alembic_upgrade("head")`
against Alembic's single baseline revision, `a1a302e871c3` — see
[alembic/README.md](../../alembic/README.md#migration-history-reset-phase-210). This tier never
depends on any of the 9 deleted incremental revisions; a genuinely empty container migrating
straight to `a1a302e871c3` is exactly what this suite already exercises on every run, so the
history reset required no fixture changes here.

## Infrastructure startup

Handled entirely by `tests/e2e/backend/conftest.py` — no manual setup beyond having Docker
running. Key fixtures:

| Fixture | Provides |
|---|---|
| `app_client` | A real `httpx.AsyncClient` bound to the FastAPI app via `ASGITransport` |
| `e2e_session_factory` | A session factory against the ephemeral test-session Postgres |
| `process_pending_job` | Drives one real `IngestionWorker.process_next_job()` call |
| `isolated_test_state` | Guarantees test isolation (separate DB/Qdrant collection per test) |

Both `tests/integration/conftest.py` and `tests/e2e/backend/conftest.py` share
`tests/support/minio_containers.py` for the MinIO container startup routine, instead of each
duplicating the `DockerContainer` setup — the container starts lazily, only when a test actually
requests it, so existing local-storage tests never pay for it.

## Execution commands

```bash
make test-e2e-backend         # pytest -m e2e tests/e2e/backend -q (includes MinIO E2E coverage)
make verify-e2e-backend       # runs the suite plus its own checks
make test-e2e-backend-minio   # the MinIO-specific E2E test only
make verify-e2e-backend-minio
```

Never run as part of `make test`/`make verify`, and not added to the pre-commit hook (too slow,
requires Docker).

## Fixture and cleanup behavior

Every test gets its own isolated Postgres database and Qdrant collection — no state leaks between
tests. Containers and all state are removed after the test session; nothing persists between runs.
A guard in `tests/integration/conftest.py`/`tests/e2e/backend/conftest.py` prevents these fixtures
from ever pointing at a production `APP_ENV`/`DATABASE_URL`/`QDRANT_URL`.

## Failure diagnosis

- **A test hangs or times out** — almost always Docker not running, or a container failing to
  start. Check `docker ps` and Docker daemon status first — see
  [docs/troubleshooting/](../troubleshooting/README.md).
- **A test fails only in this tier, not in unit tests** — likely a real Postgres locking/HTTP
  contract difference a fake session couldn't represent; re-read the failing assertion against the
  actual SQL/HTTP behavior, not the fake's approximation of it.
- **An SSE-ordering assertion fails** — the test consumes the stream incrementally; a buffering
  issue in the route or client would surface here, not in a unit test (which never streams).

## Current Limitations

- No frontend E2E coverage — no frontend exists in this repository.
- No real-Ollama coverage in this tier — model-behavior quality is checked only by the separate,
  manual real-AI smoke suite (see [docs/multilingual/](../multilingual/README.md)).
- Requires Docker; there is no fallback execution path without it.

## Deferred Behavior

- Frontend/browser E2E tests — will be added once a frontend exists in this repository.
- CI integration of this tier — currently run manually; no CI workflow exists in this repository
  at all (no `.github/workflows/`).
