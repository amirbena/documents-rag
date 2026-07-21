# Testing

Test taxonomy, suite ownership, fixture ownership, and where a new test belongs. For backend E2E
specifics see [docs/backend-e2e/](../backend-e2e/README.md).

## Taxonomy

| Tier | Location | Marker | Needs Docker? | Real Postgres/Qdrant? | Real AI models? | Run by `make verify`? |
|---|---|---|---|---|---|---|
| **Unit** | `tests/unit/` | (default, unmarked) | No | No (fakes/mocks) | No | Yes |
| **Integration** | `tests/integration/` | `@pytest.mark.integration` | Yes | Yes (Testcontainers) | No (fake providers) | No |
| **Backend E2E** | `tests/e2e/backend/` | `@pytest.mark.e2e` | Yes | Yes (Testcontainers) | No (fake providers) | No |
| **Real-AI smoke** | `scripts/smoke_multilingual_real.py` | — (not pytest) | No | No | **Yes** (real Ollama) | No — manual/nightly only |

Frontend E2E does not exist — no frontend exists in this repository yet.

## What each tier is for

- **Unit** — fakes and mocks only (fake sessions, fake providers, mocked `httpx` transports); no
  real database, no real Qdrant, no real network, no Docker. Fast — the whole suite runs in well
  under a second.
- **Integration** — real, ephemeral Postgres/Qdrant/MinIO containers via
  [Testcontainers for Python](https://testcontainers-python.readthedocs.io/), **never** the
  repository's `docker-compose.yml`, dynamically assigned ports, no persistent volumes. Covers
  behavior a fake cannot faithfully represent: Alembic migrations against a real schema,
  `SELECT ... FOR UPDATE SKIP LOCKED` claim semantics under genuine Postgres locking, Qdrant's real
  HTTP contract.
- **Backend E2E** — exercises the complete backend flow through real HTTP
  (`httpx.AsyncClient` + `ASGITransport`) against its own ephemeral Testcontainers stack — document
  upload → real worker processing → retrieval/orchestration → streaming chat SSE, consumed
  incrementally so event order/timing is genuinely exercised. See
  [docs/backend-e2e/](../backend-e2e/README.md) for full detail.
- **Real-AI smoke** — a small, manual/nightly check against a real Ollama container with real
  models pulled, to catch drift in actual model behavior without paying that cost on every commit.

## AI-provider policy in tests

**No tier below real-AI smoke ever pulls or calls a real LLM/embedding model.** Unit tests use
hand-written fake provider doubles. The integration/E2E suites monkeypatch the provider-factory
function each consuming module already imports (`FakeEmbeddingProvider`,
`FakeStreamingLLMProvider`/`FakeFailingLLMProvider` in `tests/e2e/backend/fakes.py`,
`MultilingualFakeEmbeddingProvider` in `tests/multilingual_fixtures.py`) — never a branch on
`APP_ENV` in production code. Real Ollama stays entirely outside unit/integration/E2E.

## Unit test layout (mirrors `app/` 1:1)

```
tests/unit/
├── configuration/   # Settings/.env.example consistency
├── core/            # app.core.config
├── api/             # route-level tests (dependency-override style, fake DB session)
├── services/
│   ├── documents/   # mirrors app/services/documents/
│   ├── ingestion/   # mirrors app/services/ingestion/
│   ├── indexing/    # mirrors app/services/indexing/
│   └── reconciliation/  # mirrors app/services/reconciliation/
├── rag/             # decision/orchestrator/prompt_builder/retrieval_service
│   ├── engines/     # mirrors app/rag/engines/
│   ├── prompts/     # mirrors app/rag/prompts/
│   └── providers/   # mirrors app/rag/providers/
├── storage/         # mirrors app/storage/
└── scripts/         # tests for scripts/*.py contracts (mocked dependencies, no Docker)
```

**Rule for a new test:** if it exercises one function/class with fakes only, it goes under
`tests/unit/`, in the subdirectory mirroring the production module's own package path. If the
production file is `app/services/foo/bar.py`, its unit test is
`tests/unit/services/foo/test_bar.py`.

## Integration test layout (grouped by feature, real infrastructure per contract)

```
tests/integration/
├── conftest.py                        # ephemeral Postgres/Qdrant/MinIO fixtures,
│                                       # production-environment guard, Alembic helpers
├── test_alembic_migrations.py
├── documents/{read,download,deletion,upload}/  # one file per infrastructure contract
├── ingestion/                          # retry, stale recovery, worker (Postgres/MinIO)
├── indexing/                           # re-index build/activation, vector deletion, cleanup
└── reconciliation/                     # audit/report services against real Postgres/Qdrant
```

**Rule for a new test:** if it needs real Postgres/Qdrant/MinIO locking or HTTP-contract
behavior a fake cannot faithfully represent, it goes under `tests/integration/`, grouped by
feature directory, one file per infrastructure contract (`test_postgres.py`, `test_qdrant.py`,
`test_minio.py`, `test_concurrency.py`).

## Backend E2E layout (grouped by user-visible workflow)

```
tests/e2e/backend/
├── conftest.py                         # app_client, e2e_session_factory, process_pending_job,
│                                       # isolated_test_state fixtures
├── fakes.py                            # FakeEmbeddingProvider, FakeStreamingLLMProvider, ...
├── documents/{read,deletion,upload}/
├── ingestion/
├── indexing/
└── reconciliation/
```

**Rule for a new test:** if it exercises a complete user-visible flow through the real HTTP
boundary, it goes under `tests/e2e/backend/`, grouped by workflow (not by production module name).

## Fixture / infrastructure-helper ownership

| Helper | Location | Shared by |
|---|---|---|
| `tests/integration/conftest.py` | Ephemeral Postgres/Qdrant fixtures, production-env guard, Alembic helpers | All integration tests |
| `tests/e2e/backend/conftest.py` | `app_client`, `e2e_session_factory`, `process_pending_job`, `isolated_test_state` | All backend E2E tests |
| `tests/support/minio_containers.py` | One ephemeral-MinIO-container startup routine | Both integration and E2E conftest (started lazily, only when a test actually requests it) |
| `tests/multilingual_fixtures.py` | `MultilingualFakeEmbeddingProvider`, Hebrew/English synonym table | Multilingual unit/integration/E2E tests |
| `tests/e2e/backend/fakes.py` | Fake embedding/LLM providers | All backend E2E tests |

**Rule:** a fixture/helper used by more than one test file's feature area belongs in
`tests/support/`; a fixture used only within one tier belongs in that tier's own `conftest.py`.

## Convenience test-slice commands

Beyond the full-tier commands (`make test`/`make test-integration`/`make test-e2e-backend`), a
few cross-cutting feature slices exist as a convenience — they are not a substitute for the full
verification commands, which still cover everything including these:

```bash
make test-rag-engines            make verify-rag-engines
make test-multilingual-rag       make verify-multilingual-rag
make test-storage                make verify-storage
make test-storage-integration    make verify-storage-integration
make test-minio                  make verify-minio
make test-document-read          make verify-document-read
make test-document-read-integration  make verify-document-read-integration
make test-ingestion-retry        make verify-ingestion-retry
make test-ingestion-retry-integration  make verify-ingestion-retry-integration
make test-document-deletion      make verify-document-deletion
make test-document-deletion-integration  make verify-document-deletion-integration
```

## Full commands

```bash
make test                  # pytest -m "not integration and not e2e and not slow" -q
make test-integration       # pytest -m integration -q (needs Docker)
make test-e2e-backend       # pytest -m e2e tests/e2e/backend -q (needs Docker)
make verify                 # test + lint + typecheck + compose, stopping at first failure
make verify-integration     # test-integration plus its own checks
make verify-e2e-backend     # test-e2e-backend plus its own checks
```

## Current Limitations

- Real-provider quality (actual Ollama model output correctness) is checked only manually/nightly,
  never on every commit.
- No frontend E2E tier exists — no frontend exists in this repository.
- Integration/E2E tests require Docker; there is no way to run them without it.

## Deferred Behavior

- A CI-integrated real-AI smoke suite — deliberately kept separate/manual, not part of the default
  integration run.
- Frontend E2E tests — will be added once a frontend exists.
