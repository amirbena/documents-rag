# CLAUDE.md

Operational guide for Claude Code in this repository. Read this before making changes. For system
design/rationale see [ARCHITECTURE.md](ARCHITECTURE.md); for run/test instructions see
[README.md](README.md). This file favors tables and short rules over narrative — it should let a
session answer quickly: where does this code go, which module owns it, which dependency direction
is legal, which invariant must not break, which command verifies it.

## 1. Repository Purpose

`documents-rag` is a framework-independent, multilingual RAG (Retrieval-Augmented Generation)
platform, built incrementally as a portfolio/learning project. Favor clarity and correctness over
speed or cleverness; ship small, verifiable milestones rather than large bundled changes.

- **PostgreSQL** is the lifecycle authority — `Document`/`IngestionJob`/`DocumentDeletionJob`/
  `VectorCleanupJob`/`IndexCollection` rows are the source of truth for what exists and its state.
- **Object storage** (local filesystem or MinIO, behind `FileStorage`) is the original-content
  authority — the only place a document's raw uploaded bytes live.
- **Qdrant** is rebuildable derived state — every vector can be regenerated from Postgres +
  object storage via re-index; Qdrant is never a source of truth for anything.
- **Custom and LangChain RAG engines** share one provider factory, one `PromptProvider`, and one
  `RetrievalService`/`RuleBasedRagDecider` — an engine adapts these, it never reimplements them.

## 2. Current Package Map

```
app/
├── api/
│   ├── routes/health.py          # unversioned operational health (see Dependency Rules)
│   └── v1/routes/                # versioned business API: chat.py, documents.py, providers.py
├── services/
│   ├── documents/                # document lifecycle: upload, query/download, chunk/extract, deletion
│   ├── ingestion/                 # ingestion job execution, retry, stale recovery
│   ├── indexing/                  # collection lifecycle, vector deletion, cleanup jobs, re-index
│   ├── platform_health.py         # unversioned health/readiness aggregation
│   └── ollama_client.py
├── rag/                            # engines, providers, prompts, decision, retrieval
├── storage/                        # FileStorage contract + Local/MinIO adapters, keys, errors
├── models/                         # SQLAlchemy ORM
└── schemas/                        # Pydantic request/response models

tests/
├── unit/                            # every unit test lives here — nothing under tests/*.py directly
│   ├── configuration/, core/, api/   # Settings/.env consistency, app.core, route-level tests
│   ├── services/{documents,ingestion,indexing}/  # mirrors app/services/ 1:1 per production module
│   ├── rag/, rag/{engines,prompts,providers}/    # mirrors app/rag/'s own subpackage split
│   ├── storage/                      # mirrors app/storage/
│   └── scripts/                      # tests for scripts/*.py contracts
├── integration/{documents,ingestion,indexing,storage,...}/  # feature + infrastructure contract
├── e2e/backend/{documents,ingestion,chat,multilingual,health}/  # user-visible workflow
└── support/{documents,ingestion,indexing,storage}/  # feature-owned fakes/builders (≥2 consumers only)
```

Package init files stay minimal (a one-line docstring) — they never re-export the package's
contents. Import from the canonical module directly:
`from app.services.documents.query_service import get_document`, never
`from app.services.documents import get_document`.

## 3. Dependency Rules

- `app/api/v1/routes/*` → services only; never a direct DB/Qdrant/storage call, never a worker
  internal (`IngestionWorker`/`DocumentDeletionWorker` are invoked only by scripts and tests).
- `app/services/documents/*` may call `app/services/indexing/*`'s public functions when needed
  (e.g. deletion calls vector cleanup); `app/services/indexing/*` must never import from
  `app/services/documents/*` (no reverse dependency, no import cycle).
- Within `app/services/documents/`: `download_service.py` may reuse `query_service.get_document()`
  (one-way); `query_service.py` never calls `FileStorage` and never imports `download_service.py`.
  `deletion_worker.py` may import from `deletion_service.py`; never the reverse.
- Within `app/services/ingestion/`: `retry_service.py` and `stale_recovery_service.py` both import
  shared constants/helpers from `status.py`; neither imports the other. The HTTP route
  (`documents.py`) imports only `retry_service`, never `stale_recovery_service` (no HTTP endpoint
  triggers stale recovery — see High-Risk Invariants).
- `app/rag/engines/*` resolve providers only via `app/rag/providers/provider_factory.py` — never
  construct an Ollama/OpenAI/Gemini/Anthropic/Qdrant client directly.
- Provider SDK types (MinIO/Qdrant/httpx-specific exceptions or response objects) never leave the
  adapter that wraps them — translated to `app.storage.errors`/internal exception types at the
  boundary.
- `app/api/routes/health.py` (unversioned) and `app/api/v1/routes/*` (versioned) are independent;
  a business API version bump never moves health under `/api/vN`.

## 4. Canonical File Ownership

| Concern | Canonical location |
|---|---|
| Document upload | `app/services/documents/upload_service.py` |
| Document reads (list/detail/status/failure) | `app/services/documents/query_service.py` |
| Original-content download | `app/services/documents/download_service.py` |
| Text extraction | `app/services/documents/text_extractor.py` |
| Chunking | `app/services/documents/chunker.py` |
| Deletion scheduling | `app/services/documents/deletion_service.py` |
| Deletion execution | `app/services/documents/deletion_worker.py` |
| Ingestion job execution | `app/services/ingestion/worker.py` |
| Ingestion retry | `app/services/ingestion/retry_service.py` |
| Stale ingestion recovery | `app/services/ingestion/stale_recovery_service.py` |
| Shared retry/recovery constants | `app/services/ingestion/status.py` |
| Collection lifecycle | `app/services/indexing/collection_registry.py` |
| Vector deletion (partial + full) | `app/services/indexing/vector_deletion_service.py` |
| Legacy-vector cleanup jobs | `app/services/indexing/cleanup_job_service.py` |
| Re-index | `app/services/indexing/reindex_service.py` |
| Operational health | `app/api/routes/health.py` + `app/services/platform_health.py` |

## 5. Test Ownership

- **Unit** (`tests/unit/**` — nothing lives directly under `tests/*.py` anymore) — mirrors
  production modules 1:1, organized by the same top-level areas as `app/`: `configuration/`,
  `core/`, `api/` (route-level tests), `services/{documents,ingestion,indexing}/`,
  `rag/`, `rag/{engines,prompts,providers}/`, `storage/`, `scripts/`. Fakes/mocks only, no Docker.
- **Integration** (`tests/integration/**`) — grouped by feature directory
  (`documents/{read,download,deletion}/`, `ingestion/`, `indexing/`) with one file per
  infrastructure contract inside (`test_postgres.py`, `test_minio.py`, `test_concurrency.py`,
  etc.). Real Testcontainers Postgres/Qdrant/MinIO — never SQLite, never a fixed port, never
  `docker-compose.yml`.
- **Backend E2E** (`tests/e2e/backend/**`) — grouped by user-visible workflow, not production
  module names. Real HTTP boundary (`httpx.ASGITransport`), real Postgres/Qdrant, fake AI
  providers only.
- Real Ollama and real embedding-model downloads never appear in `make test`/`test-integration`/
  `test-e2e-backend` — they belong only to `make smoke-multilingual-real`.

Canonical commands:

```
make test                      # fast unit suite
make test-integration          # full integration suite (Docker)
make test-e2e-backend          # full backend E2E suite (Docker)
make test-document-deletion(-integration)
make test-ingestion-retry(-integration)
make verify                    # test + lint + typecheck + compose — run before finishing any task
```

## 6. High-Risk Invariants

- One active (`PENDING`/`PROCESSING`) `IngestionJob` per document, and one active
  (`PENDING`/`PROCESSING`) `DocumentDeletionJob` per document — both enforced by a real Postgres
  partial unique index, never application logic alone.
- Ingestion and deletion attempts are append-only: a `FAILED`/`PARTIALLY_FAILED` row is never
  reset or deleted; retry always inserts a new row. The one exception is a stale `PROCESSING` row,
  which retry/recovery flip to `FAILED` in the same commit as the replacement — never any other
  post-creation transition.
- Full document deletion always calls `delete_all_tracked_document_vectors(..., session)`, never
  `delete_current_document_vectors()` (deliberately partial, active-collection-only). Vectors are
  always removed before storage; a partial vector failure blocks storage deletion in that attempt.
- No operation's partial success is ever exposed as full success — `VectorDeletionResult`/
  `ReindexResult`/`DeletionRequestResult` are typed results the caller must inspect, never a bare
  bool swallowing a partial-failure case.
- Qdrant is rebuildable: never treat a Qdrant write as authoritative, never make the
  Qdrant/Postgres boundary look atomic — a Qdrant write can succeed while the Postgres commit
  fails, and this is documented, not glossed over.
- Retry and re-index both reuse already-stored source content (`FileStorage`) — never re-upload,
  never re-save. Point IDs are deterministic per document/chunk, so re-upserts overwrite rather
  than duplicate.
- The current searchable index (a document's active Qdrant collection) remains in place until a
  replacement index write has already succeeded — never delete-then-write.
- A deleted document blocks ingestion retry (`409`) and returns `410 Gone` (never `404`) on
  download — the Postgres row still exists, only its content was intentionally removed.
- `GET /health` and `/health/live` never perform dependency I/O; only `/health/ready` and
  `/health/dependencies` do. No `/health*` response ever includes a raw exception, connection
  string, or credential.
- Stale-job recovery (`recover_stale_ingestion_jobs()`) has no HTTP endpoint and is never invoked
  by `make verify`/`make test*`/CI — only `scripts/recover_stale_ingestion_jobs.py`. There is no
  equivalent stale-deletion-job recovery in this codebase at all.

## 7. Style Rules (condensed)

- **Providers**: prefer calling a provider's HTTP API directly over its SDK (`httpx`, not
  `qdrant-client`/`ollama`). Future provider stubs (`OpenAIProvider`, etc.) never call external
  APIs — every method raises `ProviderNotImplementedError` until implemented. A configured
  non-Ollama provider must resolve to that provider or raise — never silently fall back to Ollama.
- **Provider vs. model config**: `LLM_PROVIDER` (backend) and `LLM_MODEL` (model) stay separate
  settings; `LLM_MODEL` falls back to `OLLAMA_CHAT_MODEL` for `.env` compatibility. Embedding
  model is fixed (`OLLAMA_EMBEDDING_MODEL`), never user-selectable — changing it needs a
  deliberate migration (see `EMBEDDING_VERSION`/`CHUNKING_VERSION`), not a config flag.
- **RAG engines**: `RAG_ENGINE` defaults to `custom`; an alternative engine is adapter-based
  (wraps `RetrievalService`/`PromptProvider`/provider factory, never reimplements them), never
  re-embeds or creates a new collection, and preserves the `POST /api/v1/chat` SSE contract
  exactly. An unsupported `RAG_ENGINE` value raises `UnsupportedRagEngineError`. LangGraph is not
  introduced until a real agentic (multi-step, conditional) workflow needs it.
- **Multilingual**: one shared `PromptCatalog`/`PromptProvider` for both engines and both
  languages — no engine-specific or language-specific catalog. `RuleBasedRagDecider` is the one
  decision service. Real Ollama/embedding models stay out of unit/integration/E2E suites
  (`MultilingualFakeEmbeddingProvider` only); real-model checks belong to
  `make smoke-multilingual-real`, never an automated gate.
- **Route layer**: routes parse/inject/call-one-service/copy-status — no business logic,
  aggregation, or direct provider/DB call in a route module. A service needing to return both a
  body and an HTTP status uses a typed result object (`response` + `status_code`), never
  route-side status derivation.
- **Docstrings**: every module gets a one/two-line docstring; every public function/class gets a
  one-line docstring stating intent, not implementation. Skip only for trivial private helpers.

## 8. Current Scope and Explicit Non-Goals

This repository's structural refactor (the `app/services/{documents,ingestion,indexing}/` split
and removal of the duplicate versioned health route) changes structure and intentionally removes
`GET /api/v1/health` — it adds no new behavior otherwise. Do not, without an explicit user
request, add any of: SHA-256/upload deduplication, orphan-vector reconciliation, version-aware
re-indexing beyond what `reindex_service.py` already does, stale deletion-job recovery, new
lifecycle statuses, new database migrations/schema changes, new provider/prompt/multilingual/
storage behavior, a `documents/` package migration beyond what's landed (query/upload/download/
chunk/extract/deletion), an ingestion/indexing package split beyond what's landed, or a new
abstraction layer (`repositories/`, `domain/`, `application/`) for a hypothetical future need.

## Quality Gates

Run `make verify` before finishing any implementation task and before pushing/opening a PR — it
runs, in order, stopping at the first failure: `make test` → `make lint` (`ruff check .`) →
`make typecheck` (`mypy app`) → `make compose` (`docker compose config`). If `make` isn't
available, run the four commands individually in that order. Never bypass a gate (`--no-verify`
or otherwise) unless the user explicitly asks. Installing the pre-commit hook
(`./scripts/install-git-hooks.sh`) runs `make verify` automatically on every commit, but does not
replace running it explicitly before considering a task done.

## Pull Request Workflow

- Verify `gh --version` / `gh auth status` before any GitHub operation; stop and report if either
  fails.
- Confirm `git branch --show-current` is the intended feature branch before pushing — never push
  from `main`.
- Review `git status`/diff before committing; stage only files that belong to the change; never
  push unrelated files.
- Prefer small, focused PRs — one milestone or concern per PR.
- Use `.github/pull_request_template.md`'s sections (Summary, Why, Changes, Verification,
  Explicit exclusions, Next recommended milestone) as the PR body structure; write it to a file
  and pass via `gh pr create --body-file`, not an ad-hoc inline `--body`.
- PR title: short, imperative, present tense, no trailing period.

## Final Report Format

At the end of any non-trivial change, report: **What changed**, **Why it changed**, **Files
changed**, **Verification** (exact commands run and real results — never a claim without output),
**Next recommended milestone**.

## Boundaries

- Do not implement new RAG business logic (ingestion, embeddings, retrieval, chat behavior)
  unless explicitly asked.
- Do not introduce new frameworks or heavy dependencies (e.g. LangChain) without being asked.
- Do not weaken or bypass quality gates to "get it green."
