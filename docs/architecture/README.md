# Architecture

Canonical entry point for system-level architecture: module boundaries, dependency direction,
major data flows, and ownership. For domain-specific detail, follow the links below rather than
expecting this page to repeat it.

## System overview

`documents-rag` is a local-first, framework-independent, multilingual RAG (Retrieval-Augmented
Generation) platform. It runs entirely via Docker Compose — FastAPI, PostgreSQL, Redis, Qdrant,
and Ollama (local LLM + embedding inference) — with no required external API calls or cloud
dependencies. An optional S3-compatible object store (MinIO) can replace local-disk storage.

Three storage systems have distinct, non-overlapping roles:

| System | Role |
|---|---|
| **PostgreSQL** | Lifecycle authority — `Document`/`IngestionJob`/`DocumentDeletionJob`/`VectorCleanupJob`/`ReindexJob`/`IndexCollection` rows are the source of truth for what exists and its state |
| **Object storage** (local filesystem or MinIO) | Original-content authority — the only place a document's raw uploaded bytes live |
| **Qdrant** | Rebuildable derived state — every vector can be regenerated from Postgres + object storage via re-index; never a source of truth for anything |

See [docs/storage/](../storage/README.md) for the full storage-ownership breakdown.

## High-level data flow

```
Upload (POST /documents)
   -> object storage save + Document/IngestionJob rows (PENDING)
   -> IngestionWorker (out-of-band): extract -> chunk -> embed -> Qdrant upsert -> Document marked indexed

Chat (POST /chat)
   -> RagEngine.stream_answer(question)
        -> RuleBasedRagDecider (decide: retrieval / direct / clarify / out-of-scope)
        -> RetrievalService (embed query -> Qdrant search)          [NEEDS_RETRIEVAL only]
        -> RagPromptBuilder (build labeled, attributed context)      [NEEDS_RETRIEVAL only]
        -> LLMProvider.stream_generate(...)                          [NEEDS_RETRIEVAL / DIRECT_LLM]
   -> Server-Sent Events: metadata -> token(s) -> done | error
```

Both `CustomRagEngine` (default) and `LangChainRagEngine` (optional, `RAG_ENGINE=langchain`)
produce this same event contract from the same underlying pieces — see
[docs/rag/](../rag/README.md) and [docs/langchain/](../langchain/README.md).

## Process lifecycle (Phase 2.10)

`app/main.py` registers a FastAPI `lifespan` (`app/core/lifespan.py`) around the ASGI app, and
wires middleware in this order: `CORSMiddleware` (`app/core/cors.py`) first, then
`correlation_id_middleware` (`app/core/middleware.py`) last. Starlette wraps middleware in
reverse-registration order, so correlation ID stays the **outermost** layer — every subsequent
middleware, exception handler, and log line inside a request can read
`app.core.correlation.get_correlation_id()`, and even a CORS-preflight response that
`CORSMiddleware` short-circuits internally still gets `X-Correlation-ID` echoed on the way back
out, since correlation wraps around whatever `call_next` returns.

**Startup order:** `get_settings()` (fail-fast config validation — see
[docs/configuration/](../configuration/README.md)) → `configure_logging()` → FastAPI app/
middleware/router construction → lifespan startup (`app_startup_begin` → `app_startup_complete`
structured log events — see [docs/operations/](../operations/README.md#structured-logging)) → the
ASGI server begins accepting requests. **Startup never probes PostgreSQL/Qdrant/MinIO/Redis/Ollama
reachability** — a temporarily unreachable remote dependency never prevents the process from
starting. `GET /health/ready` (see [docs/deployment/](../deployment/README.md)) remains the sole
dependency-readiness mechanism; it is checked per-request by whatever external prober polls it,
never by the process itself at boot.

**Shutdown order:** lifespan shutdown (`app_shutdown_begin`) → the shared SQLAlchemy engine
(`app/db/session.py`) is disposed via an `AsyncExitStack`, so a hypothetical future startup step
that raised after the engine was registered would still release it → `app_shutdown_complete`. No
provider client (Ollama/Qdrant/MinIO) is closed here — every provider client in this codebase is
already constructed and closed per operation (see [docs/providers/](../providers/README.md)), so
there is nothing process-lifetime-scoped for the lifespan to hold or release.

This lifespan governs only the API process. The standalone `scripts/process_pending_*.py` batch
scripts are a **separate process model** with their own SIGINT/SIGTERM handling, never connected to
this lifespan — see [docs/operations/](../operations/README.md#worker-signal-handling-phase-210).

## Error hierarchy

`app/core/errors.py` defines `AppError` and 8 category subclasses (`ConfigurationError`/
`ValidationError`/`NotFoundError`/`ConflictError`/`ProviderError`/`OperationTimeoutError`/
`LifecycleError`/`InternalError`), each carrying the HTTP status `app/core/exception_handlers.py`
maps it to. This is additive, not a replacement: every route's own outcome-table/try-except
mapping (e.g. `documents.py`'s `_RETRY_OUTCOME_ERRORS`) is checked first by FastAPI and remains the
primary, most-specific path for its own domain. `AppError`'s handlers are a **fallback net only**
— they run for a new `AppError` raised by lifespan/config/retry code, or any exception that
reaches the route boundary without already having been translated to an `HTTPException`. Both
fallback handlers preserve the existing `{"detail": "..."}` response shape, echo
`X-Correlation-ID`, and never return `str(exc)` or a stack trace; the ~7 pre-existing exception
hierarchies (storage, provider, RAG, documents/dedup, indexing/reindex, reconciliation) are not
reparented under `AppError` — reparenting would risk changing `isinstance` semantics those routes
already rely on, for no real benefit.

## Module ownership map

| Module | Owns | Never does |
|---|---|---|
| `app/api/routes/` | Unversioned operational health (`/health*`) | Business logic, DB/Qdrant/storage calls |
| `app/api/v1/routes/` | Versioned business API — parse/inject/call-one-service/copy-status | Direct DB/Qdrant/storage access, worker internals, aggregation logic |
| `app/services/documents/` | Upload, read queries, download, deletion scheduling/execution, dedup | Vector-store writes (delegates to `app/services/indexing/`) |
| `app/services/ingestion/` | Ingestion job execution, retry, stale recovery | `FileStorage`/vector-store calls outside the worker's own pipeline |
| `app/services/indexing/` | Collection lifecycle, vector deletion, cleanup jobs, re-index build/activation | Importing from `app/services/documents/` (one-way dependency, see below) |
| `app/services/reconciliation/` | Read-only cross-domain lifecycle audit | Any mutation, repair, retry, or cleanup execution |
| `app/rag/providers/` | Provider abstractions + concrete Ollama/Qdrant implementations | Construction outside `provider_factory.py` |
| `app/rag/engines/` | `RagEngine` implementations (custom, LangChain) | Diverging from the shared decision/retrieval/prompt contract |
| `app/rag/prompts/` | Language-aware prompt catalog/resolution | Per-engine prompt duplication |
| `app/storage/` | Provider-neutral object storage contract + Local/MinIO adapters | Leaking a provider SDK type or filesystem path past the adapter boundary |
| `app/models/` | SQLAlchemy ORM | Business logic |
| `app/schemas/` | Pydantic request/response models | ORM leakage into API responses |

See [docs/document-lifecycle/](../document-lifecycle/README.md) for how ingestion, deletion,
re-index, cleanup, and reconciliation modules interact across a document's full lifecycle, and
[docs/providers/](../providers/README.md) / [docs/rag/](../rag/README.md) for the provider and RAG
layers specifically.

## Dependency direction rules

- `app/api/v1/routes/*` → services only; never a direct DB/Qdrant/storage call, never a worker
  internal (workers are invoked only by scripts and tests).
- `app/services/documents/*` may call `app/services/indexing/*`'s public functions (e.g. deletion
  calls vector cleanup); `app/services/indexing/*` must **never** import from
  `app/services/documents/*` — no reverse dependency, no import cycle.
- `app/services/reconciliation/*` is a sibling package that legitimately imports from **both**
  `documents/*` and `indexing/*` (an audit spans both domains) — this does not violate the
  one-directional rule above, since it isn't `indexing/*` doing the importing.
- `app/rag/engines/*` resolve providers only via `app/rag/providers/provider_factory.py` — never
  construct an Ollama/OpenAI/Gemini/Anthropic/Qdrant client directly.
- Provider SDK types (MinIO/Qdrant/httpx-specific exceptions or response objects) never leave the
  adapter that wraps them — translated to `app.storage.errors`/internal exception types at the
  boundary.
- `app/api/routes/health.py` (unversioned) and `app/api/v1/routes/*` (versioned) are independent —
  a business API version bump never moves health under `/api/vN`.

Package `__init__.py` files stay minimal (a one-line docstring) — they never re-export package
contents. Import from the canonical module directly.

## Architectural invariants

- Ingestion, deletion, and re-index jobs are **append-only**: a `FAILED`/`PARTIALLY_FAILED` row is
  never reset or deleted; retry always inserts a new row. At most one active
  (`PENDING`/`PROCESSING`) job of each kind may exist per document, enforced by a **Postgres
  partial unique index**, never application logic alone.
- Full document deletion always calls the tracked (not partial) vector-deletion path — a partial
  vector-cleanup failure blocks storage deletion in that attempt.
- No operation's partial success is ever exposed as full success — typed result objects
  (`VectorDeletionResult`, `ReindexBuildResult`, `ReindexActivationResult`,
  `DeletionRequestResult`) are inspected by the caller, never collapsed to a bare bool.
- Qdrant is rebuildable: a Qdrant write can succeed while the Postgres commit fails, and this is
  documented, not glossed over — the Qdrant/Postgres boundary is never treated as atomic.
- Retry and re-index both reuse already-stored source content — never re-upload, never re-save.
  Point IDs are deterministic per document/chunk, so re-upserts overwrite rather than duplicate.
- The current searchable index (a document's active Qdrant collection) remains in place until a
  replacement index write has already succeeded — never delete-then-write.
- Reconciliation is strictly read-only, by design — see [docs/operations/](../operations/README.md)
  for how repair actually happens (through separate, bounded, domain-specific commands, never a
  generic repair engine).

Full detail on each lifecycle: [docs/document-lifecycle/](../document-lifecycle/README.md).

## Deferred Behavior

- A standalone public retrieval endpoint (retrieval is reachable only indirectly via `POST /chat`).
- Conversation memory / multi-turn context.
- An LLM-based (as opposed to rule-based) question router.
- Auth and rate limiting. Structured JSON logging and correlation IDs are implemented (Phase
  2.10) — see [docs/operations/](../operations/README.md#structured-logging) — but no metrics/
  tracing platform (Prometheus, OpenTelemetry, etc.) exists.
- Kubernetes manifests, Helm charts, or any monitoring/alerting configuration that consumes
  `/health/*` — this platform only establishes the operational health *contract*.

These are pre-existing, deliberate scope boundaries — not gaps discovered during this
documentation pass. See [docs/deployment/](../deployment/README.md) for deployment-specific
deferred items and [docs/document-lifecycle/](../document-lifecycle/README.md) for lifecycle-specific
deferred behavior.
