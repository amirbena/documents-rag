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
- Auth, rate limiting, and a structured observability/logging pipeline.
- Kubernetes manifests, Helm charts, or any monitoring/alerting configuration that consumes
  `/health/*` — this platform only establishes the operational health *contract*.

These are pre-existing, deliberate scope boundaries — not gaps discovered during this
documentation pass. See [docs/deployment/](../deployment/README.md) for deployment-specific
deferred items and [docs/document-lifecycle/](../document-lifecycle/README.md) for lifecycle-specific
deferred behavior.
