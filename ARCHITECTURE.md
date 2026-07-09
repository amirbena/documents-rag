# Architecture

## System overview

`documents-rag` is a local-first RAG (Retrieval-Augmented Generation) platform. Everything runs on
the user's machine via Docker Compose, with Ollama providing local LLM and embedding inference —
no external API calls or cloud dependencies.

This milestone is infrastructure plus eight vertical slices: a FastAPI app wired to Postgres,
Redis, Qdrant, and Ollama, with a health endpoint, an Ollama health/model-availability check, a
concrete Ollama-backed embedding provider, a concrete streaming Ollama-backed LLM provider, a
concrete Qdrant-backed vector store, a rule-based RAG decision layer, a document upload +
ingestion job skeleton, an async ingestion worker that claims and resolves those jobs, and a
document text extractor (`.txt`/`.md`/`.pdf`) that is now the worker's real first processing
step. No chunking, embedding generation, Qdrant upsert, or chat endpoint happens yet — a claimed
job's text is extracted and then discarded (nothing persists or chunks it yet); the job still
resolves to `completed` purely based on whether extraction succeeded.

## Services

| Service    | Image                    | Purpose (current)                                             | Purpose (future) |
|------------|--------------------------|------------------------------------------------------------------|-------------------|
| `app`      | built from `Dockerfile`  | FastAPI process: `/api/v1/health`, `/api/v1/providers/ollama/health`, `POST /api/v1/documents` | RAG API: ingestion processing, retrieval, chat |
| `postgres` | `postgres:16-alpine`     | Stores `documents`/`ingestion_jobs` rows via async SQLAlchemy     | Session/metadata storage |
| `redis`    | `redis:7-alpine`         | Available on the network                                         | Caching, task queues |
| `qdrant`   | `qdrant/qdrant:latest`   | Collection create/upsert/search via `QdrantVectorStore`           | Backing document retrieval in a future RAG flow |
| `ollama`   | `ollama/ollama:latest`   | Health/model checks + embeddings (`nomic-embed-text`) + streaming generation (`llama3.1`) | Backing a future public chat endpoint |

The app queries Ollama's `/api/tags` endpoint (via `app/services/ollama_client.py`) to check
reachability and whether the configured models are pulled, calls `/api/embeddings` (via
`app/rag/providers/ollama_embedding_provider.py`) to embed text with `OLLAMA_EMBEDDING_MODEL`,
and calls `/api/generate` with `stream=true` (via `app/rag/providers/ollama_llm_provider.py`) to
stream completions from the configured chat model (`LLM_MODEL`, falling back to
`OLLAMA_CHAT_MODEL` — see "LLM provider vs. model" below). The LLM provider is internal-only —
there is no public chat or SSE endpoint yet. The app also talks to Qdrant's HTTP API under
`QDRANT_URL` (via `app/rag/providers/qdrant_vector_store.py`) to create collections, upsert
vectors, and run similarity search — see "Vector store" below. Callers resolve all of these
providers through `app/rag/providers/provider_factory.py` rather than importing Ollama/Qdrant
classes directly — see "Provider factory" below. `POST /api/v1/documents` stores an uploaded
file and creates `Document`/`IngestionJob` rows in Postgres — see "Document upload and ingestion
job skeleton" below. Nothing yet processes an `IngestionJob`; Redis is still unused beyond
connection configuration.

## Provider factory

`app/rag/providers/provider_factory.py` resolves which concrete provider class to construct,
based on three config variables, so the rest of the codebase depends on the `EmbeddingProvider` /
`LLMProvider` / `VectorStore` interfaces rather than being coupled to Ollama or Qdrant directly:

- `get_embedding_provider()` — `EMBEDDING_PROVIDER` (`"ollama"` → `OllamaEmbeddingProvider`)
- `get_llm_provider()` — `LLM_PROVIDER` (`"ollama"` → `OllamaLLMProvider`; `"openai"`, `"gemini"`,
  `"anthropic"` are recognized but raise `ProviderNotImplementedError` — see "Future LLM provider
  stubs" below)
- `get_vector_store()` — `VECTOR_STORE_PROVIDER` (`"qdrant"` → `QdrantVectorStore`)

An unrecognized provider name raises `UnsupportedProviderError` (a `ValueError`) with a message
naming the offending value and the supported provider(s). All Ollama-specific logic (HTTP calls,
error handling) stays inside the Ollama provider classes — the factory only selects and
constructs; it never reimplements provider behavior, and business/service code should resolve
providers through it rather than importing `OllamaEmbeddingProvider`/`OllamaLLMProvider` directly.
The factory never falls back to Ollama for a misconfigured or unimplemented provider — every
non-`ollama` value either resolves to its own explicit failure or a real alternative
implementation.

## LLM provider vs. model

`LLM_PROVIDER` (which backend to use, e.g. `ollama`) and `LLM_MODEL` (which model that backend
should use, e.g. `llama3.1`) are deliberately separate settings — changing the model doesn't
require touching provider selection, and vice versa. `Settings.resolved_llm_model`
(`app/core/config.py`) is the single place that decides the effective model: it returns
`LLM_MODEL` if set, otherwise falls back to `OLLAMA_CHAT_MODEL` for backward compatibility.
`OllamaLLMProvider` calls `resolved_llm_model`, never `ollama_chat_model` directly, when building
its `/api/generate` request.

`OLLAMA_EMBEDDING_MODEL` is intentionally **not** part of this model-selection mechanism —
embeddings use a fixed model, independent of `LLM_MODEL`, since swapping the embedding model
would silently invalidate any previously-computed vectors. `OllamaEmbeddingProvider` always reads
`ollama_embedding_model` directly.

## Vector store

`app/rag/providers/vector_store.py` defines the abstract `VectorStore` contract plus its shared
data types:

- `VectorPoint` — one embedding vector to upsert, with `id`, `vector`, and payload metadata
  (`document_id`, `chunk_id`, `text`, `source`, optional `page_number`).
- `VectorSearchResult` — one nearest-neighbor match: `id`, `score`, plus the same payload fields.

`VectorStore` methods: `create_collection_if_not_exists(collection_name, vector_size)`,
`upsert_vectors(collection_name, points)`, `search_similar(collection_name, query_vector, limit)`.

`QdrantVectorStore` (`app/rag/providers/qdrant_vector_store.py`) is the concrete implementation.
It talks to Qdrant's REST API directly under `QDRANT_URL` via async `httpx` — **no official
Qdrant SDK is used**, matching the same pattern as the Ollama providers. It calls:

- `GET /collections/{name}` + `PUT /collections/{name}` — check-then-create a collection
  (skips creation if the collection already exists).
- `PUT /collections/{name}/points?wait=true` — upsert points (id, vector, payload).
- `POST /collections/{name}/points/search` — similarity search, returning parsed
  `VectorSearchResult` objects.

`QdrantVectorStoreError` is raised on an unreachable server, a non-200 response, or a malformed
response (e.g. missing payload fields). This provider is internal-only: no document
ingestion/upload, no chat/SSE endpoint, and no caller wires it into a retrieval pipeline yet.

## RAG decision layer

`app/rag/decision.py` is a small internal decision/orchestration layer that classifies a user
question *before* any retrieval or generation happens — it does not itself perform retrieval,
generation, ingestion, or document upload, and is not wired to any public API endpoint.

- `RagDecision` (a `StrEnum`) — one of `NEEDS_RETRIEVAL`, `DIRECT_LLM`,
  `CLARIFICATION_NEEDED`, `OUT_OF_SCOPE`.
- `DecisionResult` — a dataclass with `decision`, `reason`, and an optional `confidence`.
- `RuleBasedRagDecider.decide(question) -> DecisionResult` — deterministic, keyword/pattern-based
  routing. **No LLM call is made to route** — rules are checked in order:
  1. Empty or very short question → `CLARIFICATION_NEEDED`.
  2. Sensitive/private data extraction requests (SSN, passwords, API keys, credentials, etc.) →
     `OUT_OF_SCOPE` — checked *before* the retrieval keywords, so a request that mentions both
     documents and sensitive data is still rejected.
  3. Question references uploaded/indexed documents (`document`, `uploaded`, `pdf`, `knowledge
     base`, etc.) → `NEEDS_RETRIEVAL`.
  4. Otherwise → `DIRECT_LLM` (general question, no document reference, nothing sensitive).

This is deliberately the simplest possible decider — a future milestone may replace or augment it
with an LLM-based router, but the rule-based version exists first so the decision *contract*
(`RagDecision`/`DecisionResult`) is fixed and testable before anything calls out to a model for
routing.

## Document upload and ingestion job skeleton

`POST /api/v1/documents` (`app/api/v1/routes/documents.py`) is the first public endpoint that
touches the database. It accepts a multipart file upload and does exactly three things, all
inside one request:

1. Saves the file via `LocalFileStorage` (`app/services/local_file_storage.py`) under
   `storage/documents/`, using a **generated, filesystem-safe stored filename** (a UUID plus a
   sanitized extension) — never the raw original filename, which may contain Unicode, spaces, or
   path-unsafe characters. No S3 or other remote backend exists yet.
2. Inserts a `Document` row (`app/models/document.py`): `original_filename` (stored exactly as
   received — Hebrew or any other Unicode text is preserved verbatim, since Postgres/SQLAlchemy
   `String` columns are UTF-8 natively), `stored_filename`, `content_type`, `file_size`,
   `stored_path`.
3. Inserts an `IngestionJob` row (`app/models/ingestion_job.py`) with `status=PENDING`,
   referencing the `Document` via `document_id`. `IngestionStatus` (a `StrEnum`): `PENDING`,
   `PROCESSING`, `COMPLETED`, `FAILED` — stored in Postgres as their lowercase `.value`
   (`pending`, `processing`, ...), not the enum member name.

The endpoint returns `202 Accepted` with `{document_id, job_id, status}` — **it does not parse,
chunk, embed, or index the document inside the request.** An empty (zero-byte) upload is
rejected with `400` before any row is created. `IngestionWorker` (below) is what eventually
picks up and resolves the `pending` job it creates.

## Ingestion worker

`IngestionWorker` (`app/services/ingestion_worker.py`) is an internal service — **no public API**
— that claims and resolves one pending `IngestionJob` at a time via `process_next_job(session)`:

1. **Claim**: `SELECT ... WHERE status='pending' ORDER BY created_at LIMIT 1 FOR UPDATE SKIP
   LOCKED` — Postgres row-level locking so multiple worker instances never claim the same job.
   Returns `None` if there is no pending job.
2. Flips the claimed job to `PROCESSING` and **commits immediately** — a separate transaction
   boundary from the outcome below, so the claim is durable before any processing is attempted.
3. Looks up the associated `Document` and calls the injected processing step (default:
   `_default_process_document`, which calls `DocumentTextExtractor` — see "Document text
   extraction" below) with `(document, job)`.
4. On success: `PROCESSING` → `COMPLETED`, committed.
   On any exception: `PROCESSING` → `FAILED`, `error_message` set to `str(exception)`, committed.

The processing step is injected via the constructor (`IngestionWorker(process_document=...)`)
specifically so a **later milestone can extend it** with chunking, embedding generation, and
Qdrant upsert — without changing the claim/lock/transition logic. `IngestionWorker` never
imports or calls `EmbeddingProvider`, `LLMProvider`, `VectorStore`, or `provider_factory` itself.

**Idempotent by construction**: once a job leaves `PENDING` (to `PROCESSING`, then `COMPLETED`
or `FAILED`), the claim query's `WHERE status='pending'` filter can never select it again —
calling `process_next_job()` repeatedly does not require any separate "already processed" check.

`with_for_update(skip_locked=True)` is Postgres-specific row-locking syntax; SQLite does not
represent it correctly even if it accepts the same SQLAlchemy call, so this project deliberately
does not add SQLite/`aiosqlite` for testing this worker. Its tests use a fake `AsyncSession`
double that faithfully simulates the pending-job filter and `Document` lookup instead.

## Document text extraction

`DocumentTextExtractor` (`app/services/document_text_extractor.py`) is the ingestion worker's
default processing step: it loads a `Document`'s `stored_path` file and extracts its raw text —
**no chunking, embedding, or Qdrant upsert**. Supports exactly three file types by extension:

- `.txt` / `.md` — read as UTF-8 text and returned as a single `ExtractedPage` with
  `page_number=None`. Hebrew and other non-ASCII Unicode content is preserved exactly.
- `.pdf` — extracted page by page via `pypdf` (`PdfReader`), producing one `ExtractedPage` per
  page with a 1-indexed `page_number`, so downstream chunking/citation can reference the
  original page a piece of text came from.

Data types:

- `ExtractedPage` — `text: str`, `page_number: int | None`.
- `ExtractedDocument` — `document_id: str`, `pages: list[ExtractedPage]`, plus a `full_text`
  property that joins all pages' text.

`extract(document)` runs the actual file I/O and PDF parsing off the event loop via
`asyncio.to_thread` (both are blocking operations). `DocumentTextExtractionError` is raised for:
a missing `stored_path` file, an unsupported extension, or a file whose extracted text is empty
or whitespace-only. Any of these propagate up through `IngestionWorker.process_next_job()` and
resolve the job to `failed` with the error message stored — extraction never crashes the worker
process itself.

The extracted result is currently discarded after extraction succeeds — there is no table or
field yet that persists extracted text, since chunking/embedding (the next milestones) will
decide how it's actually stored.

## Future LLM provider stubs

`OpenAIProvider`, `GeminiProvider`, and `AnthropicProvider`
(`app/rag/providers/{openai,gemini,anthropic}_provider.py`) are explicit placeholders for
providers with no real implementation yet. Each implements `LLMProvider` via a shared base,
`LLMProviderStub` (`app/rag/providers/llm_provider_stub.py`), whose `generate()` always raises
`ProviderNotImplementedError` (`app/rag/providers/errors.py`) with a message naming the provider
— they make no HTTP calls and read no external API keys. `get_llm_provider()` raises the same
error immediately when `LLM_PROVIDER` names one of these, before any provider object does
anything, so a misconfigured provider fails loudly at resolution time rather than silently
falling back to Ollama or failing later on first use.

Adding a new stub for another future provider means: create a class inheriting
`LLMProviderStub` with its own `NOT_IMPLEMENTED_MESSAGE`, and add it to `_LLM_STUBS` in
`provider_factory.py`. See [CLAUDE.md](CLAUDE.md) for the standing rule on how stubs must behave.

## Docker Compose topology

All services join the default Compose network (`documents-rag_default`) and address each other by
service name (Docker's embedded DNS). The app reaches its dependencies at:

- `postgres:5432`
- `redis:6379`
- `qdrant:6333` (HTTP)
- `ollama:11434` (HTTP)

Only the ports needed for host-side debugging are published (`8000`, `5432`, `6379`, `6333`,
`11434`). In a production deployment, only `app`'s port would typically be exposed.

```
host:8000 ──► app ──► postgres:5432
                 ├──► redis:6379
                 ├──► qdrant:6333
                 └──► ollama:11434
```

## Environment variables

Set via `docker-compose.yml` for containers, or `.env` (copy from `.env.example`) for local runs
outside Docker. `app/core/config.py` (`Settings`) is the single source of truth for defaults.

| Variable                  | Default                                                              | Notes |
|----------------------------|-----------------------------------------------------------------------|-------|
| `APP_ENV`                 | `local`                                                                | Echoed by `/health` |
| `LOG_LEVEL`                | `INFO`                                                                 | Not yet wired to a logger |
| `DATABASE_URL`             | `postgresql+asyncpg://postgres:postgres@postgres:5432/rag_db`         | Async SQLAlchemy engine; stores `documents`/`ingestion_jobs` |
| `REDIS_URL`                | `redis://redis:6379/0`                                                | Not yet consumed |
| `QDRANT_URL`               | `http://qdrant:6333`                                                  | Used by `QdrantVectorStore` for collection/upsert/search |
| `OLLAMA_BASE_URL`          | `http://ollama:11434`                                                 | Used by `OllamaClient` for health/model checks |
| `OLLAMA_CHAT_MODEL`        | `llama3.1`                                                             | Checked for availability; backward-compatible fallback for `LLM_MODEL` if unset |
| `OLLAMA_EMBEDDING_MODEL`   | `nomic-embed-text`                                                     | Checked for availability; always used by `OllamaEmbeddingProvider` — fixed, not selectable via `LLM_MODEL` |
| `LLM_PROVIDER`             | `ollama`                                                               | Selects the `LLMProvider` implementation; `openai`/`gemini`/`anthropic` are recognized stubs |
| `LLM_MODEL`                | *(unset)*                                                              | Selects the model `OllamaLLMProvider` uses; falls back to `OLLAMA_CHAT_MODEL` if unset (see "LLM provider vs. model") |
| `EMBEDDING_PROVIDER`       | `ollama`                                                               | Selects the `EmbeddingProvider` implementation via the provider factory |
| `VECTOR_STORE_PROVIDER`    | `qdrant`                                                               | Selects the `VectorStore` implementation via the provider factory |

## Current boundaries

- `app/api` — FastAPI routers: `/health`, `/providers/ollama/health`, and `POST /documents`
  (see "Document upload and ingestion job skeleton" above).
- `app/core` — configuration and cross-cutting concerns.
- `app/db` — SQLAlchemy async engine/session setup.
- `app/models` — ORM models: `Document`, `IngestionJob`/`IngestionStatus` (see "Document upload
  and ingestion job skeleton" above).
- `app/schemas` — Pydantic request/response schemas.
- `app/services` — business logic layer: `OllamaClient` (`app/services/ollama_client.py`), a thin
  async HTTP client scoped strictly to reachability and model-availability checks — it
  intentionally does not call generation or embedding endpoints; `LocalFileStorage`
  (`app/services/local_file_storage.py`), which saves uploaded files to local disk under a
  generated safe filename (see "Document upload and ingestion job skeleton" above); and
  `IngestionWorker` (`app/services/ingestion_worker.py`), which claims and resolves pending
  ingestion jobs (see "Ingestion worker" above) — no public API, no provider calls; and
  `DocumentTextExtractor` (`app/services/document_text_extractor.py`), which extracts text from
  a document's stored `.txt`/`.md`/`.pdf` file (see "Document text extraction" above) — no
  chunking, embedding, or Qdrant upsert.
- `app/rag/providers` — abstract interfaces for embedding, LLM, and vector store providers, a
  `provider_factory.py` that resolves the configured implementation for each (see "Provider
  factory" above), and three concrete implementations:
  - `OllamaEmbeddingProvider` (`app/rag/providers/ollama_embedding_provider.py`) — calls
    `POST /api/embeddings` for `OLLAMA_EMBEDDING_MODEL` only.
  - `OllamaLLMProvider` (`app/rag/providers/ollama_llm_provider.py`) — calls
    `POST /api/generate` with `stream=true` for `Settings.resolved_llm_model`
    (`LLM_MODEL`, falling back to `OLLAMA_CHAT_MODEL`), exposing
    `stream_generate(prompt) -> AsyncIterator[str]` (yields text chunks as Ollama streams them)
    and `generate(prompt) -> str` (joins the streamed chunks, satisfying the abstract
    `LLMProvider` contract). Internal-only — no ingestion, no Qdrant writes, no public chat/SSE
    endpoint.
  - `QdrantVectorStore` (`app/rag/providers/qdrant_vector_store.py`) — calls Qdrant's HTTP API
    for collection create/upsert/search only (see "Vector store" above). Internal-only — no
    ingestion, no document upload, no chat/SSE endpoint, no full RAG flow.

  Three future-provider stubs also exist — `OpenAIProvider`, `GeminiProvider`,
  `AnthropicProvider` (`app/rag/providers/{openai,gemini,anthropic}_provider.py`) — which
  implement `LLMProvider` but always raise `ProviderNotImplementedError` (see "Future LLM
  provider stubs" above).

  `OllamaClient` (health checks) is deliberately kept separate from these provider interfaces so
  health checks don't get entangled with the generation/embedding/storage contracts.
- `app/rag/decision.py` — the RAG decision layer (see "RAG decision layer" above): `RagDecision`,
  `DecisionResult`, `RuleBasedRagDecider`. Separate from `app/rag/providers` since it doesn't call
  any provider itself — it only classifies a question.
- `app/workers` — background job placeholders.

## What is intentionally not implemented yet

- Document chunking
- Embedding generation for uploaded documents (the embedding *provider* exists; nothing calls it
  from the upload/ingestion path yet)
- Upserting uploaded-document vectors into Qdrant (the vector *store* exists; nothing calls it
  from the upload/ingestion path yet)
- Persisting extracted text anywhere — `DocumentTextExtractor`'s result is discarded once the
  worker's default processing step returns; there's no table or field for it yet
- Anything that continuously runs `IngestionWorker.process_next_job()` in a loop (no scheduler
  or long-running process invokes it yet — it's called directly, one job at a time)
- A public chat/query endpoint (including SSE streaming to clients)
- A public API endpoint for embeddings, vector store, or decision-layer operations (all internal-only)
- An LLM-based (as opposed to rule-based) question router
- Any pipeline wiring the decision layer, embeddings, vector store, retrieval, and generation into
  a full RAG flow
- Auth, rate limiting, observability/logging pipeline

These land in later milestones once the infrastructure is confirmed stable.
