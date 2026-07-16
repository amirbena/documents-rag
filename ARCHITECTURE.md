# Architecture

## System overview

`documents-rag` is a local-first RAG (Retrieval-Augmented Generation) platform. Everything runs on
the user's machine via Docker Compose, with Ollama providing local LLM and embedding inference —
no external API calls or cloud dependencies.

This milestone is infrastructure plus ten vertical slices: a FastAPI app wired to Postgres,
Redis, Qdrant, and Ollama, with a health endpoint, an Ollama health/model-availability check, a
concrete Ollama-backed embedding provider, a concrete streaming Ollama-backed LLM provider, a
concrete Qdrant-backed vector store, a rule-based RAG decision layer, a document upload +
ingestion job skeleton, an async ingestion worker that claims and resolves those jobs, a
document text extractor (`.txt`/`.md`/`.pdf`/`.docx`/`.xlsx`), a document chunker that splits
extracted text into fixed-size, overlapping, word-boundary-aware chunks, and chunk
embedding/Qdrant indexing — the worker's pipeline is now Document → extraction → chunking →
embedding → Qdrant upsert. No retrieval or chat endpoint happens yet — a claimed job's chunks
are embedded and their vectors upserted into Qdrant, but nothing yet reads them back out for
retrieval; the job resolves to `completed` only if extraction, chunking, embedding, and the
Qdrant upsert all succeed.

## Services

| Service    | Image                    | Purpose (current)                                             | Purpose (future) |
|------------|--------------------------|------------------------------------------------------------------|-------------------|
| `app`      | built from `Dockerfile`  | FastAPI process: `/health`, `/api/v1/providers/ollama/health`, `POST /api/v1/documents` | RAG API: ingestion processing, retrieval, chat |
| `postgres` | `postgres:16-alpine`     | Stores `documents`/`ingestion_jobs` rows via async SQLAlchemy     | Session/metadata storage |
| `redis`    | `redis:7-alpine`         | Available on the network                                         | Caching, task queues |
| `qdrant`   | `qdrant/qdrant:latest`   | Collection create/upsert/search via `QdrantVectorStore`           | Backing document retrieval in a future RAG flow |
| `ollama`   | `ollama/ollama:latest`   | Health/model checks + embeddings (`bge-m3`) + streaming generation (`llama3.1`) | Backing a future public chat endpoint |

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
  (`document_id`, `chunk_id`, `text`, `source`, optional `page_number`, optional `sheet_name`).
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
response (e.g. missing payload fields). `IngestionWorker` is now a caller (see "Chunk embedding
and Qdrant indexing" below) — no other caller wires it into a retrieval pipeline yet.

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

**Multilingual decision patterns.** Every pattern above has a Hebrew equivalent, in the same
single `RuleBasedRagDecider` — neither `CustomRagEngine` nor `LangChainRagEngine` implements its
own decision logic (no `CustomHebrewDecider`/`LangChainHebrewDecider`). Retrieval-intent patterns
cover Hebrew document/file references (`מסמך`/`מסמכים`, `קובץ`/`קבצים`, uploaded/attached/indexed
conjugations, "according to the file/document," "based on," "knowledge base," "written in the
file/document") — meaningful intent phrasing, never bare Hebrew-script detection: a Hebrew
question with none of these phrasings (e.g. "how does the system store vectors?") routes to
`DIRECT_LLM` like an equivalent English one. Extraction-intent patterns mirror the English
verb-near-sensitive-noun proximity check with Hebrew verbs (show/reveal/give/send/extract/
disclose) and nouns (password, credit card, ID number, credentials, secret/private key, bank
account), so "how do I reset a password?" (`איך מאפסים סיסמה?`) stays in scope exactly like its
English equivalent. A question mixing Hebrew and embedded English technical identifiers (e.g.
`איך Qdrant שומר את ה-embeddings לפי המסמך?`) is classified on the Hebrew retrieval-intent phrase
alone, same as an English question with an embedded Hebrew entity name.

## Storage Abstraction (Phase 2.6/2.7)

`app/storage/` is the provider-neutral storage layer every upload/ingestion/extraction/re-index
code path depends on — never a filesystem path or a MinIO SDK type directly.

- **`FileStorage`** (`app/storage/contract.py`) — the abstract contract: `save`, `read`,
  `delete`, `exists`, `get_metadata`, `generate_download_url`. Returns only `StoredFile`/
  `FileMetadata` (both `app/storage/contract.py`), never a provider SDK response type or a raw
  filesystem path.
- **`LocalFileStorage`** (`app/storage/local_storage.py`) — local-disk implementation. Object
  keys are relative POSIX-style paths resolved safely under a configured root
  (`LOCAL_STORAGE_ROOT`, default `storage/documents`); `app/storage/keys.py`'s
  `validate_object_key` rejects absolute paths and `..` traversal before any filesystem call.
- **`MinioFileStorage`** (`app/storage/minio_storage.py`) — S3-compatible object storage via the
  official `minio` Python SDK (used directly rather than raw `httpx`, unlike this project's
  Ollama/Qdrant providers — S3 request signing (SigV4) makes a raw-HTTP reimplementation a much
  larger, riskier undertaking than this phase calls for). Every SDK/`urllib3` exception is
  translated to a `StorageError` subclass before leaving this module.
- **`create_file_storage(settings)`** (`app/storage/factory.py`) — the *only* place a concrete
  storage class is constructed, mirroring `app/rag/providers/provider_factory.py`'s pattern
  exactly: one `if` on `FILE_STORAGE_PROVIDER` (`local`/`minio`), one dedicated
  `StorageConfigurationError` for anything else. No route, worker, or service branches on the
  provider name itself.
- **Object keys** (`app/storage/keys.py`) — `generate_object_key(document_id, original_filename)`
  builds `documents/{document_id}/{uuid-hex}{safe-suffix}`; the storage provider never invents
  its own key. `resolve_document_storage_key(document)` returns `document.storage_key` if set,
  else falls back to `document.stored_path` — this is the sole backward-compatibility path for
  documents written before this migration (see "Document/collection indexing metadata" below).
- **Storage error hierarchy** (`app/storage/errors.py`) — `StorageError` and subclasses
  (`StorageUnavailableError`, `StorageObjectNotFoundError`, `StorageWriteError`,
  `StorageReadError`, `StorageDeleteError`, `StorageMetadataError`,
  `StorageConfigurationError`, `StorageUrlGenerationError`, `StorageKeyError`). Every
  implementation translates its own provider-specific failures into one of these before the
  exception leaves the adapter, preserving the original exception as `__cause__`. Messages
  include the operation/key where safe; never a credential, connection string, or signed URL.
- **Download URLs** (`generate_download_url`) — `LocalFileStorage` returns a `file://` URI (an
  internal representation only, never returned to a client). `MinioFileStorage` returns a
  time-limited presigned GET URL (`MINIO_PRESIGNED_URL_EXPIRY_SECONDS`, default 3600s). Neither
  is ever persisted — the persisted identity is always `storage_provider`/`storage_bucket`/
  `storage_key`, never a URL. **`GET /api/v1/documents/{document_id}/download`** (Phase 2.8.2 —
  see "Document read APIs and original download" below) deliberately does *not* use
  `generate_download_url()` at all: it streams bytes through the application via
  `FileStorage.read()` instead, so a MinIO endpoint/bucket/credential is never exposed to a
  client, regardless of the configured provider.
- **Bucket initialization** (`MinioFileStorage.ensure_bucket()`) — checks whether the configured
  bucket exists; creates it only if `MINIO_CREATE_BUCKET_IF_MISSING` (default `true`); a
  `BucketAlreadyOwnedByYou`/`BucketAlreadyExists` race from concurrent startup is treated as
  success, never a failure. Never recreates or resets an existing bucket.

**Cross-system boundary — storage and PostgreSQL are not one atomic transaction.**
`app/services/documents/upload_service.py`'s `upload_document()` sequence is: save the object to
`FileStorage` → persist `Document` + `IngestionJob` rows → commit. If the commit fails after the
object was already saved, a best-effort delete of that object is attempted (failure there is
logged, never raised) before the *original* DB exception is re-raised unchanged — this is not
described as atomic anywhere, and a partially-completed attempt is not indistinguishable from one
that never ran (the orphaned object may still exist in storage until a future cleanup pass; no
orphan-cleanup worker exists yet — out of scope for this phase).

## Document upload and ingestion job skeleton

`POST /api/v1/documents` (`app/api/v1/routes/documents.py`) is the first public endpoint that
touches the database. The route is a thin controller — it reads the upload, rejects an empty
file with `400`, and delegates the save/persist/commit sequence to
`app/services/documents/upload_service.py`'s `upload_document()`, which:

1. Generates an object key (`generate_object_key`) and saves the file via the injected
   `FileStorage` (resolved once per request through `create_file_storage()` — see "Storage
   Abstraction" above; the route depends on the abstract `FileStorage`, never a concrete class).
2. Inserts a `Document` row (`app/models/document.py`): `original_filename` (stored exactly as
   received — Hebrew or any other Unicode text is preserved verbatim), `stored_filename`,
   `content_type`, `file_size`, `stored_path` (legacy, kept for backward read compatibility),
   plus the provider-neutral `storage_provider`/`storage_bucket`/`storage_key`/`storage_etag`.
3. Inserts an `IngestionJob` row (`app/models/ingestion_job.py`) with `status=PENDING`,
   referencing the `Document` via `document_id`. `IngestionStatus` (a `StrEnum`): `PENDING`,
   `PROCESSING`, `COMPLETED`, `FAILED` — stored in Postgres as their lowercase `.value`
   (`pending`, `processing`, ...), not the enum member name.

The endpoint returns `202 Accepted` with `{document_id, job_id, status}` — **it does not parse,
chunk, embed, or index the document inside the request.** An empty (zero-byte) upload is
rejected with `400` before any row is created. `IngestionWorker` (below) is what eventually
picks up and resolves the `pending` job it creates.

## Document read APIs and original download (Phase 2.8.2)

Five strictly read-only endpoints (`app/api/v1/routes/documents.py`) let a client inspect a
document's lifecycle and download its original content. **None of them mutate anything** —
Postgres, object storage, Qdrant, `IngestionJob` rows, and `VectorCleanupJob` rows are all
untouched by every route in this section. `POST /api/v1/documents/{document_id}/ingestion/retry`
(Phase 2.8.3, see "Ingestion retry and stale-job recovery" below) is the one mutating route this
module adds; there is still no delete, re-index (a separate existing endpoint/service), or
reconciliation endpoint.

- `GET /api/v1/documents?limit=&offset=` — one page of documents (`created_at` DESC, `id` DESC
  tiebreaker), each with its derived lifecycle `status` and total row count. `limit` defaults to
  20, capped at 100; `offset` defaults to 0. Always `200`.
- `GET /api/v1/documents/{document_id}` — one document's detail: identity, size, content type,
  `storage_provider` (never `storage_bucket`/`storage_key`/`storage_etag` — see "Storage
  Abstraction" above), indexing metadata (`collection_name`/`embedding_version`/
  `chunking_version`/`indexed_at`), and its latest ingestion job's id/status. `404` if the
  document doesn't exist.
- `GET /api/v1/documents/{document_id}/ingestion` — the document's latest `IngestionJob`'s
  id/status/`created_at`/`updated_at`. `404` only if the *document* doesn't exist; a document
  with no ingestion job yet is a `200` with `job_id`/`status`/`created_at`/`updated_at` all
  `null` — a legitimate lifecycle state, not a missing resource (see "Lifecycle status
  derivation" below for whether this is actually reachable in practice).
- `GET /api/v1/documents/{document_id}/failure` — the document's latest **FAILED** ingestion
  job, with a sanitized `safe_message` (see "Ingestion failure sanitization" below). `404` both
  when the document itself is missing and when the document exists but has never failed — "no
  failure to inspect" is treated as absent-resource, unlike the ingestion-status endpoint's
  `200`-with-null choice above (deliberately different, since "inspect the failure" implies one
  should exist).
- `GET /api/v1/documents/{document_id}/download` — streams the document's original bytes.
  `404` if the document row doesn't exist. `409` if the row exists but its storage object is
  missing (a real document/storage inconsistency — deliberately *not* `404`, since the document
  itself is real). `503` if the storage backend itself is unreachable/failing for a reason other
  than not-found. On success: `200` with `Content-Type` set from `Document.content_type` and
  `Content-Disposition: attachment` carrying both an ASCII-fallback `filename="..."` and an
  RFC 5987/6266 `filename*=UTF-8''...` percent-encoded form, so a Hebrew/Unicode original
  filename survives HTTP header encoding (headers are Latin-1/ASCII) without being mangled or
  raising. Reads the full object into memory via `FileStorage.read()` (both `LocalFileStorage`
  and `MinioFileStorage` return `bytes`, not a stream) and returns it in one `Response` — the
  same unbounded-memory characteristic `POST /api/v1/documents` already has today (`await
  file.read()`, no size limit anywhere in this codebase); this is not a new risk introduced
  here, and is a known, accepted limitation to revisit if large-file support becomes a
  requirement. Uses **application streaming**, never a redirect to a presigned URL — a MinIO
  endpoint/bucket/credential is never exposed to a client this way, and `generate_download_url()`
  is not called by this endpoint at all.

**Query layer**: `app/services/documents/query_service.py` owns every read query behind these
routes, following the flat function-based style of `app/services/indexing/collection_registry.py` (no
`app/repositories/` abstraction exists in this codebase). Routes stay thin per CLAUDE.md's
"Route Layer Style": parse query params, inject `AsyncSession`/`FileStorage` via `Depends`, call
one service function, copy its typed result's `status_code` (`DocumentDetailResult`/
`DocumentIngestionResult`/`DocumentFailureResult`/`DocumentDownloadResult`, mirroring
`platform_health.ReadinessResult`'s pattern) — no aggregation/business logic lives in the route
module, and no route ever returns a SQLAlchemy `Document`/`IngestionJob` directly (always through
a Pydantic response schema in `app/schemas/documents.py`).

**Lifecycle status derivation** (`DocumentLifecycleStatus`, `app/schemas/documents.py`; computed
by `derive_lifecycle_status()` in `query_service.py`) is sourced from a document's
*latest* `IngestionJob` (`created_at` DESC, `id` DESC tiebreaker) plus `Document.indexed_at`:

| Latest job | `indexed_at` | Status |
| --- | --- | --- |
| none | — | `uploaded` |
| `PENDING` | — | `pending` |
| `PROCESSING` | — | `processing` |
| `FAILED` | — | `failed` |
| `COMPLETED` | any | `indexed` |

No `IngestionJob` at all (`uploaded`) is, in practice, unreachable through the normal upload
flow — `upload_document()` always creates exactly one `Document` and one `IngestionJob` row in
the same commit — so this status exists defensively for any pre-existing/malformed data, not as
a state a client should expect to see from a document created through the API. A `COMPLETED` job
should always imply `indexed_at is not None` (`IngestionWorker.process_next_job()` commits both
in the same transaction, including for zero-chunk documents — see "Zero-chunk behavior" below),
but `derive_lifecycle_status()` still reports `indexed` even in the theoretical case where
`indexed_at` is somehow null, treating the job's own status as authoritative rather than
inventing a new status. No `deleting`/`deleted` status exists — there is no delete endpoint yet.

**Ingestion failure sanitization**: `IngestionJob.error_message` is a raw `str(exception)` (see
"Ingestion worker" below) that can embed a connection/host detail (e.g. a `QdrantVectorStoreError`
wrapping an `httpx` connect error's message). `sanitize_ingestion_error()` never returns this raw
string to a client — the `/failure` endpoint's `safe_message` is always one fixed, generic
constant ("Document ingestion failed. See server logs for the underlying error."), mirroring
`app/api/v1/routes/chat.py`'s `_SAFE_ERROR_MESSAGE` pattern rather than attempting to pattern-match
"safe" substrings out of arbitrary exception text. The raw message stays in Postgres for operator/
log inspection. `IngestionFailureResponse` has no `retryable` field — this codebase has no
attempt-count tracking, so a boolean here would be fabricated rather than genuinely derived. This
sanitization rule now extends to retry/recovery too (Phase 2.8.3): a stale-recovered job's
`error_message` (the fixed `STALE_RECOVERY_ERROR_PREFIX` marker — see below) is stored in
Postgres exactly like any other `error_message`, and passes through the same
`sanitize_ingestion_error()` before ever reaching an API response — no new sanitization path was
added, and none was needed.

**N+1 avoidance**: `GET /api/v1/documents` resolves every page row's latest job with exactly one
batched query (`get_latest_jobs_for_documents()` — `WHERE document_id IN (...)`, grouped by
`document_id` in Python), never one query per row.

## Ingestion retry and stale-job recovery (Phase 2.8.3)

`app/services/ingestion/retry_service.py` and `app/services/ingestion/stale_recovery_service.py`
add two Postgres-only, transactional operations on top of the existing `IngestionWorker`/
`IngestionJob` model — neither touches `FileStorage` or a vector store directly:

- **`retry_ingestion()`** — the service behind `POST /api/v1/documents/{document_id}/ingestion/retry`.
- **`recover_stale_ingestion_jobs()`** — an internal maintenance operation with no HTTP endpoint,
  invoked by the standalone `scripts/recover_stale_ingestion_jobs.py` (`make
  recover-stale-ingestion-jobs`).

### `POST /api/v1/documents/{document_id}/ingestion/retry`

Decision table, driven by the document's *latest* `IngestionJob` (same `created_at` DESC, `id`
DESC ordering as the read APIs above):

| Latest job | Stale? | Outcome | HTTP |
| --- | --- | --- | --- |
| Document doesn't exist | — | `DOCUMENT_NOT_FOUND` | `404` |
| none | — | `CREATED` (treated like FAILED — unreachable via the normal upload flow, same defensive stance as `DocumentLifecycleStatus.UPLOADED` above) | `202` |
| `PENDING` | — | `ALREADY_ACTIVE` (existing job returned, nothing new scheduled) | `200` |
| `PROCESSING` | no | `ALREADY_ACTIVE` | `200` |
| `PROCESSING` | yes (`> INGESTION_STALE_AFTER_SECONDS`) | `CREATED` — the stale row is flipped to `FAILED` in the *same commit* as the new job (see below) | `202` |
| `FAILED` | — | `CREATED`; the FAILED row is never modified | `202` |
| `COMPLETED` | — | `ALREADY_COMPLETED` (re-index is a separate endpoint/service, not this one) | `409` |

A `CREATED` response inserts a brand-new `PENDING` `IngestionJob` row for the existing
`IngestionWorker` to claim and process exactly like a first attempt — retry never resets,
deletes, or replays an old row's status; ingestion history is append-only.

**Why a stale-PROCESSING retry must flip the old row, unlike a FAILED retry**: the partial unique
index below allows at most one `PENDING`/`PROCESSING` row per document. A `FAILED` row is already
outside that set, so a new `PENDING` insert alongside it is always legal. A still-`PROCESSING` row
is *inside* that set — inserting a new `PENDING` row while it stays `PROCESSING` would violate the
index outright. So `retry_ingestion()` transitions a stale `PROCESSING` row to `FAILED` (using the
identical fixed `STALE_RECOVERY_ERROR_PREFIX` marker `recover_stale_ingestion_jobs()` uses) as
part of the same commit that creates the replacement — a client-triggered "reactive" recovery and
the background "proactive" one (below) are indistinguishable in stored data, just triggered from
two different call sites.

### One active job per document — a real Postgres constraint

At most one `IngestionJob` per `document_id` may be `PENDING`/`PROCESSING` — enforced by a real
partial unique index, `ix_ingestion_jobs_one_active_per_document` (migration `b7e2f6a1c9d4`,
`ON ingestion_jobs (document_id) WHERE status IN ('pending', 'processing')`), not merely
application logic. `IngestionStatus` is stored as a plain `VARCHAR` (`native_enum=False`), so the
partial index's `WHERE` clause matches the lowercase string values directly.

The migration also runs a defensive, idempotent cleanup pass immediately before creating the
index: for any document somehow already having more than one active row, it keeps only the most
recently created one active and marks the rest `FAILED` with a fixed migration-reason message.
Given the codebase at the time of this migration — `upload_document()` creates exactly one
`PENDING` job per upload, and no prior retry/re-index path could create a second job while one was
already active — this is believed genuinely unreachable in any installation that only ever ran
code up to this PR; the cleanup pass exists so the migration cannot fail outright even if that
assumption turns out to be wrong for some out-of-band data, without silently dropping a row.

**Concurrency**: `retry_ingestion()` takes a blocking `SELECT ... FOR UPDATE` on the document's
existing job rows before deciding whether to insert, so two concurrent retries for an
already-active document serialize instead of racing. A residual race remains when the latest job
is `FAILED`/absent/stale-`PROCESSING` — inserting a brand-new row is never covered by a lock taken
on rows that already existed when the lock was acquired — closed by catching the unique index's
`IntegrityError` at commit time and re-reading/returning the now-existing active job instead of
raising. Proven against real Postgres (not a fake session) by
`tests/integration/ingestion/test_concurrency.py::test_two_concurrent_retries_produce_exactly_one_new_active_job`,
which runs two independent `AsyncSession`s racing via `asyncio.gather()`.

### `recover_stale_ingestion_jobs()`

Finds `PROCESSING` jobs whose row hasn't been updated in `INGESTION_STALE_AFTER_SECONDS`, locks up
to `INGESTION_RECOVERY_BATCH_SIZE` of them with `SELECT ... FOR UPDATE SKIP LOCKED` (`updated_at
ASC, id ASC`, mirroring `IngestionWorker._claim_next_pending_job()`'s exact locking pattern so two
concurrent recovery runs never both recover the same row), marks each `FAILED` (fixed
`STALE_RECOVERY_ERROR_PREFIX` message, never deleted/reset), and creates one fresh `PENDING`
replacement per recovered row. Idempotent — a job it already recovered is `PROCESSING` no longer,
so a later call never re-selects it.

**Stale detection is an approximation, not a liveness proof.** `IngestionJob` has no dedicated
heartbeat column — `updated_at` (`onupdate=func.now()`) is the only available signal. A
`PROCESSING` job whose `updated_at` is older than the threshold is *probably* abandoned
(crashed/killed worker), but a genuinely slow-but-alive worker looks identical.
`INGESTION_STALE_AFTER_SECONDS` (default `900`) should be set well above the platform's expected
worst-case single-document processing time.

**Trigger mechanism**: `scripts/recover_stale_ingestion_jobs.py` (`make
recover-stale-ingestion-jobs`) — a standalone script, not an `app/cli/` package (this repo has no
existing CLI-package convention; introducing one for a single operation would be more machinery
than warranted), mirroring `scripts/smoke_multilingual_real.py`'s style. It is optional/manual,
never invoked by `make verify`/`make test*`/CI — the same operational boundary
`smoke-multilingual-real` already establishes for real-provider checks. A real scheduler
deployment (cron, Kubernetes CronJob, etc.) wiring this script to run periodically is intentionally
out of scope for this phase.

### Vector idempotency is free — no cleanup mechanism needed for retry

`IngestionWorker._default_process_document()` performs exactly one embedding call followed by
exactly one `vector_store.upsert_vectors()` call; if extraction/chunking/embedding raises, that
happens strictly before `upsert_vectors()` is ever reached — **a `FAILED` (or stale-recovered) job
never wrote any vectors to Qdrant.** Chunk IDs (`f"{document.id}-{chunk_index}"`,
`app/services/documents/chunker.py`) and their derived Qdrant point IDs
(`uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)`, `to_vector_point()` in
`app/services/ingestion/worker.py`) are fully deterministic for a given document, so a retry's
eventual successful upsert naturally overwrites the same point IDs a first successful attempt
would have used — Qdrant's own upsert-by-ID semantics make this idempotent with no extra
mechanism. No delete-before-retry step, temporary-collection-swap, or new cleanup logic was added
for this phase, and none is needed.

This was verified against real Qdrant, not just asserted:
`tests/integration/ingestion/test_retry_postgres.py::
test_retry_after_real_failure_writes_no_orphaned_vectors_then_succeeds` forces a real first
attempt to fail (an embedding-provider error injected before `upsert_vectors()`), confirms zero
points exist in the real ephemeral Qdrant container for that document, then retries and lets the
real `IngestionWorker` process the new job with a working embedding provider, confirming it
completes and its points are searchable via `search_similar()`. The one edge case this does *not*
cover —
orphaned points if chunking parameters change *between two genuinely-successful* indexing runs of
the same document with different chunk counts — is structurally unreachable within retry/recovery's
scope: both only ever fire for a job that never reached `upsert_vectors()`. That scenario belongs
to a future chunking-upgrade migration, deliberately out of scope here (see "Boundaries" in
CLAUDE.md).

### `IngestionWorker` claim safety — confirmed, not modified

`IngestionWorker._claim_next_pending_job()`'s existing `SELECT ... FOR UPDATE SKIP LOCKED` claim
(see "Ingestion worker" below) was reviewed and found to already satisfy every property this phase
needed: atomic claim, `PROCESSING` committed before expensive work starts, a crashed worker leaves
a recoverable `PROCESSING` row (which is exactly what stale recovery targets), and a
`COMPLETED`/`FAILED` job is never reclaimed. It was not modified, and does not need to be — retry
and recovery both only ever create a new `PENDING` row for the existing worker to pick up through
its unchanged claim path.

## Full document deletion (Phase 2.8.4)

Safe, idempotent, asynchronous deletion of a document's vectors and original stored object,
across PostgreSQL, Qdrant, and object storage. **PostgreSQL remains authoritative and the
`Document` row is never physically deleted** — a successful deletion never removes the
`Document` row, nor any `IngestionJob`/`VectorCleanupJob`/`DocumentDeletionJob` history; only the
document's external resources (Qdrant vectors, the stored object) are removed. This phase does
not implement hash deduplication, orphan reconciliation, version-aware re-indexing, a background
scheduler deployment, retention/purge policy, bulk deletion, or physical row deletion — those
remain explicitly out of scope.

### Documents service package (`app/services/documents/`)

Deletion service code lives in a small package, split by dependency direction rather than in one
mixed module:

- `app/services/documents/deletion_service.py` — request-scoped deletion state and scheduling:
  deletion-status reads (`get_latest_deletion_job()`/`get_latest_deletion_jobs_for_documents()`),
  `request_document_deletion()`, the `DeletionRequestOutcome`/`DeletionRequestResult` types routes
  map to HTTP status, and public error sanitization (`DeletionErrorCode`/
  `sanitize_deletion_error()`). This is what API routes and `query_service.py`/`retry_service.py`/
  `stale_recovery_service.py` import.
- `app/services/documents/deletion_worker.py` — background deletion execution:
  `DocumentDeletionWorker`, the claim/vector-cleanup/storage-cleanup/completion orchestration.
  This is what `scripts/process_pending_document_deletions.py` and tests exercising execution
  import.

`deletion_worker.py` depends on `deletion_service.py` (for the shared `DeletionErrorCode`
constants) — never the reverse. This mirrors the real production dependency direction: API routes
only ever need scheduling/status, never execution; only the out-of-band script/tests need
execution, and execution itself needs the shared error-code vocabulary defined on the service
side.

### `DocumentDeletionJob` — an append-only deletion-attempt ledger

`app/models/document_deletion_job.py` mirrors `IngestionJob`'s lifecycle style: `id`,
`document_id`, `status` (`DocumentDeletionStatus`: `PENDING`, `PROCESSING`, `PARTIALLY_FAILED`,
`COMPLETED` — stored as a plain `VARCHAR`, `native_enum=False`, exactly like `IngestionStatus`),
`vector_cleanup_completed`/`storage_cleanup_completed` (booleans tracking that attempt's
progress through the two-step cleanup order below), `error_code` (a stable, machine-identifiable
marker — see "Error model" below), `error_message` (the raw internal detail, Postgres-only, never
returned by a public API verbatim), `created_at`/`updated_at`/`completed_at`. A
`PARTIALLY_FAILED` row is never reset back to `PENDING` and never deleted — retrying always
creates a brand-new row for the same `document_id`; deletion history is append-only, exactly like
ingestion retry.

**One active deletion job per document** is enforced by a real partial unique index,
`ix_document_deletion_jobs_one_active_per_document` (migration `c8f3a2b6d1e7`,
`ON document_deletion_jobs (document_id) WHERE status IN ('pending', 'processing')`), mirroring
`b7e2f6a1c9d4`'s ingestion-job index exactly. Unlike that migration, `c8f3a2b6d1e7` adds **no**
defensive duplicate-row cleanup before creating the index: `document_deletion_jobs` is a
brand-new table created in the same migration, so no duplicate-active-row data can possibly exist
yet in any installation — there is no reachable path to duplicate data here at all, not merely an
unreachable-in-practice one (unlike the ingestion-job migration, which added its index onto a
pre-existing, already-populated table).

### Scheduling: `request_document_deletion()` — the service behind `DELETE /api/v1/documents/{id}`

`app/services/documents/deletion_service.py`'s `request_document_deletion()` only ever schedules a
deletion by inserting a `PENDING` `DocumentDeletionJob` row — it never performs the actual
cross-system cleanup itself, so the HTTP request never blocks on unbounded external I/O. Decision
table, driven by the document's latest `DocumentDeletionJob` and (when relevant) its latest
`IngestionJob`:

| Condition | Outcome | HTTP |
| --- | --- | --- |
| Document doesn't exist | `DOCUMENT_NOT_FOUND` | `404` |
| Latest deletion job `COMPLETED` | `ALREADY_DELETED` (idempotent; no new job) | `200` |
| Latest deletion job `PENDING`/`PROCESSING` | `ALREADY_ACTIVE` (existing job returned) | `202` |
| Latest deletion job `PARTIALLY_FAILED`, or none yet, **and** latest ingestion job is `PENDING`/`PROCESSING` | `INGESTION_ACTIVE` — deletion never races an in-flight ingestion | `409` |
| Latest deletion job `PARTIALLY_FAILED`, or none yet, and ingestion is not active | `CREATED` — a new `PENDING` `DocumentDeletionJob` is inserted | `202` |

A completed or failed ingestion never blocks deletion — full tracked cleanup is still required
even for a document whose ingestion never succeeded, since an original object/historical vectors/
cleanup records may still exist for it.

**Concurrency**: `request_document_deletion()` takes a blocking `SELECT ... FOR UPDATE` on the
document's existing deletion-job rows before deciding whether to insert, exactly mirroring
`retry_ingestion()`'s pattern — so two concurrent delete requests for the same document serialize
instead of racing. The residual insert race (a brand-new row is never covered by a lock on rows
that already existed) is closed by catching the partial unique index's `IntegrityError` at commit
time and re-reading/returning the now-existing active job instead of raising.

### Execution: `DocumentDeletionWorker` — out-of-band, mirroring `IngestionWorker`

`app/services/documents/deletion_worker.py`'s `DocumentDeletionWorker` is the execution side —
kept in a separate module from `deletion_service.py`'s request-scoped scheduling (see "Documents
service package" below for the full module boundary).

**Design choice**: this codebase has no deployed background-worker *process* for `IngestionJob`
either — `IngestionWorker.process_next_job()` is only ever invoked by test fixtures and
`scripts/`, never inline inside `POST /api/v1/documents`. `DocumentDeletionWorker` mirrors this
exact architecture rather than introducing a new one: `DELETE /api/v1/documents/{id}` only ever
schedules (inserts a `PENDING` row); the actual cross-system cleanup is a separate, independently
testable operation — `DocumentDeletionWorker.process_next_job(session)` — invoked by
`scripts/process_pending_document_deletions.py` (`make process-pending-document-deletions`,
optional/manual, mirroring `scripts/recover_stale_ingestion_jobs.py`'s boundary: never run by
`make verify`/`make test*`/CI) or directly by test code.

`DocumentDeletionWorker.process_next_job(session)`:

1. **Claim**: `SELECT ... WHERE status='pending' ORDER BY created_at LIMIT 1 FOR UPDATE SKIP
   LOCKED` — the identical locking pattern `IngestionWorker._claim_next_pending_job()` uses, so
   multiple worker invocations never claim the same job. Returns `None` if there is no pending job.
2. Flips the claimed job to `PROCESSING` and **commits immediately**, before any external I/O.
3. **Vector cleanup, strictly before storage cleanup**: calls
   `vector_deletion_service.delete_all_tracked_document_vectors(document, vector_store, session)` — the
   full-tracked-collection operation (targets the document's active collection *and* every
   distinct historical collection still tracked by a pending/failed `VectorCleanupJob`) — **never**
   `delete_current_document_vectors()` (the deliberately-partial, active-collection-only sibling).
   If any targeted collection's deletion fails, the job is marked `PARTIALLY_FAILED` with
   `vector_cleanup_completed=False` and `error_code=DOCUMENT_VECTOR_CLEANUP_FAILED`, and the
   method returns immediately — **storage cleanup is never attempted in that call**, enforced
   structurally by the code path (there is no branch reaching storage deletion without first
   observing `VectorDeletionResult.fully_deleted is True`), not merely as a documented intention.
4. On full vector-cleanup success, `vector_cleanup_completed` is committed `True`, then
   `FileStorage.delete(storage_key)` is called. An already-missing object is treated as an
   idempotent success (per `FileStorage.delete()`'s existing contract — see "Storage Abstraction"
   above); any other failure (`StorageError` subclass) marks the job `PARTIALLY_FAILED` with
   `storage_cleanup_completed=False` and `error_code=DOCUMENT_STORAGE_CLEANUP_FAILED`.
5. Only when both steps succeed is the job marked `COMPLETED` with `completed_at` set — this is
   the only path a document's public lifecycle status becomes `deleted` (see below).

**Why vectors before storage**: searchable derived content (vectors) must stop being searchable
before a document is ever reported as deleted; the original object plays no role in retrieval, so
it can safely be cleaned up afterward (and retried independently) without risking a document that
looks "deleted" while still being findable through chat/retrieval.

**Retry is append-only, not resumable-in-place.** A `PARTIALLY_FAILED` job's
`vector_cleanup_completed`/`storage_cleanup_completed` flags describe *that* attempt only.
Retrying (calling `request_document_deletion()` again) creates a brand-new `PENDING` row that
re-attempts both steps from scratch — always safe, because `delete_all_tracked_document_vectors()`
and `FileStorage.delete()` are both independently idempotent (re-deleting already-absent
vectors/objects is a harmless no-op success), so re-running an already-completed step costs
nothing but the extra I/O. No cross-job "resume" bookkeeping exists or is needed.

`IngestionWorker._claim_next_pending_job()` itself was reviewed and left **unmodified** — it
already satisfies every property this phase needed, and `DocumentDeletionWorker` uses its own,
independent claim query against `document_deletion_jobs`, never touching `ingestion_jobs`.

### Lifecycle precedence — deletion always wins

`derive_lifecycle_status()` (`app/services/documents/query_service.py`) now takes the document's
latest `DocumentDeletionJob` (if any) as an additional input and checks it **first**, before the
existing ingestion-derived rule:

| Latest deletion job | Status |
| --- | --- |
| `PENDING`/`PROCESSING` | `deleting` |
| `PARTIALLY_FAILED` | `deletion_failed` |
| `COMPLETED` | `deleted` |
| none | falls through to the ingestion-derived rule (see "Document read APIs" above) |

Once a document is `deleted`, nothing about its (unchanged) `Document.collection_name`/
`indexed_at`/`IngestionJob` columns can ever make it look `indexed`/`pending`/etc. again — the
deletion job, not those columns, is authoritative for lifecycle purposes from that point on. A
deleted document remains listed/inspectable via `GET /api/v1/documents` and `GET
/api/v1/documents/{id}` (lifecycle=`deleted`) — read APIs never filter it out.

### Public API

- **`DELETE /api/v1/documents/{document_id}`** — `202` when a new deletion was scheduled
  (`created=true`) or an existing active job was returned (`created=false`); `200` when the
  document was already fully deleted (idempotent, `created=false`); `404` if the document does
  not exist; `409` if the document has an active (`PENDING`/`PROCESSING`) ingestion job. Body:
  `{document_id, deletion_job_id, status, created}`.
- **`GET /api/v1/documents/{document_id}/deletion`** — the document's latest deletion attempt:
  `{document_id, deletion_job_id, status, vector_cleanup_completed, storage_cleanup_completed,
  safe_message, created_at, updated_at, completed_at}`. `404` if no deletion was ever requested
  for this document. Never exposes a storage key/bucket, Qdrant collection name, or raw provider
  exception — `safe_message` is always `sanitize_deletion_error()`'s fixed, generic text (`None`
  when there is no recorded failure).
- **`GET /api/v1/documents/{document_id}/download`** on a `deleted` document returns **`410
  Gone`**, not `404` — the Postgres resource still exists; only its content was intentionally
  removed. This is checked before any storage I/O is attempted.
- **`POST /api/v1/documents/{document_id}/ingestion/retry`** rejects with **`409`** whenever *any*
  `DocumentDeletionJob` exists for the document at all (`deleting`/`deletion_failed`/`deleted`) —
  checked first, before any of `retry_ingestion()`'s existing ingestion-status branches — so a
  document is never implicitly resurrected by retrying its ingestion once deletion has begun.

### Error model

Stable, machine-identifiable public error codes (`DeletionErrorCode`, StrEnum):
`DOCUMENT_VECTOR_CLEANUP_FAILED`, `DOCUMENT_STORAGE_CLEANUP_FAILED`. `sanitize_deletion_error()`
maps a stored `error_code` to one of two fixed, generic public messages (or a generic fallback for
an unrecognized code) — the raw `error_message` (which may embed a Qdrant/MinIO connection detail)
never reaches a client; it stays in Postgres for operator/log inspection only, mirroring
`sanitize_ingestion_error()`'s existing pattern exactly.

### Explicit exclusions (Phase 2.8.4)

Physical deletion of the `Document` row; cascading deletion of `IngestionJob`/`VectorCleanupJob`/
`DocumentDeletionJob` history; hash deduplication; orphan reconciliation; version-aware
re-indexing; re-upload/restore; a deployed background-scheduler process; retention/purge policy;
tenant/user authorization changes; bulk deletion; a vector-only deletion HTTP endpoint; and a
stale-`PROCESSING`-deletion-job recovery mechanism (unlike ingestion, deletion execution has no
analogous `recover_stale_*` operation in this phase — a `PROCESSING` deletion row that never
resolves stays `PROCESSING` until a future phase adds one, exactly as `IngestionJob` did before
Phase 2.8.3 introduced stale recovery for ingestion specifically).

## Ingestion worker

`IngestionWorker` (`app/services/ingestion/worker.py`) is an internal service — **no public API**
— that claims and resolves one pending `IngestionJob` at a time via `process_next_job(session)`:

1. **Claim**: `SELECT ... WHERE status='pending' ORDER BY created_at LIMIT 1 FOR UPDATE SKIP
   LOCKED` — Postgres row-level locking so multiple worker instances never claim the same job.
   Returns `None` if there is no pending job.
2. Flips the claimed job to `PROCESSING` and **commits immediately** — a separate transaction
   boundary from the outcome below, so the claim is durable before any processing is attempted.
3. Looks up the associated `Document` and calls the injected processing step (default:
   `_default_process_document`, which runs Document → extraction (`DocumentTextExtractor` —
   see "Document text extraction" below) → chunking (`DocumentChunker` — see "Document
   chunking" below) → embedding → Qdrant upsert (see "Chunk embedding and Qdrant indexing"
   below)) with `(document, job)`.
4. On success: `PROCESSING` → `COMPLETED`, committed.
   On any exception (extraction, chunking, embedding, or Qdrant upsert): `PROCESSING` → `FAILED`,
   `error_message` set to `str(exception)`, committed.

The processing step is injected via the constructor (`IngestionWorker(process_document=...)`) so
tests can substitute a fake pipeline without changing the claim/lock/transition logic. A
`FileStorage` is also injected (`IngestionWorker(file_storage=...)`, defaulting to
`create_file_storage()` if omitted) and threaded into the default processing step's
`DocumentTextExtractor` — the worker never resolves a filesystem path itself.
`IngestionWorker` never imports or calls `LLMProvider` itself — ingestion embeds and indexes
chunks, it never generates text.

**Idempotent by construction**: once a job leaves `PENDING` (to `PROCESSING`, then `COMPLETED`
or `FAILED`), the claim query's `WHERE status='pending'` filter can never select it again —
calling `process_next_job()` repeatedly does not require any separate "already processed" check.

`with_for_update(skip_locked=True)` is Postgres-specific row-locking syntax; SQLite does not
represent it correctly even if it accepts the same SQLAlchemy call, so this project deliberately
does not add SQLite/`aiosqlite` for testing this worker. Its tests use a fake `AsyncSession`
double that faithfully simulates the pending-job filter and `Document` lookup instead.

## Document text extraction

`DocumentTextExtractor` (`app/services/documents/text_extractor.py`) is the ingestion worker's
default processing step: given an injected `FileStorage`, it reads a `Document`'s content via
`storage.read(resolve_document_storage_key(document))` and extracts its raw text entirely in
memory — **no chunking, embedding, or Qdrant upsert, and no temporary file materialization**.
pypdf/python-docx/openpyxl all accept an in-memory `io.BytesIO` stream directly, so bytes read
from `FileStorage` are parsed without ever touching a local path — this is true regardless of
which `FileStorage` implementation is configured (local or MinIO). It routes by file extension
(from `original_filename`), then validates the file's basic structure/content against what that
extension claims before attempting to parse it (see "Routing and validation" below). It supports
exactly five file types:

- `.txt` / `.md` — read as UTF-8 text and returned as a single `ExtractedPage` with
  `page_number=None`, `sheet_name=None`. Hebrew and other non-ASCII Unicode content is preserved
  exactly.
- `.pdf` — extracted page by page via `pypdf` (`PdfReader`), producing one `ExtractedPage` per
  page with a 1-indexed `page_number`, so downstream chunking/citation can reference the
  original page a piece of text came from.
- `.docx` — extracted via `python-docx`: all paragraph text joined into a single `ExtractedPage`
  (`page_number=None`, `sheet_name=None`) — plain text only, no tables, headers/footers, or
  pagination.
- `.xlsx` — extracted sheet by sheet via `openpyxl` (`load_workbook(..., read_only=True,
  data_only=True)`), producing one `ExtractedPage` per worksheet with `sheet_name` set to the
  worksheet's title and `page_number=None`; each row's non-empty cell values are tab-joined.

Any other extension raises `DocumentTextExtractionError("Unsupported file type: ...")` — there
is no fallback or content-based detection.

Data types:

- `ExtractedPage` — `text: str`, `page_number: int | None` (PDF only), `sheet_name: str | None`
  (XLSX only) — both `None` for `.txt`/`.md`/`.docx`, which have no natural pagination.
- `ExtractedDocument` — `document_id: str`, `pages: list[ExtractedPage]`, plus a `full_text`
  property that joins all pages' text.

`extract(document)` runs the actual file I/O and PDF parsing off the event loop via
`asyncio.to_thread` (both are blocking operations). `DocumentTextExtractionError` is raised for:
a missing `stored_path` file, an unsupported extension, or a file whose extracted text is empty
or whitespace-only. Any of these propagate up through `IngestionWorker.process_next_job()` and
resolve the job to `failed` with the error message stored — extraction never crashes the worker
process itself.

### Routing and validation

`DocumentTextExtractor` decides how to parse a file from `Path(stored_path).suffix`, then
validates the file's basic structure/content against what that extension claims **before**
handing it to the corresponding parser (`_validate_file_type`, called from `_extract_sync`
ahead of any extraction call):

| Extension | Handler | Validation before parsing |
|-----------|---------|----------------------------|
| `.txt`    | UTF-8 plain text (`path.read_text(encoding="utf-8")`) | Readable as UTF-8 (a `UnicodeDecodeError` is caught and re-raised as `DocumentTextExtractionError`) |
| `.md`     | UTF-8 markdown/plain text (same as `.txt` — no Markdown parsing) | Same as `.txt` |
| `.pdf`    | `pypdf` (`PdfReader`), page by page | First 4 bytes equal the PDF header `%PDF` |
| `.docx`   | `python-docx` (`docx.Document`), paragraph text | Valid ZIP archive (`zipfile.is_zipfile`) containing `word/document.xml` |
| `.xlsx`   | `openpyxl` (`load_workbook`), sheet by sheet | Valid ZIP archive containing `xl/workbook.xml` |

This is still lightweight, structural validation, not full content sanitization — it catches a
mismatched or corrupt file before wasting effort on the wrong parser (e.g. an `.xlsx` renamed to
`.pdf`, or arbitrary bytes given a document extension), each raising a specific
`DocumentTextExtractionError`. It does not validate the upload's `content_type` header, do deep
MIME/magic-byte sniffing beyond the checks above, or scan file contents for malicious payloads —
those remain future hardening if ever needed.

The extracted result is passed directly into chunking (see "Document chunking" below) — nothing
persists the raw extracted text itself.

## Document chunking

`DocumentChunker` (`app/services/documents/chunker.py`) is the ingestion worker's second
processing step: it takes the `ExtractedDocument` that `DocumentTextExtractor` produced and
splits each page's text into fixed-size, overlapping chunks — **no embedding generation, no
Qdrant upsert, no retrieval**.

- Input: `ExtractedDocument` (one `ExtractedDocument` covering all of a document's pages/sheets).
- Output: `list[DocumentChunk]`.
- `DocumentChunk` — `document_id: str`, `chunk_id: str`, `text: str`, `chunk_index: int`,
  `page_number: int | None`, `sheet_name: str | None`. `page_number`/`sheet_name` are copied
  straight from the source `ExtractedPage` a chunk came from — `page_number` set for chunks from
  a PDF page, `sheet_name` set for chunks from an XLSX sheet, both `None` for `.txt`/`.md`/`.docx`
  chunks (which have no natural pagination).

Chunking rules:

- **Fixed target size** (`chunk_size`, in characters) and **configurable overlap**
  (`chunk_overlap`, in characters) — both configured via `CHUNK_SIZE`/`CHUNK_OVERLAP` (see
  "Environment variables" below), and both also settable directly via
  `DocumentChunker(chunk_size=..., chunk_overlap=...)`. The constructor raises `ValueError` if
  `chunk_overlap >= chunk_size` or either value is non-positive/negative.
- **Word-boundary-aware**: chunks are built by accumulating whole words up to `chunk_size`
  characters — a chunk never ends or starts mid-word. Overlap is built from the trailing whole
  words of the previous chunk, up to `chunk_overlap` characters.
- **Empty chunks are ignored**: a page whose text is empty or whitespace-only (after
  `str.split()`) produces zero chunks, not an empty-text chunk.
- **Deterministic**: `chunk()` is a pure function of its input — the same `ExtractedDocument`
  always produces the same chunks, in the same order, with the same `chunk_id`s
  (`f"{document_id}-{chunk_index}"`, where `chunk_index` increments continuously across all
  pages of the document, not reset per page).

The chunker's output feeds directly into embedding (see "Chunk embedding and Qdrant indexing"
below) — there is still no table or field that persists chunk text itself in Postgres; Qdrant is
the system of record for chunk vectors and their metadata.

## Chunk embedding and Qdrant indexing

The ingestion worker's third and fourth processing steps (`app/services/ingestion/worker.py`)
turn each `DocumentChunk` into an indexed vector — **no retrieval, chat, or SSE endpoint reads
these back out yet**:

1. **Embed**: `get_embedding_provider()` (reads `EMBEDDING_PROVIDER`) embeds every chunk's text
   in one call — `embedding_provider.embed([chunk.text for chunk in chunks])` — returning one
   vector per chunk, in the same order.
2. **Build `VectorPoint`s**: each chunk + its vector becomes a `VectorPoint` with `id` set to a
   deterministic `uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)` (so re-processing a document
   overwrites the same Qdrant points instead of duplicating them, and the id is always a
   Qdrant-valid UUID regardless of `chunk_id`'s own format), and payload fields `document_id`,
   `chunk_id`, `text`, `source` (the `Document.original_filename`), `page_number`, and
   `sheet_name` carried straight over from the chunk.
3. **Index**: `get_vector_store()` (reads `VECTOR_STORE_PROVIDER`) creates the collection named
   `QDRANT_COLLECTION_NAME` with dimensionality `VECTOR_SIZE` if it doesn't already exist
   (cheap no-op if it does), then upserts all points in one `upsert_vectors` call.

A document that produces zero chunks (e.g. empty/whitespace-only extracted text) short-circuits
before calling the embedding provider or vector store at all. A failure at either step
(`EmbeddingProvider.embed` or `VectorStore.create_collection_if_not_exists`/`upsert_vectors`
raising) propagates up through `_default_process_document` exactly like an extraction or
chunking failure — `IngestionWorker` catches it and marks the job `FAILED` with
`error_message = str(exception)`. `IngestionWorker` still never imports or calls `LLMProvider` —
this pipeline embeds and indexes, it never generates text.

## Retrieval service

`RetrievalService` (`app/rag/retrieval_service.py`) is the internal read-side counterpart to
chunk embedding/indexing: given a query, it embeds it and searches Qdrant for relevant chunks —
**no public retrieval/chat/SSE endpoint exists yet, and no LLM call is made**.

`retrieve(query: str, limit: int | None = None) -> list[VectorSearchResult]`:

1. **Validate**: an empty/whitespace-only `query` raises `EmptyQueryError` before any provider is
   called.
2. **Embed**: `get_embedding_provider()` (reads `EMBEDDING_PROVIDER`) embeds the query text —
   `embedding_provider.embed([query])[0]` — using the same fixed embedding model ingestion used,
   so query and chunk vectors stay comparable.
3. **Search**: `get_vector_store()` (reads `VECTOR_STORE_PROVIDER`) runs
   `search_similar(QDRANT_COLLECTION_NAME, query_vector, limit)`, where `limit` is the caller's
   explicit `limit` if given, else `RETRIEVAL_TOP_K`. Qdrant returns results already ordered by
   score, and `RetrievalService` preserves that order.
4. **Threshold filter**: if `RETRIEVAL_SCORE_THRESHOLD` is set, results scoring below it are
   dropped; left unset (`None`), no score filtering happens.

Each returned `VectorSearchResult` preserves `document_id`, `chunk_id`, `text`, `source`,
`page_number`, `sheet_name`, and `score`. A failure in either the embedding provider or the
vector store propagates unchanged — `RetrievalService` does not catch or wrap it — and zero
matching results (or all filtered out by the threshold) simply return an empty list rather than
fabricating context. `RetrievalService` never imports or calls `LLMProvider`.

## RAG prompt builder

`RagPromptBuilder` (`app/rag/prompt_builder.py`) is a pure, synchronous, deterministic function
of its inputs: given a question and a list of `VectorSearchResult`s (from `RetrievalService`), it
builds a `BuiltRagPrompt` — **no LLM call, no public chat/SSE endpoint, no retrieval of its own,
no conversation memory**.

`build(question: str, results: list[VectorSearchResult]) -> BuiltRagPrompt`:

1. **Filter**: results with empty/whitespace-only `text` are dropped before anything else — they
   never appear in the context or in `sources`.
2. **No-results path**: if nothing remains after filtering, `context` is set to a fixed sentence
   stating no relevant context was found, `sources` is `[]`, and `user_prompt` is built from that
   same fixed context — deterministic, and no fallback content is fabricated.
3. **Label and format**: otherwise, each remaining result is processed **in the order given**
   (the caller's retrieval rank is preserved, never re-sorted) and assigned a stable label —
   `[S1]`, `[S2]`, ... — used as both the context block's marker and the implicit index into
   `sources`. Each context block is `"{label} {source}[ page {page_number}][ sheet
   {sheet_name}]\n{text}"`, joined with blank lines.
4. **Attribution**: each context block has a matching `PromptSource` (`document_id`, `chunk_id`,
   `source`, `score`, `page_number`, `sheet_name`) appended to `sources` in the same order.

`BuiltRagPrompt` fields:

| Field | Type | Contents |
|---|---|---|
| `system_prompt` | `str` | Fixed instruction: answer only from the supplied context, never invent missing information, say explicitly when the answer isn't present |
| `user_prompt` | `str` | The question plus the formatted `context`, with a closing reminder to answer only from context |
| `context` | `str` | The joined, labeled context blocks (or the fixed no-results sentence) |
| `sources` | `list[PromptSource]` | Attribution metadata per context block, in context order |

`RagPromptBuilder` never mutates the `VectorSearchResult`s or the list passed to `build()`, and
never imports or calls `LLMProvider` or `RetrievalService` itself — it only shapes already-ranked
results into prompt text, leaving the actual retrieval call to whoever composes it with
`RetrievalService`.

## RAG orchestrator

`RagOrchestrator` (`app/rag/orchestrator.py`) is the single component that composes
`RuleBasedRagDecider`, `RetrievalService`, `RagPromptBuilder`, and `LLMProvider` (via
`get_llm_provider()`) into one call — **no conversation memory, and no silent fallback between
decisions or providers**. It's exposed publicly via `POST /api/v1/chat` (see "Streaming chat
endpoint" below).

`stream_answer(question: str) -> AsyncIterator[OrchestratorMetadata | OrchestratorToken]`:

1. **Decide**: `RuleBasedRagDecider.decide(question)` runs first, exactly as it does standalone
   (see "RAG decision layer" above) — no LLM call is made to route.
2. **`CLARIFICATION_NEEDED` / `OUT_OF_SCOPE`**: yields one `OrchestratorMetadata`
   (`retrieval_used=False`, `sources=[]`) then a single fixed `OrchestratorToken` — neither
   `RetrievalService` nor any `LLMProvider` method is called on this path.
3. **`NEEDS_RETRIEVAL`**: calls `RetrievalService.retrieve(question)`, passes the results to
   `RagPromptBuilder.build(question, results)`, yields one `OrchestratorMetadata`
   (`retrieval_used=True`, `sources` from the built prompt), then streams
   `LLMProvider.stream_generate(f"{system_prompt}\n\n{user_prompt}")` chunk by chunk as
   `OrchestratorToken`s.
4. **`DIRECT_LLM`**: yields one `OrchestratorMetadata` (`retrieval_used=False`, `sources=[]`),
   then streams `LLMProvider.stream_generate(...)` directly from a fixed system prompt plus the
   question — no retrieval call.

`OrchestratorMetadata` (`decision`, `reason`, `retrieval_used`, `sources: list[PromptSource]`) is
always the first event of a `stream_answer()` run; every subsequent event is an
`OrchestratorToken(text: str)`, in the exact order the LLM (or the fixed message) produced them.
A failure raised by `RetrievalService.retrieve()` or `LLMProvider.stream_generate()` propagates
unchanged out of the async generator — `RagOrchestrator` does not catch it to substitute a
direct-LLM answer for a failed retrieval, and does not catch a provider failure to retry with a
different provider; `get_llm_provider()`'s existing no-silent-fallback guarantee (see "Provider
factory" above) is preserved end-to-end.

This required extending `LLMProvider`'s abstract contract with `stream_generate(prompt) ->
AsyncIterator[str]` alongside the existing `generate(prompt) -> str` (previously only
`OllamaLLMProvider` exposed streaming). `LLMProviderStub` — the shared base for
`OpenAIProvider`/`GeminiProvider`/`AnthropicProvider` — now raises
`ProviderNotImplementedError` from `stream_generate()` too, keeping the "stub never calls out"
guarantee for both methods.

## RAG Engine Compatibility Layer

`RagOrchestrator` remains the platform's single reference RAG implementation, but
`POST /api/v1/chat` no longer depends on it directly — it depends on a small `RagEngine`
abstraction (`app/rag/engine.py`), so a second, LangChain-backed execution engine can be
selected without touching the public API, the SSE contract, or any existing
provider/retrieval/prompt/orchestration code:

```
RagEngine (app/rag/engine.py)
├── CustomRagEngine    (app/rag/engines/custom_engine.py)   — default
└── LangChainRagEngine (app/rag/engines/langchain_engine.py) — optional
```

**Contract**: `stream_answer(question: str) -> AsyncIterator[OrchestratorMetadata |
OrchestratorToken]`, plus an `answer(question) -> str` default that collects the streamed
tokens. Both concrete engines yield the exact same `OrchestratorMetadata`/`OrchestratorToken`
dataclasses `RagOrchestrator` already defines — `RagEngine` is independent of FastAPI and SSE
formatting, so `app/api/v1/routes/chat.py`'s SSE mapping needs no engine-specific branch.

**Engine-selection flow**: `get_rag_engine(settings)`
(`app/rag/engines/engine_factory.py`) reads `RAG_ENGINE` and resolves the concrete engine — the
same "resolve, don't branch" shape as `provider_factory.py`. `RAG_ENGINE=custom` (the default)
resolves to `CustomRagEngine`; `RAG_ENGINE=langchain` resolves to `LangChainRagEngine`; any other
value raises `UnsupportedRagEngineError` immediately. There is no silent fallback to `custom` and
no silent provider switch — mirroring the **no-fallback rule** `provider_factory.py` already
established for `LLM_PROVIDER`/`EMBEDDING_PROVIDER`/`VECTOR_STORE_PROVIDER`.

**`CustomRagEngine`** adds no logic: it constructs (or accepts) a `RagOrchestrator` and delegates
`stream_answer()` straight to it. `RuleBasedRagDecider`, `RetrievalService`, `RagPromptBuilder`,
`LLMProvider.stream_generate()`, source metadata, and failure propagation are byte-for-byte the
same as before this layer existed — this is why `RagOrchestrator` remains the reference
implementation: every other engine is judged against its behavior, not the other way around.

**`LangChainRagEngine`** (`app/rag/engines/langchain_engine.py`) reuses
`RuleBasedRagDecider.decide(question)` directly, unmodified, *outside* any LangChain `Runnable` —
this keeps the decision contract (which four `RagDecision` values exist, and their `reason`
strings) identical to `CustomRagEngine`'s without needing LangChain to model routing at all.
`CLARIFICATION_NEEDED`/`OUT_OF_SCOPE`/`NEEDS_RETRIEVAL`-with-no-results stream a fixed,
language-appropriate message resolved via `PromptProvider` (see "Multilingual RAG Foundation"
below) — neither engine owns this text or depends on the other's implementation module, and the
two can never drift apart on it — with no retrieval (for clarification/out-of-scope) and no LLM
call at all for any of the three. For `NEEDS_RETRIEVAL`-with-sources/`DIRECT_LLM`, it builds a
LangChain `ChatPromptValue` (from literal `SystemMessage`/`HumanMessage` content — never an
interpolated `ChatPromptTemplate`, so arbitrary document text containing `{`/`}` characters can
never be misparsed as a template variable) using `PromptProvider`'s resolved, language-aware
system instruction, and pipes it through a LangChain `RunnableLambda(...) | ProviderBackedLLM`
chain, streaming the result chunk by chunk.

**Adapter boundaries and provider-factory reuse** (`app/rag/engines/langchain_adapters.py`) —
the *only* place `LangChainRagEngine` touches a LangChain provider-facing base class:

- `ProviderBackedLLM` (`langchain_core.language_models.llms.LLM`) streams from whatever
  `LLMProvider` `app.rag.providers.provider_factory.get_llm_provider()` resolved.
- `ProviderBackedEmbeddings` (`langchain_core.embeddings.Embeddings`) wraps whatever
  `EmbeddingProvider` `get_embedding_provider()` resolved.
- `ProviderBackedRetriever` (`langchain_core.retrievers.BaseRetriever`) wraps the existing
  `RetrievalService` — the same `QdrantVectorStore`, `QDRANT_COLLECTION_NAME`,
  `RETRIEVAL_TOP_K`/`RETRIEVAL_SCORE_THRESHOLD` filtering, and embedding provider every other
  caller of `RetrievalService` gets. There is no `langchain-community` Qdrant vector-store
  integration and no second Qdrant SDK path — `QdrantVectorStore`'s own `httpx`-based HTTP calls
  are the only thing that ever talks to Qdrant, in either engine.

None of the three adapters construct an Ollama, OpenAI, Gemini, Anthropic, or Qdrant client
directly — each one is handed an already-resolved provider/service instance and only adapts its
interface, never its configuration or selection.

**Shared Qdrant/embedding contract**: `ProviderBackedRetriever` returns LangChain `Document`s
built from `VectorSearchResult` (`document_to_search_result()`/`_search_result_to_document()` in
`langchain_adapters.py` are exact inverses of each other), and `LangChainRagEngine` converts
retrieved `Document`s straight back into `VectorSearchResult`s before handing them to the
existing, unmodified `RagPromptBuilder`. This means: same embedding model, same `VECTOR_SIZE`,
same `QDRANT_COLLECTION_NAME`, same vectors and payload metadata, same `[S1]`/`[S2]` source
labels and rank order, same governance instructions ("answer only from context", "say so if the
answer isn't present"), and the same Hebrew/Unicode handling as `CustomRagEngine` — nothing about
switching engines re-embeds a document, creates a new collection, or changes a chunk/point ID.

**API/SSE independence**: `app/api/v1/routes/chat.py` depends on `RagEngine` (via
`get_rag_engine()`, a route-local dependency wrapping the factory), not `RagOrchestrator` or
`LangChainRagEngine` — the route has no knowledge of which concrete engine is configured, no
`RAG_ENGINE` branch, and (per "Route Layer Style" in CLAUDE.md) no decision/retrieval/prompt
logic of its own either way.

**No-results behavior is identical across engines**: when `RetrievalService.retrieve()` returns
results but `RagPromptBuilder` finds nothing attributable (`built.sources` empty), both engines
stream a fixed, language-appropriate `no_results` message (see "Multilingual RAG Foundation"
below) with `sources=[]` and **no LLM call at all** — `LangChainRagEngine` never substitutes a
`DIRECT_LLM` answer or fabricates a source in that case, matching `CustomRagEngine` exactly.

**Why LangGraph is intentionally deferred**: LangChain's `Runnable`/prompt/retriever primitives
are sufficient to express this platform's existing four-way decision routing plus a single
retrieval-then-generate step — there is no multi-step agent loop, no tool calling, and no
conversation memory for LangGraph's graph/state machinery to add value to. Introducing LangGraph
now would add a second orchestration paradigm with nothing for it to orchestrate; it belongs in a
future milestone only once a real agentic workflow (multi-step tool use, conditional branching
driven by intermediate LLM output, etc.) actually requires it.

## Multilingual RAG Foundation

Phase 2.5 makes multilingual (Hebrew + English) retrieval and language-aware prompting shared
platform capabilities, reached identically by both `CustomRagEngine` and `LangChainRagEngine` —
neither engine implements its own language detection, prompt catalog, embedding-version
selection, or collection routing.

```
Question
   ↓
LanguageDetector          (app/rag/language.py)
   ↓
PromptProvider            (app/rag/prompts/provider.py)
   ↓
PromptCatalog             (app/rag/prompts/catalog.py)
   ↓
ResolvedPrompt             (app/rag/prompts/types.py)
   ↓
RagEngine
   ├── CustomRagEngine
   └── LangChainRagEngine
```

### Embedding/index versioning (`app/rag/embedding_config.py`)

`EmbeddingIndexConfig` is the versioned identity of "how this platform is currently indexing
documents" — `collection_prefix`, `provider`, `model`, `dimension`, `embedding_version`,
`chunking_version`. `get_active_embedding_config(settings)` is the *only* place that reads
`EMBEDDING_PROVIDER`/`EMBEDDING_MODEL` (or `OLLAMA_EMBEDDING_MODEL`)/`VECTOR_SIZE`/
`EMBEDDING_VERSION`/`CHUNKING_VERSION` for indexing purposes — `IngestionWorker` (write side),
`RetrievalService` (read side), and `app/services/indexing/reindex_service.py` all call this function
rather than reading those settings directly, so they can never resolve to different
configurations. Every field is validated non-empty/positive at construction — see "Configuration
Must Be Explicit" below.

`EmbeddingIndexConfig.collection_name` derives a deterministic, sanitized Qdrant collection name
from all five fields (`documents__ollama__bge-m3__ev2__cv1__d1024`-shaped) — changing
*any* field (a different model, dimension, embedding version, or chunking version) always
produces a different collection name, so incompatible vectors can never land in the same
collection. `QDRANT_COLLECTION_NAME` now serves as the `collection_prefix` input to this
identity, not a literal collection name by itself.

### Why incompatible dimensions cannot share a collection

`app/services/indexing/collection_registry.py`'s `ensure_active_collection()` is the one gate every
write/search path passes through before touching Qdrant: it calls
`VectorStore.get_collection_vector_size()` (new on the `VectorStore`/`QdrantVectorStore`
contract, alongside `delete_by_document_id()`) and raises `IncompatibleIndexConfigurationError`
if an existing collection's dimension doesn't match the active config's — this should be
unreachable in practice (the collection name itself encodes the dimension) but is checked anyway
as a hard safety net against any Qdrant/Postgres drift. **Never** silently recreates or deletes a
mismatched collection; an operator must resolve the conflict deliberately.

### Document/collection indexing metadata (Postgres)

`IndexCollection` (`app/models/index_collection.py`) tracks one row per distinct collection ever
created: `collection_name` (primary key), `embedding_provider`, `embedding_model`,
`embedding_dimension`, `embedding_version`, `chunking_version`, `status` (`active`/`retired`),
`created_at`. `Document` (`app/models/document.py`) gained matching `embedding_*`/
`chunking_version`/`collection_name`/`indexed_at` columns, populated only by
`app/services/indexing/collection_registry.py`'s `mark_document_indexed()` **after** a successful
index/re-index — a failed attempt never updates them. `is_document_stale(document, config)`
compares `document.collection_name` against the active config's collection name — a document
with vectors sitting in some collection is not "current" merely because vectors exist somewhere;
it is current only if its stored configuration matches the active one exactly. Migration:
`alembic/versions/07f849bf2b95_...py`. `Document` also carries the provider-neutral storage
identity added by Phase 2.6/2.7 (`storage_provider`/`storage_bucket`/`storage_key`/
`storage_etag`), migration `alembic/versions/a3f9c7d2e1b5_add_document_storage_identity_
columns.py` — see "Storage Abstraction" above and "Backward compatibility for pre-migration
documents" below.

**Content-hash deduplication (Phase 2.8.5).** `Document.content_hash` (nullable `VARCHAR(64)`,
migration `alembic/versions/4a4f5c0674f4_add_document_content_hash_column.py`) holds a lowercase
hex SHA-256 digest of a document's uploaded bytes, enforced unique when non-null via the named
index `uq_documents_content_hash` — a normal (non-partial) unique index is sufficient since
PostgreSQL never treats two `NULL`s as equal. `app.services.documents.upload_service
.upload_document()` computes this hash and calls `app.services.documents.dedup_service
.decide_upload()` as a fast path before ever writing to storage: a matching document with no
blocking deletion state is reused (no new object, no new `Document`, no new `IngestionJob`); a
matching document with an active/incomplete deletion raises a typed internal exception instead of
being treated as reusable. The database unique index — never the fast-path lookup — is what
actually guarantees exactly one logical document survives two genuinely concurrent identical
uploads: the losing commit's `IntegrityError` is inspected via the PostgreSQL diagnostic
`constraint_name` (never message-text matching) to confirm it is specifically
`uq_documents_content_hash` before being treated as a race, its own just-written object is
best-effort deleted, and the winning row is reloaded and re-evaluated through the same lifecycle
decision. `app.services.documents.deletion_worker.DocumentDeletionWorker` releases a document's
hash (`content_hash = NULL`) only in the same commit as its deletion job reaching `COMPLETED` —
never on `PENDING`/`PROCESSING`/`PARTIALLY_FAILED` — so a later upload of the same bytes may claim
it again only once deletion has genuinely, fully finished. No existing row is backfilled.
`POST /api/v1/documents` exposes this decision publicly (Phase 2.8.5 subtask 4):
`DocumentUploadOutcome` (`CREATED`/`REUSED_ACTIVE`/`REUSED_INDEXED`/`REUSED_FAILED`) plus a
dynamic `202`/`200` status, and `DeletionActiveError`/`DeletionIncompleteError` map to a sanitized
`409` — never the raw internal exception, never `content_hash`/storage internals in the response
body. See `app/api/v1/routes/documents.py`'s upload route and `app/schemas/documents.py`.

Phase 2.8.5 status, as proven end-to-end (Backend E2E, local storage) by
`tests/e2e/backend/documents/upload/`:

- Hash persistence ✅
- Sequential duplicate reuse ✅
- Concurrent duplicate safety ✅
- No duplicate ingestion lifecycle ✅
- No duplicate logical vectors ✅
- Filename-independent exact-byte identity ✅
- Failed-ingestion reuse ✅
- Deletion conflict behavior ✅
- Completed deletion re-upload ✅
- Local storage Backend E2E ✅

MinIO parity for this feature has not been separately re-run at the Backend E2E tier — dedup
decisions live above the storage provider, and Local/MinIO parity is already covered at the
storage-abstraction unit/integration tier (see "Storage Abstraction"). Request idempotency keys,
tenant-scoped deduplication, and backfilling existing pre-hash rows remain out of scope and are
not implemented.

PostgreSQL remains the source-of-truth for document lifecycle/metadata, storage identity, and
active versions; `FileStorage` (local disk or MinIO, per `FILE_STORAGE_PROVIDER`) holds the
original file content; Qdrant is a **derived** index, rebuildable at any time from the persisted
file + the active `EmbeddingIndexConfig` via re-index (below) — never itself the source of truth
for what a document "is." MinIO is object storage, not the lifecycle source of truth either —
`Document`'s Postgres row is what determines a document's lifecycle state.

### Backward compatibility for pre-migration documents

A `Document` row written before Phase 2.6/2.7 has `storage_key IS NULL`. Migration
`a3f9c7d2e1b5` backfills `storage_provider='local'` and `storage_key=stored_filename` for every
existing row at upgrade time — the pre-migration `LocalFileStorage` always wrote files flat under
its configured root, keyed by `stored_filename`, so that value is exactly the object key the new
`LocalFileStorage` needs to locate the same file; no file content is read and no data is moved
during the migration. `stored_path`/`stored_filename` are kept (not dropped, not renamed) for
this reason. Any code addressing a document's content calls
`app.storage.keys.resolve_document_storage_key(document)`, which returns `storage_key` if set,
else falls back to `stored_path` — so a pre-migration row remains fully readable without any
special-casing at the call site.

### Re-index (`app/services/indexing/reindex_service.py`)

**Build and activation are two separate operations (Phase 2.8.6, subtask 1) — never one combined
function.** This is the foundation for operator-controlled build-ahead migration: a document's
vectors can be built into a new target collection while the running process keeps serving that
document from its current collection, untouched, for as long as an operator wants. Scheduling,
concurrency, a worker, and any public API around these two primitives are later subtasks — this
phase only establishes the safe split itself.

`build_reindex_target(document, session, settings, file_storage, target_config, *,
target_chunk_size, target_chunk_overlap) -> ReindexBuildResult` re-derives a document's vectors
from its already-persisted stored content under an explicit, caller-pinned `target_config` — no
new upload required. Flow: derive a target-scoped `Settings` copy via `build_settings_for_target()`
-> validate that those derived settings actually reproduce `target_config` exactly (raising
`TargetConfigurationMismatchError` before any storage read otherwise) -> re-extract
(`DocumentTextExtractor`) -> re-chunk (`DocumentChunker`, the explicit pinned `target_chunk_size`/
`target_chunk_overlap` — never live settings, never the chunking-version label alone) -> re-embed
(the target's own provider/model, resolved via the derived settings) -> validate real vector
dimensions -> `ensure_active_collection()` -> upsert into the target collection. **It never calls
`mark_document_indexed()`, never modifies `Document.collection_name`/`embedding_*`/
`chunking_version`/`indexed_at`, and never reads, deletes, or schedules cleanup for the document's
previous (currently-serving) collection.** A successful build proves only that the target was
built — not that it is active. Idempotent: point IDs are derived identically to the initial-ingest
path (`app.services.ingestion.worker.to_vector_point`), so re-running against the same target
overwrites rather than duplicates. Any extraction/chunking/embedding/validation/vector-store-write
failure propagates with the document's metadata completely untouched, since this function never
commits anything about the `Document` row at all.

`build_settings_for_target()` explicitly overrides every field the build actually reads —
`embedding_provider`, both `embedding_model` **and** `ollama_embedding_model` (kept identical,
since `OllamaEmbeddingProvider` reads `ollama_embedding_model` directly while
`resolved_embedding_model` prefers `embedding_model` — letting these diverge would mean generating
embeddings under one model while writing them into a collection whose name claims another),
`vector_size`, `embedding_version`, `chunking_version`, `qdrant_collection_name`, and the explicit
numeric `chunk_size`/`chunk_overlap` (not part of `EmbeddingIndexConfig`, since chunking-version is
only a label — these must be supplied explicitly by the caller) — via `Settings.model_copy()`,
never mutating the base `Settings` instance passed in.

`activate_reindexed_document(document, session, target_config) -> ReindexActivationResult` is the
separate, later operation that actually switches a document's serving identity: it calls
`mark_document_indexed()` for `target_config`, and — in the exact same commit, never a second one
—persists a fresh `VectorCleanupJob` (`PENDING`, `attempts=0`) for whatever collection the document
is leaving behind, if any. It never deletes Qdrant vectors itself; actual removal of the vacated
collection is left entirely to `cleanup_job_service.retry_cleanup_job()`, run independently and
later. Idempotent: calling it again once the document already carries `target_config`'s identity
is a no-op and creates no cleanup job. A commit failure rolls back both the metadata switch and
the cleanup-job row together (`session.rollback()` + `session.expire(document)`), so a crash can
never leave a document "activated" without its corresponding cleanup obligation durably recorded.

Neither primitive is wired into a worker, script, or public API yet — that is deliberately out of
scope for this subtask; only later Phase 2.8.6 subtasks add scheduling, concurrency guarantees, and
exposure.

### Re-index job scheduling (`app/services/indexing/reindex_scheduling_service.py`)

`ReindexJob` (`app/models/reindex_job.py`; migration `a8685da857f3`) is the durable, append-only
record of one re-index build attempt — mirroring `IngestionJob`/`DocumentDeletionJob`'s lifecycle
style exactly: a `FAILED` row is never reset or reused, retrying always inserts a brand-new
`PENDING` row, and at most one `PENDING`/`PROCESSING` ("active") row may exist per document at a
time, enforced by the partial unique index `ix_reindex_jobs_one_active_per_document` — the database,
not application logic alone. `target_collection_name` is a mandatory foreign key into
`IndexCollection` (never a duplicated copy of the target's provider/model/dimension/version
identity); `target_chunk_size`/`target_chunk_overlap` are mandatory, pinned numeric values —
`EmbeddingIndexConfig` only carries the `chunking_version` label, never these numbers, so a job's
build snapshot would be irreproducible without persisting them explicitly on every row.

`schedule_reindex(session, document, vector_store, target_config, *, target_chunk_size,
target_chunk_overlap)` decides whether a build may be scheduled, in this order: a document with
`collection_name IS NULL` (never successfully indexed) is `INELIGIBLE_NEVER_INDEXED` — re-indexing
is never a second initial-ingestion recovery mechanism, that stays exclusively
`retry_service.py`'s concern; already matching the target is `ALREADY_CURRENT`; an existing active
`ReindexJob` is `ALREADY_ACTIVE` (the existing job is returned, never duplicated); an active
(`PENDING`/`PROCESSING`) `IngestionJob` is `INGESTION_ACTIVE`; the latest `DocumentDeletionJob`
being `PENDING`/`PROCESSING`/`PARTIALLY_FAILED`/`COMPLETED` is `DELETION_ACTIVE`/
`DELETION_INCOMPLETE`/`DOCUMENT_DELETED` respectively; otherwise `ensure_active_collection()`
persists the target `IndexCollection` row and one `PENDING` `ReindexJob` is inserted and committed
-> `CREATED`. This subtask schedules only — no build, activation, Qdrant write, or object-storage
read happens here.

Two sessions may both pass the active-job check before either commits; the partial unique index is
the actual guarantee. The losing insert's `IntegrityError` is classified via the PostgreSQL
diagnostic `constraint_name` (never message-text matching, mirroring
`dedup_service.is_content_hash_violation()`'s exact approach) to confirm it is specifically
`ix_reindex_jobs_one_active_per_document`; any other integrity error is re-raised unchanged. On a
confirmed race, the loser rolls back and reloads the winning job, returning `ALREADY_ACTIVE` — or
raises `MissingActiveReindexJobAfterRaceError` in the should-be-unreachable case where no winner
can be reloaded.

**Active re-index blocks document deletion.** `deletion_service.request_document_deletion()`
rejects with `REINDEX_ACTIVE` (`409`) while an active `ReindexJob` exists for the document,
symmetric with its existing `INGESTION_ACTIVE` check — deletion must never race an in-flight
re-index build. Full-deletion tracking of a completed-but-not-yet-activated target build — so
`delete_all_tracked_document_vectors()` can prove and clean both the serving and target collections
for a document deleted mid-build — is implemented too: it also resolves every distinct
`target_collection_name` from a `COMPLETED` `ReindexJob` for the document (via
`get_completed_reindex_target_collections()`), never `PENDING`/`PROCESSING`/`FAILED` ones. See
"Two separate vector-deletion operations, deliberately not one" above.

### Re-index worker (`app/services/indexing/reindex_worker.py`)

**Build-only — the worker never activates anything.** `ReindexWorker.process_next_job(session,
settings)` claims one `PENDING` `ReindexJob` (`SELECT ... FOR UPDATE SKIP LOCKED`, mirroring
`IngestionWorker`/`DocumentDeletionWorker` exactly, ordered `created_at ASC, id ASC`), commits the
`PENDING -> PROCESSING` transition before any external I/O, reconstructs the job's pinned target
`EmbeddingIndexConfig` from its `IndexCollection` foreign key plus its own persisted
`target_chunk_size`/`target_chunk_overlap`, and delegates entirely to
`reindex_service.build_reindex_target()` — the worker never duplicates
extraction/chunking/embedding/validation/collection/upsert logic itself. `ReindexJob.status ==
COMPLETED` means only **"the pinned target build succeeded"** — never "the target is active or
serving." `Document.collection_name`/`embedding_*`/`chunking_version`/`indexed_at` are never
touched by this worker, no `VectorCleanupJob` is ever created by it, and no vector is ever deleted
from any collection by it. Activation (`reindex_service.activate_reindexed_document()`, added in
subtask 1) remains completely unwired into any runtime code as of this subtask.

`IndexCollection` does not itself persist the `collection_prefix` used to derive its
`collection_name` (only the fully joined name is a column) — the worker sources it from its own
base `Settings.qdrant_collection_name` (a platform-wide constant that should never legitimately
differ from what was used at scheduling time) and then verifies the reconstructed config's
`collection_name` matches the job's pinned `target_collection_name` exactly before building,
failing the job cleanly rather than silently building into the wrong collection if it does not.

**Defense in depth against deletion races.** `schedule_reindex()` already refuses to create a new
`ReindexJob` while a document's deletion is active/incomplete/completed, but the worker re-checks
the same condition immediately before building anyway, against the residual window between
scheduling and a worker actually claiming the job. A blocked build is recorded as
`ReindexJob.status = FAILED` (the existing status vocabulary — no new database status is
introduced for this) with a stable internal error message; the worker's own return value
additionally distinguishes this case (`ReindexWorkerOutcome.SKIPPED_DELETED`) from a genuine build
exception (`FAILED`) for the caller's benefit, without any corresponding DB-level distinction.

**Transaction boundaries.** Claiming is its own committed transaction before any external I/O; no
row lock is held across storage reads, extraction, chunking, embedding, or the Qdrant upsert.
Resolving the terminal status is a second, separate commit. The failure path always rolls back
first, then reloads the job strictly by its captured scalar `id` — never by re-accessing the
original (possibly now-expired) ORM object's attributes, which is exactly the `MissingGreenlet`
defect found and fixed in subtask 2's PostgreSQL verification.

**No stale-`PROCESSING` recovery yet.** If a worker process crashes between claiming a job and
marking it terminal, that `ReindexJob` row remains `PROCESSING` indefinitely in this subtask —
there is no automatic reset back to `PENDING`. This mirrors `IngestionJob`'s own
`recover_stale_ingestion_jobs()` precedent (script-only, never HTTP-triggered, never run by `make
verify`/CI) — a future task would extend that same convention to `ReindexJob`, not invent a new
mechanism; this subtask does not register or extend it.

### Re-index activation (`app/services/indexing/reindex_activation.py`)

**Build completion and activation are separate, independently-durable states.** `ReindexJob.status
== COMPLETED` (set by the worker above) means only "the pinned target build succeeded" — it says
nothing about whether the document is actually being served from that target. Activation is the
explicit, separate operation that performs the cutover: `activate_reindexed_document(session,
reindex_job_id) -> ReindexActivationResult` locks the `ReindexJob` and its `Document`
(`SELECT ... FOR UPDATE`, not `SKIP LOCKED` — a caller intentionally targeting one specific job
should wait for a concurrent activator rather than silently skip), re-validates every precondition
against current state (job exists / is `COMPLETED` / is not already activated; document exists and
its deletion lifecycle isn't active/incomplete/completed; the document's *current*
`collection_name` still equals the job's pinned `source_collection_name`; the target
`IndexCollection` row still exists), and — in that same transaction — atomically switches
`Document.collection_name`/`embedding_provider`/`embedding_model`/`embedding_dimension`/
`embedding_version`/`chunking_version`/`indexed_at` to the target's own persisted `IndexCollection`
columns, creates one `VectorCleanupJob` for the vacated source collection, and sets
`ReindexJob.activated_at` (added by this subtask, alongside `ReindexJob.source_collection_name` —
the document's serving collection *as captured by `schedule_reindex()` at scheduling time*, used
for the staleness check above). The document cutover and the cleanup-job creation share one commit
— activation never durably exposes a document already pointing at the target with no cleanup job
yet recorded for the source, or vice versa.

**Never rebuilds, never touches Qdrant, never consults live Settings.** Unlike the worker, this
module reconstructs the target's embedding/chunking identity purely from the target
`IndexCollection` row's own columns — never from a reconstructed `EmbeddingIndexConfig` (which
would require a `collection_prefix` that isn't itself a persisted column) and never from live
`Settings`, so a stale `QDRANT_COLLECTION_NAME` on the activating process can never corrupt a
cutover. It never calls `build_reindex_target()`, never issues a Qdrant write or delete, and never
retires or otherwise mutates the target `IndexCollection`'s own status.

**Old vectors remain present after activation; cleanup execution is asynchronous and not part of
this subtask.** The `VectorCleanupJob` row created here is picked up by the existing
`cleanup_job_service.retry_cleanup_job()` machinery on its own schedule — activation only records
the obligation, it never executes it inline. A document therefore has live, searchable vectors in
*both* its old and new collections for an indeterminate window after activation, by design.

**Idempotent via row locking, not a new unique constraint.** Two concurrent activation attempts
against the *same* `ReindexJob` id serialize on that row's `FOR UPDATE` lock; the second caller
proceeds only after the first commits, then observes `activated_at` already set and returns
`ALREADY_ACTIVATED` with no further mutation and no duplicate `VectorCleanupJob`. This differs from
`ix_reindex_jobs_one_active_per_document` (a multi-row insert race) because activation contention
is always over one already-existing row.

**Out of scope for this subtask (all remain future work):** no public HTTP API is exposed for
activation; no batch/campaign orchestration across multiple documents or jobs; no collection
retirement; no automatic write-freeze or process-restart coordination around a cutover (that
remains an external operational concern); no automated end-to-end migration workflow — build,
schedule, and activate remain three separately-invoked operations, not one automated pipeline.

### Zero-chunk behavior

A document whose extracted text produces zero chunks (e.g. genuinely empty content) is still a
valid build — `build_reindex_target()` reports `ReindexBuildOutcome.BUILT_EMPTY` and writes zero
vectors, never failing merely because there was nothing to embed. `IngestionWorker`'s default
processing step applies the equivalent behavior on the initial-ingest path (marking the document
indexed with zero vectors) — no false claim that searchable content exists, and predictable
no-results behavior at retrieval time.

### Legacy-vector cleanup (`VectorCleanupJob`)

`VectorCleanupJob` (`app/models/vector_cleanup_job.py`; migration
`alembic/versions/1c2d9f3a7b4e_...py`) durably tracks a legacy collection's vectors still needing
deletion after a document's serving identity is switched to a new collection, independently of
whether the document itself is still considered stale relative to any other configuration — so a
cleanup obligation is never silently lost or conflated with build/activation failure, and stays
discoverable/retryable indefinitely. One row per `(document_id, collection_name)`, with `status`
(`pending`/`failed`/`completed`), `attempts`, `last_error`, `created_at`, `completed_at`. Multiple
pending rows for the same document (different historical collections) are supported and never
overwrite each other.

`app/services/indexing/cleanup_job_service.py` provides `get_pending_cleanup_jobs()` (every
`pending`/`failed` row, optionally scoped to one document) and `retry_cleanup_job()` (retries the
delete; marks `completed` on success or increments `attempts`/records `last_error` and stays
`failed` otherwise — idempotent, since retrying a delete against already-empty vectors is a
harmless no-op). `create_cleanup_job()` remains available as a standalone, independently-committing
helper for other callers, but `activate_reindexed_document()` does not use it — a cleanup job
created as part of activation must share activation's own single commit with the metadata switch,
so `activate_reindexed_document()` constructs and `session.add()`s a `VectorCleanupJob` row
directly instead.

**Two separate vector-deletion operations, deliberately not one.**
`delete_current_document_vectors(document, vector_store)` targets only the document's
currently-tracked collection (`document.collection_name`) — it never consults
`VectorCleanupJob` or `ReindexJob` (no `session` parameter at all), so it is explicitly a
*partial* operation, valid only for call sites that provably have no historical cleanup to check
(e.g. a document never re-indexed) or that intentionally want the narrower scope (rollback,
current-index repair). `delete_all_tracked_document_vectors(document, vector_store, session)` is
the *full* operation — `session` is mandatory, since a full deletion requires a PostgreSQL lookup
of every collection the document's vectors could still exist in. Its tracked-collection set is
resolved from **three** sources: (1) `document.collection_name`, when non-null; (2) every distinct
historical collection from a `pending`/`failed` `VectorCleanupJob` for the document (see
`get_pending_cleanup_jobs()`); (3) every distinct `target_collection_name` from a **COMPLETED**
`ReindexJob` for the document (see `get_completed_reindex_target_collections()`, Phase 2.8.6,
subtask 3) — durable proof that a build-ahead re-index target may already hold a full vector set
even though `Document.collection_name` still points at the serving collection, not yet activated.
`PENDING`/`PROCESSING`/`FAILED` re-index jobs never contribute a target collection: `PENDING` has
built nothing yet, `PROCESSING` may still be writing (and is already prevented from racing
deletion by the scheduling interlock — see "Re-index job scheduling" above), and `FAILED` is not
durable proof a complete vector set exists. All three sources are merged with deterministic
deduplication (a collection seen from more than one source is attempted exactly once), and —
critically — every resolved collection is attempted independently: one collection's delete
failing never stops, skips, or aborts attempts against the others, and never falls back to
active-only semantics. It returns a typed `VectorDeletionResult` (`document_id`,
`attempted_collections`, and one `CollectionVectorDeletionResult` per collection with
`succeeded`/`error`, plus a `fully_deleted` convenience property) rather than a bare `None`/`bool`,
precisely because a partial cleanup must never be reported — or silently treated by a caller — as
a complete document deletion. Calling it again after a partial failure safely retries every
tracked collection, not just the ones that failed previously (deletes are idempotent against
Qdrant). Any user-facing or lifecycle-level document deletion must call
`delete_all_tracked_document_vectors()` — `deletion_worker.py`'s full document deletion already
does. This function does not mutate `VectorCleanupJob`/`ReindexJob` bookkeeping, does not delete
the `Document` row, object-storage file, or ingestion job — those remain separate, deliberate
operations. Completed cleanup jobs are retained (audit trail), never auto-deleted.

`app/services/indexing/collection_registry.py` additionally provides `get_stale_documents()` (list every
document whose `collection_name` isn't the active one) and `retire_collection()`
(bookkeeping-only status flip, never deletes Qdrant data). Migrating to a new embedding/chunking
version is therefore: bump `EMBEDDING_VERSION`/`CHUNKING_VERSION` -> the next re-index run creates
the new collection -> `get_stale_documents()` finds what still needs re-indexing -> old
collections are never auto-deleted at startup; `retire_collection()` plus a manual Qdrant
collection delete is the explicit cleanup boundary for once a migration is known-successful. A
full admin migration UI is out of scope.

### Language detection (`app/rag/language.py`)

`LanguageDetector` (ABC) / `ScriptBasedLanguageDetector` (the only implementation) resolves a
question to `SupportedLanguage.HE` or `SupportedLanguage.EN` — deterministic, word-level
Hebrew/Latin script-dominance counting (not character-level, and not an ML model): each
whitespace/punctuation-split word is classified as Hebrew, Latin, or ignored (digits/
punctuation-only), and whichever script has more *words* wins. Word-level (not character-level)
classification is what keeps a handful of Latin-script technical identifiers (Kafka, Qdrant,
Kubernetes, LangChain) embedded in an otherwise-Hebrew sentence from outweighing the surrounding
natural-language Hebrew words, and vice versa. An exact tie, or no Hebrew/Latin words at all
(empty/punctuation/numbers-only), falls back to `DEFAULT_RESPONSE_LANGUAGE`. Neither engine calls
this directly — both reach it only through `PromptProvider`.

### PromptCatalog / PromptProvider / ResolvedPrompt (`app/rag/prompts/`)

`PromptType` (`grounded_answer`, `direct_answer`, `clarification`, `no_results`, `out_of_scope`)
x `SupportedLanguage` (`he`, `en`) -> `PromptCatalog`. The two generation-backed types
(grounded_answer/direct_answer) use **one English-authored governance instruction each — never
duplicated per language** (`get_shared_instructions()`): answer only from context, do not
fabricate, preserve quoted source text and `[S1]`/`[S2]` labels untranslated, never translate
code/API names/class names/filenames/commands/environment variables/error messages, state
explicitly when context is insufficient. An explicit, per-language response-language directive
(`get_response_language_directive()`) is appended — `"Respond directly and naturally in Hebrew
(he)."` / `"...in English (en)."` — never "answer in English and translate," never a claim that
the model "thinks in English." `get_system_text()` composes
`shared_instructions + "\n\n" + language_directive`. The three no-LLM-call types
(clarification/no_results/out_of_scope) remain naturally authored per language via
`get_response_text()`, since they bypass the LLM entirely and there is no
instruction-language-vs-output-language distinction to make for them.
`UnsupportedPromptLanguageError`/`UnsupportedPromptTypeError` fail explicitly for anything the
catalog has no content for. `PromptProvider.resolve(prompt_type, question)` detects the
question's language (via `LanguageDetector`) and returns a `ResolvedPrompt` (`prompt_type`,
`language`, `prompt_version` from `PROMPT_CATALOG_VERSION` — `v2` since this structural change —
and exactly one of `system_text`/`response_text`); for generation-backed types, `system_text`'s
two components are also exposed individually as `shared_instructions`/`language_directive`. This
is the single seam both engines call.

**Supersedes `app.rag.responses`** (removed): the previous English-only fixed-constants module
from the LangChain compatibility layer milestone is gone; `RagOrchestrator` and
`LangChainRagEngine` both import from `app.rag.prompts.provider` directly, never from each
other's implementation module.

**Phase 2.5 boundary**: this catalog is a flat, hardcoded he/en dict — no persistence, no
runtime-editable prompts, no additional languages, no language detection beyond
script-dominance. A future milestone may introduce a database-backed, runtime-editable prompt
system if the platform ever needs more languages or non-developer prompt edits; do not add that
speculative machinery ahead of an actual need.

### Engine integration

Both `RagOrchestrator.stream_answer()` and `LangChainRagEngine.stream_answer()` now: resolve
`CLARIFICATION_NEEDED`/`OUT_OF_SCOPE` via `PromptProvider.resolve(..., question).response_text`
(no retrieval, no LLM call); resolve `NEEDS_RETRIEVAL`'s system instruction via
`PromptProvider.resolve(PromptType.GROUNDED_ANSWER, question).system_text`, combined with
`RagPromptBuilder`'s unchanged context/ranking/labeling; short-circuit to a fixed `no_results`
message (also via `PromptProvider`, no LLM call) when `RagPromptBuilder` found nothing
attributable; and resolve `DIRECT_LLM`'s system instruction via
`PromptProvider.resolve(PromptType.DIRECT_ANSWER, question).system_text`. `LangChainRagEngine`
builds its `ChatPromptValue` from the resolved `system_text` instead of a hardcoded English
string — everything else about its LangChain `Runnable` composition, brace-injection safety, and
provider adapters is unchanged from the LangChain compatibility layer milestone.

### Multilingual citation behavior

Source titles/filenames, quoted source text, and page/sheet metadata all come from
`RagPromptBuilder`/`VectorSearchResult` unchanged by this milestone — a Hebrew answer can cite an
English-titled source and vice versa; nothing here ever translates a citation or a document
title. Hebrew/Unicode text already survives JSON/SSE serialization (see "Streaming chat endpoint"
below) unchanged — this milestone adds no new serialization path.

### Multilingual embedding model selection

`OLLAMA_EMBEDDING_MODEL`'s Python-level default is **`bge-m3`** (1024-dim, BAAI's embedding model
supporting 100+ languages including Hebrew) — this is the actual default runtime configuration,
not merely a documented override; a fresh installation must run `ollama pull bge-m3` before
ingesting documents. `EMBEDDING_VERSION`'s default moved from `v1` to `v2` in the same change, so
an installation upgrading from Phase 2.5 (which defaulted to `nomic-embed-text`/768-dim/`v1`)
never silently reuses that now-incompatible collection: the active `EmbeddingIndexConfig`'s
`collection_name` changes, existing documents are reported stale by `is_document_stale()`, and
must go through a build (`build_reindex_target()`) and activation (`activate_reindexed_document()`)
— see "Re-index" below — to be searchable again under the new config. **The previous `nomic-embed-text`/`v1` collection and its
vectors are never deleted automatically** — `retire_collection()` remains a bookkeeping-only
status flip.

The legacy English-oriented `nomic-embed-text` (768-dim) model remains configurable — set
`EMBEDDING_MODEL=nomic-embed-text` + `VECTOR_SIZE=768` + `EMBEDDING_VERSION=v1` — but
`.env.example`/README no longer present it as the recommended default; it is documented only as
an explicit opt-out for installations that don't need Hebrew retrieval.

Automated tests (unit/integration/E2E) never depend on a real embedding model or download — they
use `MultilingualFakeEmbeddingProvider` (`tests/multilingual_fixtures.py`), a deterministic
bag-of-concepts hashing embedding with a small Hebrew/English synonym table (e.g. "vacation" and
"חופשה" hash to the same dimension), so equivalent cross-language concepts score genuinely
higher than an unrelated distractor — this demonstrates the retrieval *wiring* works
cross-language, not real multilingual model quality. See "Real multilingual runtime smoke" below
for an optional, manual, non-blocking check against a real `bge-m3` Ollama model; broader
recall/ranking evaluation on a larger corpus remains future work, and this project's automated
suites deliberately never pull or call a real embedding/LLM model (see "AI-provider policy in
tests" below).

## Streaming chat endpoint

`POST /api/v1/chat` (`app/api/v1/routes/chat.py`) is the first public endpoint that produces an
end-to-end RAG answer. It is deliberately a **thin route**: it validates the request, resolves a
`RagEngine` via a FastAPI dependency (`get_rag_engine()` in the route module, wrapping
`app.rag.engines.engine_factory.get_rag_engine()` — see "RAG Engine Compatibility Layer" above),
and formats `stream_answer()`'s output as Server-Sent Events — it contains no decision,
retrieval, or prompt-building logic, makes no direct call to any provider factory
(`get_embedding_provider()`/`get_vector_store()`/`get_llm_provider()`) or to
`RuleBasedRagDecider`/`RetrievalService`/`RagPromptBuilder`, and does not know or branch on
whether `CustomRagEngine` or `LangChainRagEngine` is configured — those all live inside whichever
`RagEngine` is resolved, which the route only consumes.

- **Request**: `ChatRequest` (`app/schemas/chat.py`) — `{"question": str}` only. A
  `field_validator` rejects an empty/whitespace-only `question`, which FastAPI turns into a
  standard `422` response before the route body ever runs. There is no `model` field: neither
  `RagEngine.stream_answer()` nor `LLMProvider` currently accepts a validated per-request
  model override, so none is exposed — adding one is future work, not a silent gap. The embedding
  model is never client-selectable, matching the existing `OLLAMA_EMBEDDING_MODEL`-is-fixed rule
  (see "LLM provider vs. model" above).
- **Response**: `StreamingResponse(..., media_type="text/event-stream")`, wrapping an async
  generator (`_stream_chat_events`) that iterates `RagOrchestrator.stream_answer(question)` and
  yields each event already SSE-formatted (`event: <name>\ndata: <JSON>\n\n`) — tokens are
  written to the response as the orchestrator produces them, never buffered into one full-text
  response first.
- **Event mapping**: `OrchestratorMetadata` → `metadata` (`decision.value`, `reason`,
  `retrieval_used`, `sources` — each `PromptSource` becomes `document_id`, `chunk_id`, `source`,
  `score`, plus `page_number`/`sheet_name` only when not `None`); `OrchestratorToken` → `token`
  (`text`); normal generator completion → `done` (`status: "completed"`), emitted exactly once
  and only after every token; an exception raised while consuming `stream_answer()` → `error`
  (`message` — a single fixed string, `status: "failed"`), and the stream ends there with no
  `done` event.
- **Error safety**: the `error` event's `message` is always the fixed string `"Failed to
  generate a response."` — the route never serializes an exception's `str()`, so no stack trace,
  prompt text, secret, credential, internal URL (e.g. `OLLAMA_BASE_URL`, `QDRANT_URL`), or raw
  provider response body can leak into a client-visible event.
- **No silent fallback**: because the route does not catch exceptions from individual pipeline
  stages separately — only from the orchestrator's combined stream — a `RetrievalService`
  failure can never be swallowed and silently answered via `DIRECT_LLM`, and a provider failure
  can never be silently retried against a different provider; both simply become the one
  `error` event.
- **Cancellation**: the route catches `Exception`, not `BaseException`, so
  `asyncio.CancelledError` (raised into the generator when a client disconnects mid-stream) is
  never caught as a normal failure — it propagates and lets the ASGI server clean up the
  connection normally, instead of the route trying to write an `error` event to an already-closed
  socket.

```bash
curl -N -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"What does the uploaded document say?"}'
```

## Test architecture

Tests are split into tiers by what they need to run, each with a different speed/fidelity
trade-off:

- **Unit tests** (`tests/*.py`, unmarked/default) — fakes and mocks only (fake sessions, fake
  providers, mocked `httpx` transports); no real database, no real Qdrant, no real network, no
  Docker. Fast (the whole suite runs in well under a second) and always run by `make test`/
  `make verify`. **One deliberate exception**: document-deletion unit tests live under
  `tests/unit/services/documents/` (`test_deletion_service.py`/`test_deletion_worker.py`),
  mirroring `app/services/documents/`'s module split — this is currently the only nested unit-test
  directory in the repository; every other unit test file remains flat directly under `tests/`.
  Extending this nested convention to other features is a deliberate future decision, not
  something this split silently generalizes.
- **Integration tests** (`tests/integration/*.py`, `@pytest.mark.integration`, auto-applied by
  `tests/integration/conftest.py`) — real, ephemeral Postgres and Qdrant containers started via
  [Testcontainers for Python](https://testcontainers-python.readthedocs.io/), never the
  repository's `docker-compose.yml`, on dynamically assigned ports with no persistent volumes.
  Covers behavior a mock/fake cannot faithfully represent: Alembic migrations against a real
  schema, `IngestionWorker`'s `SELECT ... FOR UPDATE SKIP LOCKED` claim semantics under genuine
  Postgres transaction locking, and Qdrant's actual HTTP request/response contract. AI providers
  stay fake and deterministic even here — no real Ollama container, no model pulled — see
  "AI-provider policy" below. Run via `make test-integration`/`make verify-integration`, never as
  part of `make test`/`make verify`.
- **Backend E2E tests** (`tests/e2e/backend/*.py`, `@pytest.mark.e2e`, auto-applied by
  `tests/e2e/backend/conftest.py`) — exercises the complete backend user flow through real HTTP:
  document upload → real `IngestionWorker` processing (extraction, chunking, Qdrant upsert) →
  retrieval/orchestration → the streaming chat SSE endpoint, consumed incrementally so event
  order/timing is genuinely exercised rather than inspected as one buffered string. Runs the real
  FastAPI app behind a real ASGI HTTP client (`httpx.AsyncClient` + `ASGITransport`), against its
  own ephemeral Testcontainers-managed Postgres and Qdrant — never `docker-compose.yml`, never
  fixed ports, with an isolated database and Qdrant collection per test. AI providers stay fake
  and deterministic here too — no real Ollama container, no model pulled — see "AI-provider
  policy" below. Run via `make test-e2e-backend`/`make verify-e2e-backend`, never as part of
  `make test`/`make verify`, and not added to the pre-commit hook.
- **Frontend E2E tests** — future milestone; no frontend exists yet in this repository.
- **Real-AI smoke tests** — future milestone, kept deliberately separate from the unit,
  integration, and backend E2E suites: a small, manual/nightly suite that runs against a real
  Ollama container with real models pulled, to catch drift in actual model behavior/output shape
  without paying that cost (container pull time, model pull time, non-determinism) on every
  commit.

**Local development** (running the app, trying it end-to-end by hand) continues to use
`docker-compose.yml` exactly as before — nothing about that workflow changes. Tests, in any tier,
must never depend on `docker-compose.yml` being up or on any state it created; the integration and
backend E2E suites' fixtures (`tests/integration/conftest.py`, `tests/e2e/backend/conftest.py`)
start their own containers from scratch every session and guard against ever pointing at a
production `APP_ENV`/`DATABASE_URL`/`QDRANT_URL`.

The RAG engine compatibility layer's tests span all three tiers above rather than forming a
separate tier of their own: `tests/test_rag_engine_factory.py`/`test_custom_rag_engine.py`/
`test_langchain_rag_engine.py`/`test_langchain_adapters.py`/`test_prompt_provider_engine_parity.py`
are unit tests, `tests/integration/test_langchain_rag_engine_integration.py` is an integration
test (real ephemeral Qdrant, existing `VectorPoint` format, fake embeddings), and
`tests/e2e/backend/test_rag_engine_parity.py` runs the full backend E2E flow under both
`RAG_ENGINE=custom` and `RAG_ENGINE=langchain`, comparing source IDs/ranking/metadata/SSE
ordering/error/no-results behavior for equivalence (the generated answer text may legitimately
differ). `make test-rag-engines`/`make verify-rag-engines` run just these files across all three
tiers as a convenience — they are not a substitute for `make verify`/`verify-integration`/
`verify-e2e-backend`, which still cover everything including this layer.

The Phase 2.5 multilingual RAG foundation's tests follow the same span-all-three-tiers pattern:
`tests/test_embedding_config.py`/`tests/unit/services/indexing/test_collection_registry.py`/
`test_vector_deletion_service.py`/`test_cleanup_job_service.py`/`test_reindex_service.py`/
`tests/test_language_detector.py`/`test_prompt_catalog.py` (plus the shared
`test_prompt_provider_engine_parity.py` above) are unit tests,
`tests/integration/test_multilingual_indexing.py` is an integration test (real ephemeral
Postgres/Qdrant — indexing metadata persistence, dimension-mismatch rejection, staleness
detection, re-index, document-vector cleanup, mixed Hebrew/English round-trip), and
`tests/e2e/backend/test_multilingual_matrix.py` runs the full Hebrew/English/mixed-language
document-and-question matrix under both engines. Both suites use
`MultilingualFakeEmbeddingProvider`/fixtures from `tests/multilingual_fixtures.py` — never a real
embedding model. `make test-multilingual-rag`/`make verify-multilingual-rag` run just these files
as a convenience, same caveat as `verify-rag-engines` above.

MinIO storage coverage spans two tiers, plus one focused backend-E2E addition (there is no
MinIO-specific unit tier — `FileStorage`'s contract/local/factory/upload/ingestion-wiring unit
tests already cover the provider-neutral seams without a real object store). Real-adapter and
real-pipeline coverage: `tests/integration/test_minio_storage.py` exercises `MinioFileStorage`
directly against a real, ephemeral MinIO container (bucket init, save/read/delete/exists/metadata,
presigned URLs, not-found/error translation), and
`tests/integration/ingestion/test_worker_minio.py` runs the real `IngestionWorker` pipeline
(extraction → chunking → fake embeddings → real Qdrant upsert) reading content that only ever
lived in MinIO. Public-contract coverage: `tests/e2e/backend/test_minio_e2e.py` runs the same
upload → ingestion → retrieval → streaming chat flow as `test_upload_to_streaming_chat.py`, but
through the real HTTP boundary with `FILE_STORAGE_PROVIDER=minio` resolved via the app's actual
`Settings`/`create_file_storage()` dependency chain (never a hand-substituted storage instance),
under both `RAG_ENGINE=custom` and `RAG_ENGINE=langchain`. It additionally verifies the uploaded
object landed in MinIO (queried via `Document.storage_key`, read back through `MinioFileStorage`,
byte-compared) and that no MinIO implementation detail (bucket name, endpoint, credentials) leaks
into the public response. The ephemeral MinIO container itself is started by
`tests/support/minio_containers.py`, a small shared helper both `tests/integration/conftest.py`
and `tests/e2e/backend/conftest.py` call, instead of duplicating the `DockerContainer` setup in
each — the container is only started lazily when a test actually requests it, so the existing
local-storage E2E tests never pay for it. `make test-storage`/`make test-storage-integration`/
`make test-minio` and `make test-e2e-backend-minio`/`make verify-e2e-backend-minio` run these
tiers individually as a convenience, same caveat as `verify-rag-engines` above. None of this
covers browser/frontend E2E, real AWS S3/Cloudflare R2, orphan-object cleanup, document-lifecycle
deletion, or production storage benchmarking — see "What is intentionally not implemented yet".

### AI-provider policy in tests

No tier pulls or calls a real LLM/embedding model. Unit tests use hand-written fake provider
doubles (see e.g. `tests/test_retrieval_service.py`, `tests/test_rag_orchestrator.py`). The
integration suite's one end-to-end pipeline test
(`tests/integration/ingestion/test_worker_postgres.py`) runs the real `IngestionWorker` default
pipeline against real Postgres and real Qdrant, but with `get_embedding_provider` monkeypatched
to a small fixed-vector fake. The backend E2E suite goes one step further and exercises the real
HTTP/chat surface too, with `FakeEmbeddingProvider` (deterministic bag-of-words hashing, so a
query genuinely matches its relevant indexed chunks under Qdrant's real cosine search) and
`FakeStreamingLLMProvider`/`FakeFailingLLMProvider` (`tests/e2e/backend/fakes.py`) swapped in by
monkeypatching the provider-factory function each consuming module already imports — never a
branch on `APP_ENV` in production code. Real Ollama stays entirely outside all three suites,
reserved for the future real-AI smoke suite described above.

## Operational Health Contract

`app/api/routes/health.py` exposes four **unversioned** endpoints — `GET /health`,
`/health/live`, `/health/ready`, `/health/dependencies` — registered on `app` with **no
`/api/v1` prefix**. This is deliberate: business API versioning (`/api/v1`, and any future
`/api/v2`) is about the shape of request/response contracts for clients of the RAG features;
operational health is a different, version-independent contract consumed by infrastructure —
Kubernetes probes, load balancers, ArgoCD, monitoring/alerting — that must never need to change
just because the business API moved to a new version. Moving these endpoints under a versioned
prefix later would break every external prober pointed at them; see the standing rule in
[CLAUDE.md](CLAUDE.md).

**Why four separate endpoints, not one**: each answers a different operational question and is
polled at different rates by different consumers.

- **`GET /health`** — "is the process up." A static, zero-dependency summary
  (`status`/`service`/`version`), for a human or a very cheap uptime check.
- **`GET /health/live`** (liveness) — "is the process alive and not deadlocked." Never calls
  Postgres, Redis, Qdrant, or Ollama. This is what a Kubernetes `livenessProbe` should point at:
  if it ever returns non-200 or times out, the pod should be restarted — but a downstream
  dependency being temporarily down is *not* a reason to restart this process, so liveness must
  stay independent of every external service.
- **`GET /health/ready`** (readiness) — "can this instance actually serve traffic right now."
  Calls `app/services/platform_health.get_readiness_result()`, which runs every check and
  returns `200` only if every **required** check passes, else `503`. This is what a Kubernetes
  `readinessProbe` and a load balancer's health check should point at: `503` here means "stop
  routing traffic to this instance," not "restart it" — `live` can (and often will) stay `200`
  while `ready` is `503` (e.g. Qdrant is temporarily unreachable but the process itself is fine).
- **`GET /health/dependencies`** — the same checks as readiness, but always returns `200` with
  the full per-dependency detail in the body (`status` per check, `required` per check, a safe
  `detail` string on failure). Intended for monitoring/alerting dashboards and human debugging,
  not for gating traffic — that's what `/health/ready`'s HTTP status code is for.

**Dependency/readiness semantics** — `app/services/platform_health.py`:

| Check | Method | Required for readiness? |
|---|---|---|
| `postgres` | `SELECT 1` via a short-lived async engine | Yes |
| `redis` | `PING` via `redis.asyncio` | No — no application code path reads/writes Redis yet |
| `qdrant` | `GET /collections` (same reachability check `create_collection_if_not_exists` uses) | Yes |
| `ollama` | Reuses `OllamaClient.check_health()` (reachability) | Yes |
| `ollama_chat_model` | Same call, `chat_model_available` | Yes |
| `ollama_embedding_model` | Same call, `embedding_model_available` | Yes |
| `file_storage` | Local: writes+deletes a small probe object under the configured root. MinIO: `ensure_bucket()` (endpoint reachable, credentials accepted, bucket exists/creatable) | Yes |

Every check runs concurrently (`asyncio.gather`) with its own `CHECK_TIMEOUT_SECONDS` (3s)
timeout, wrapping `asyncio.timeout(...)` around the actual I/O — no automatic retries beyond
that timeout, and no check ever mutates or restarts the dependency it's probing (a `SELECT 1`, a
`PING`, a `GET`, nothing else). A failed check is never silently dropped: `run_all_checks()`
always returns one `DependencyCheckResult` per check, and both `/health/ready` and
`/health/dependencies` surface every result. `redis` is checked and reported everywhere (so
observability doesn't lose it) but is `required=False`, so a down Redis alone can never flip
readiness to `503` — reflecting that nothing in this codebase actually depends on it yet (see the
environment variable table below); this is a deliberate, documented choice, not an oversight, and
should be revisited the moment any code path starts using `REDIS_URL`.

Every `DependencyCheckResult`'s `detail` is a fixed, generic string per failure mode (e.g.
`"Postgres is unreachable."`) — none of the checks ever return a raw exception message, a
connection string, a credential, or a provider's raw response body to the client.

**Thin-controller route, aggregation in the service layer**: `app/api/routes/health.py`'s
`readiness`/`dependencies` handlers do only three things — resolve `Settings` via `Depends`, call
one function in `app/services/platform_health.py`, and apply the status code / return the body
that function already produced. All required-check filtering, failed-check calculation, overall
status calculation, and safe error-summary construction live in the service module as pure,
synchronous, directly-unit-testable functions:

- `build_readiness_result(checks) -> ReadinessResult` — `ReadinessResult` is a small dataclass
  (`response: ReadinessResponse`, `status_code: int`) so the route never computes `200`/`503`
  itself, it only copies a value the service already decided.
- `build_dependencies_response(checks) -> DependenciesResponse`
- `get_readiness_result(settings)`/`get_dependencies_response(settings)` — thin async wrappers
  that call `run_all_checks(settings)` then delegate to the two functions above; these are what
  the route actually calls.

This mirrors how `POST /api/v1/chat` (see "Streaming chat endpoint" above) stays a thin route
over `RagOrchestrator` — routes handle HTTP concerns (validation, dependency injection, status
codes, response shape) and delegate everything else to a service; see the standing rule in
[CLAUDE.md](CLAUDE.md).

**Future DevOps consumers** this contract is designed for (none wired up yet in this repository):
Kubernetes liveness/readiness probes, load balancer health checks, ArgoCD rollout health checks
(a rollout can gate on `/health/ready` before shifting traffic to a new revision),
monitoring/alerting systems polling `/health/dependencies` for per-dependency status, and a
future backend E2E suite's own startup check (poll `/health/ready` before running E2E tests
against a freshly started stack, instead of a fixed sleep).

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
- `minio:9000` (S3 API), `minio:9001` (console) — only reached when `FILE_STORAGE_PROVIDER=minio`

Only the ports needed for host-side debugging are published (`8000`, `5432`, `6379`, `6333`,
`11434`, `9000`, `9001`). In a production deployment, only `app`'s port would typically be
exposed.

```
host:8000 ──► app ──► postgres:5432
                 ├──► redis:6379
                 ├──► qdrant:6333
                 ├──► ollama:11434
                 └──► minio:9000 (only when FILE_STORAGE_PROVIDER=minio)
```

`minio` is a local/dev-only service — the app does not require it while
`FILE_STORAGE_PROVIDER` stays `local` (the default). Start it explicitly with
`docker compose up -d minio`, or switch `FILE_STORAGE_PROVIDER=minio` in the `app` service's
environment to use it. Credentials in `docker-compose.yml`/`.env.example` (`minioadmin`/
`minioadmin`) are for local development only — never reuse them anywhere else.

## Environment variables

Set via `docker-compose.yml` for containers, or `.env` (copy from `.env.example`) for local runs
outside Docker. `app/core/config.py` (`Settings`) is the single source of truth for defaults.

| Variable                  | Default                                                              | Notes |
|----------------------------|-----------------------------------------------------------------------|-------|
| `APP_ENV`                 | `local`                                                                | Read by application/provider configuration |
| `LOG_LEVEL`                | `INFO`                                                                 | Not yet wired to a logger |
| `DATABASE_URL`             | `postgresql+asyncpg://postgres:postgres@postgres:5432/rag_db`         | Async SQLAlchemy engine; stores `documents`/`ingestion_jobs`; also `SELECT 1`-checked by `GET /health/ready`/`/health/dependencies` |
| `REDIS_URL`                | `redis://redis:6379/0`                                                | Not yet consumed by any business code path; `PING`-checked (not required) by `GET /health/ready`/`/health/dependencies` |
| `QDRANT_URL`               | `http://qdrant:6333`                                                  | Used by `QdrantVectorStore` for collection/upsert/search; also checked by `GET /health/ready`/`/health/dependencies` |
| `OLLAMA_BASE_URL`          | `http://ollama:11434`                                                 | Used by `OllamaClient` for health/model checks (also reused by `GET /health/ready`/`/health/dependencies`) |
| `OLLAMA_CHAT_MODEL`        | `llama3.1`                                                             | Checked for availability; backward-compatible fallback for `LLM_MODEL` if unset |
| `OLLAMA_EMBEDDING_MODEL`   | `bge-m3`                                                     | Checked for availability; always used by `OllamaEmbeddingProvider` — fixed, not selectable via `LLM_MODEL` |
| `LLM_PROVIDER`             | `ollama`                                                               | Selects the `LLMProvider` implementation; `openai`/`gemini`/`anthropic` are recognized stubs |
| `LLM_MODEL`                | *(unset)*                                                              | Selects the model `OllamaLLMProvider` uses; falls back to `OLLAMA_CHAT_MODEL` if unset (see "LLM provider vs. model") |
| `EMBEDDING_PROVIDER`       | `ollama`                                                               | Selects the `EmbeddingProvider` implementation via the provider factory |
| `VECTOR_STORE_PROVIDER`    | `qdrant`                                                               | Selects the `VectorStore` implementation via the provider factory |
| `CHUNK_SIZE`               | `1000`                                                                 | Target chunk size in characters, used by `DocumentChunker` |
| `CHUNK_OVERLAP`            | `200`                                                                  | Overlap between consecutive chunks in characters, used by `DocumentChunker` |
| `QDRANT_COLLECTION_NAME`   | `documents`                                                            | The **prefix/namespace** `EmbeddingIndexConfig.collection_name` derives the real, versioned Qdrant collection name from (see "Multilingual RAG Foundation") — not a literal collection name by itself |
| `VECTOR_SIZE`              | `1024`                                                                  | Vector dimensionality — part of the active `EmbeddingIndexConfig`; must match the embedding provider's output size (`nomic-embed-text` produces 768-dim vectors; `bge-m3` produces 1024) |
| `RETRIEVAL_TOP_K`          | `5`                                                                    | Default number of results `RetrievalService.retrieve()` asks Qdrant for, when no explicit `limit` is passed |
| `RETRIEVAL_SCORE_THRESHOLD`| *(unset)*                                                              | Minimum Qdrant score a result must meet to be returned; unset/`null` disables score filtering |
| `RAG_ENGINE`               | `custom`                                                               | Selects the `RagEngine` implementation via `get_rag_engine()` (see "RAG Engine Compatibility Layer"); `langchain` is the only other recognized value — anything else raises `UnsupportedRagEngineError` |
| `EMBEDDING_MODEL`          | *(unset)*                                                              | Generic, provider-agnostic embedding model override; falls back to `OLLAMA_EMBEDDING_MODEL` if unset (same pattern as `LLM_MODEL`/`OLLAMA_CHAT_MODEL`) — part of the active `EmbeddingIndexConfig` |
| `EMBEDDING_VERSION`        | `v2`                                                                   | Part of the active `EmbeddingIndexConfig` — bump whenever the embedding model/dimension changes meaningfully, to roll onto a new Qdrant collection instead of silently mixing incompatible vectors |
| `CHUNKING_VERSION`         | `v1`                                                                   | Part of the active `EmbeddingIndexConfig` — bump whenever `CHUNK_SIZE`/`CHUNK_OVERLAP`/the chunking algorithm changes meaningfully |
| `DEFAULT_RESPONSE_LANGUAGE`| `en`                                                                   | Fallback language `ScriptBasedLanguageDetector` resolves to when a question has no Hebrew/Latin words at all, or an exact word-count tie; must be `he` or `en` |
| `PROMPT_CATALOG_VERSION`   | `v2`                                                                   | Stamped onto every `ResolvedPrompt.prompt_version` — see "Multilingual RAG Foundation" |
| `FILE_STORAGE_PROVIDER`    | `local`                                                                | Selects the `FileStorage` implementation via `create_file_storage()`; `minio` is the only other recognized value — anything else raises `StorageConfigurationError` |
| `LOCAL_STORAGE_ROOT`       | `storage/documents`                                                    | Root directory `LocalFileStorage` resolves object keys under; only read when `FILE_STORAGE_PROVIDER=local` |
| `MINIO_ENDPOINT`           | *(unset)*                                                              | `host:port` of the MinIO/S3-compatible endpoint; required when `FILE_STORAGE_PROVIDER=minio` |
| `MINIO_ACCESS_KEY`         | *(unset)*                                                              | Only read when `FILE_STORAGE_PROVIDER=minio` |
| `MINIO_SECRET_KEY`         | *(unset)*                                                              | Only read when `FILE_STORAGE_PROVIDER=minio`; never logged |
| `MINIO_BUCKET`             | *(unset)*                                                              | Bucket name; required when `FILE_STORAGE_PROVIDER=minio` |
| `MINIO_SECURE`             | `false`                                                                | Whether to use HTTPS against the MinIO endpoint |
| `MINIO_REGION`             | *(unset)*                                                              | Optional S3 region hint passed to the `minio` SDK client |
| `MINIO_PRESIGNED_URL_EXPIRY_SECONDS` | `3600`                                                       | Expiry for `MinioFileStorage.generate_download_url()`'s presigned GET URLs |
| `MINIO_CREATE_BUCKET_IF_MISSING` | `true`                                                           | Whether `ensure_bucket()` creates the configured bucket when missing, vs. failing readiness |
| `INGESTION_STALE_AFTER_SECONDS` | `900`                                                             | Seconds a `PROCESSING` `IngestionJob`'s row can go untouched before it's treated as stale by `retry_ingestion()`/`recover_stale_ingestion_jobs()` — an approximation, not a liveness proof (see "Ingestion retry and stale-job recovery") |
| `INGESTION_RECOVERY_BATCH_SIZE` | `50`                                                              | Maximum stale `PROCESSING` jobs `recover_stale_ingestion_jobs()` recovers per call/script run |

## Current boundaries

- `app/api/routes` — **unversioned** operational routes: `GET /health`, `/health/live`,
  `/health/ready`, `/health/dependencies` (see "Operational Health Contract" above). Registered
  on `app` with no prefix — never move these under `app/api/v1`.
- `app/api/v1/routes` — versioned business API routers: `GET /providers/ollama/health`,
  `POST /documents` plus the five read-only `GET /documents*` routes (see "Document upload and
  ingestion job skeleton" and "Document read APIs and original download (Phase 2.8.2)" above),
  `POST /documents/{document_id}/ingestion/retry` (see "Ingestion retry and stale-job recovery
  (Phase 2.8.3)" above), `DELETE /documents/{document_id}` / `GET .../deletion` (Phase 2.8.4),
  and `POST /chat` (see "Streaming chat endpoint" above) — `chat.py` is the only router that
  depends on `app/rag`. The duplicate versioned `GET /health` route was removed in the
  Phase 2.8.4 structural refactor; `app/api/routes/health.py`'s unversioned endpoints (above)
  are the only supported health API.
- `app/core` — configuration and cross-cutting concerns, plus `version.py` (`SERVICE_NAME`/
  `SERVICE_VERSION` — the single source of truth for both the FastAPI app's own metadata and the
  unversioned platform health responses).
- `app/db` — SQLAlchemy async engine/session setup.
- `app/models` — ORM models: `Document`, `IngestionJob`/`IngestionStatus` (see "Document upload
  and ingestion job skeleton" above), `IndexCollection`/`IndexCollectionStatus`
  (`app/models/index_collection.py`, see "Multilingual RAG Foundation" above), and
  `VectorCleanupJob`/`VectorCleanupStatus` (`app/models/vector_cleanup_job.py`, see "Legacy-vector
  cleanup" above). `Document` also carries `embedding_*`/`chunking_version`/`collection_name`/
  `indexed_at` columns.
- `app/schemas` — Pydantic request/response schemas.
- `app/storage` — the provider-neutral storage layer (see "Storage Abstraction" above):
  `FileStorage`/`StoredFile`/`FileMetadata` (`contract.py`), `LocalFileStorage`
  (`local_storage.py`), `MinioFileStorage` (`minio_storage.py`), `create_file_storage()`
  (`factory.py`), object-key generation/validation (`keys.py`), and the `StorageError` hierarchy
  (`errors.py`).
- `app/services` — business logic layer, split into three feature packages plus two flat
  cross-cutting modules (see "Documents service package"/CLAUDE.md's package map for the full
  ownership rationale):
  - `app/services/documents/` — `upload_service.py`'s `upload_document()` (saves content via the
    injected `FileStorage`, persists `Document`/`IngestionJob`); `query_service.py`, the
    read-only query/lifecycle-derivation layer backing the five `GET /documents*` routes — never
    writes to Postgres, object storage, or Qdrant, and never calls `FileStorage`;
    `download_service.py`, the original-content download path (the only module in this package
    that calls `FileStorage.read()`); `text_extractor.py`'s `DocumentTextExtractor`, which
    extracts text from a document's stored `.txt`/`.md`/`.pdf`/`.docx`/`.xlsx` content, read via
    the injected `FileStorage`; `chunker.py`'s `DocumentChunker`, which splits an
    `ExtractedDocument` into `DocumentChunk`s; and `deletion_service.py`/`deletion_worker.py`
    (Phase 2.8.4 scheduling/execution split — see "Documents service package" above).
  - `app/services/ingestion/` — `worker.py`'s `IngestionWorker`, which claims and resolves
    pending ingestion jobs — no public API — and whose default pipeline calls the
    embedding/vector-store providers (see "Chunk embedding and Qdrant indexing" above), the only
    place in this layer that does; `retry_service.py`'s `retry_ingestion()` and
    `stale_recovery_service.py`'s `recover_stale_ingestion_jobs()` (see "Ingestion retry and
    stale-job recovery (Phase 2.8.3)" above) — the write paths in this package that touch
    `IngestionJob`, strictly Postgres-only (never `FileStorage`, never a vector store); and
    `status.py`, the small shared constants/helpers both of those two modules require.
  - `app/services/indexing/` — `collection_registry.py`, the collection-safety and
    document-indexing-metadata service (see "Multilingual RAG Foundation" above) —
    `ensure_active_collection()`, `mark_document_indexed()`, `is_document_stale()`,
    `get_stale_documents()`, `retire_collection()`; `vector_deletion_service.py` —
    `delete_current_document_vectors()`, `delete_all_tracked_document_vectors()`;
    `cleanup_job_service.py` — `create_cleanup_job()`, `get_pending_cleanup_jobs()`,
    `retry_cleanup_job()`; and `reindex_service.py`'s `build_reindex_target()`/
    `activate_reindexed_document()`, the backend re-index build/activation capability.
  - `app/services/ollama_client.py` — `OllamaClient`, a thin async HTTP client scoped strictly to
    reachability and model-availability checks — it intentionally does not call generation or
    embedding endpoints.
  - `app/services/platform_health.py` — the dependency checks backing `GET /health/ready`/
    `/health/dependencies` (see "Operational Health Contract" above) — reuses `OllamaClient` for
    the Ollama check rather than duplicating it.
- `app/rag/retrieval_service.py` — `RetrievalService`, the internal read-side counterpart to
  ingestion's embed/upsert steps (see "Retrieval service" above). It is the second caller of
  `get_embedding_provider()`/`get_vector_store()` alongside `IngestionWorker`, and it never calls
  `LLMProvider`. Resolves the collection to search via
  `app.rag.embedding_config.get_active_embedding_config()`, never `QDRANT_COLLECTION_NAME`
  directly.
- `app/rag/embedding_config.py` — `EmbeddingIndexConfig`, `get_active_embedding_config()`,
  `InvalidEmbeddingIndexConfigError` (see "Multilingual RAG Foundation" above). The single source
  of the active indexing configuration for both `IngestionWorker` and `RetrievalService`.
- `app/rag/embedding_validation.py` — `validate_embeddings()`,
  `EmbeddingDimensionMismatchError`, `EmbeddingResultCountMismatchError` (see "Multilingual RAG
  Foundation" above). Called by `IngestionWorker`, `reindex_service.py`, and `RetrievalService`
  before any Qdrant write/search or document-indexed marking.
- `app/rag/language.py` — `LanguageDetector`, `ScriptBasedLanguageDetector`, `SupportedLanguage`
  (see "Multilingual RAG Foundation" above).
- `app/rag/prompts/` — `PromptType`, `ResolvedPrompt` (`types.py`), `PromptCatalog`
  (`catalog.py`), `PromptProvider` (`provider.py`) — see "Multilingual RAG Foundation" above.
  Supersedes the removed `app/rag/responses.py`.
- `app/rag/prompt_builder.py` — `RagPromptBuilder`, `BuiltRagPrompt`, `PromptSource` (see "RAG
  prompt builder" above). Pure and synchronous — it calls no provider at all (not even
  `get_embedding_provider()`/`get_vector_store()`), consuming only the `VectorSearchResult`s a
  caller already obtained from `RetrievalService`.
- `app/rag/providers` — abstract interfaces for embedding, LLM, and vector store providers, a
  `provider_factory.py` that resolves the configured implementation for each (see "Provider
  factory" above), and three concrete implementations:
  - `OllamaEmbeddingProvider` (`app/rag/providers/ollama_embedding_provider.py`) — calls
    `POST /api/embeddings` for `OLLAMA_EMBEDDING_MODEL` only.
  - `OllamaLLMProvider` (`app/rag/providers/ollama_llm_provider.py`) — calls
    `POST /api/generate` with `stream=true` for `Settings.resolved_llm_model`
    (`LLM_MODEL`, falling back to `OLLAMA_CHAT_MODEL`), implementing both abstract `LLMProvider`
    methods: `stream_generate(prompt) -> AsyncIterator[str]` (yields text chunks as Ollama
    streams them) and `generate(prompt) -> str` (joins the streamed chunks). Internal-only — no
    ingestion, no Qdrant writes, no public chat/SSE endpoint.
  - `QdrantVectorStore` (`app/rag/providers/qdrant_vector_store.py`) — calls Qdrant's HTTP API
    for collection create/upsert/search only (see "Vector store" above). Internal-only — no
    document upload, no chat/SSE endpoint, no full RAG flow; `IngestionWorker` (write side) and
    `RetrievalService` (read side) are its only callers so far.

  `LLMProvider` (`app/rag/providers/llm_provider.py`) declares both `generate(prompt) -> str` and
  `stream_generate(prompt) -> AsyncIterator[str]` as abstract methods — every implementation,
  including the future-provider stubs, must implement both.

  Three future-provider stubs also exist — `OpenAIProvider`, `GeminiProvider`,
  `AnthropicProvider` (`app/rag/providers/{openai,gemini,anthropic}_provider.py`) — which
  implement `LLMProvider` but always raise `ProviderNotImplementedError` from both `generate()`
  and `stream_generate()` (see "Future LLM provider stubs" above).

  `OllamaClient` (health checks) is deliberately kept separate from these provider interfaces so
  health checks don't get entangled with the generation/embedding/storage contracts.
- `app/rag/decision.py` — the RAG decision layer (see "RAG decision layer" above): `RagDecision`,
  `DecisionResult`, `RuleBasedRagDecider`. Separate from `app/rag/providers` since it doesn't call
  any provider itself — it only classifies a question.
- `app/rag/orchestrator.py` — `RagOrchestrator`, `OrchestratorMetadata`, `OrchestratorToken` (see
  "RAG orchestrator" above). The only component that composes the decision layer, retrieval
  service, prompt builder, and LLM provider together — no other module in `app/rag` calls more
  than one of them.
- `app/rag/engine.py` — the `RagEngine` abstraction (see "RAG Engine Compatibility Layer" above).
- `app/rag/engines/` — concrete `RagEngine` implementations: `custom_engine.py`
  (`CustomRagEngine`, the default, wrapping `RagOrchestrator`), `langchain_engine.py`
  (`LangChainRagEngine`, optional), `langchain_adapters.py` (`ProviderBackedLLM`/
  `ProviderBackedEmbeddings`/`ProviderBackedRetriever`), and `engine_factory.py`
  (`get_rag_engine()`, `UnsupportedRagEngineError`).
- `app/workers` — background job placeholders.
- `tests/integration/` — the Testcontainers-based integration suite (see "Test architecture"
  above): `conftest.py` (ephemeral Postgres/Qdrant/MinIO fixtures, the production-environment
  guard, the Alembic-migration helpers), `test_alembic_migrations.py`,
  `test_worker_postgres.py`, `test_qdrant_vector_store_integration.py`,
  `test_langchain_rag_engine_integration.py`, `test_minio_storage.py`,
  `test_worker_minio.py`. Entirely separate from `tests/*.py` (unit tests); auto-marked
  `@pytest.mark.integration` and excluded from `make test`/`make verify`.
- `tests/support/` — small, non-test helper modules reused by more than one test tier;
  `minio_containers.py` holds the one ephemeral-MinIO-container startup routine
  `tests/integration/conftest.py` and `tests/e2e/backend/conftest.py` both call, instead of each
  duplicating the `DockerContainer` setup.

## What is intentionally not implemented yet

- A standalone public retrieval endpoint — `RetrievalService` is only reachable indirectly, via
  `POST /api/v1/chat`'s `NEEDS_RETRIEVAL` path; there's no endpoint that returns raw
  `VectorSearchResult`s on their own
- Persisting extracted text or chunks in Postgres — `DocumentChunker`'s output is only persisted
  as vectors in Qdrant (via the embedding/upsert step); there's no relational table for chunk text
- Anything that continuously runs `IngestionWorker.process_next_job()` in a loop (no scheduler
  or long-running process invokes it yet — it's called directly, one job at a time)
- A public API endpoint for embeddings, vector store, chunking, prompt-building, or
  decision-layer operations on their own (all internal-only; only reachable indirectly through
  `POST /api/v1/chat`)
- An LLM-based (as opposed to rule-based) question router
- Conversation memory / multi-turn context (in prompt building or the orchestrator)
- A client-selectable model override on `POST /api/v1/chat` — `ChatRequest` has no `model` field
- A document-deletion endpoint, an orphan-object cleanup worker, hash-based content
  deduplication, a bulk re-index endpoint, or a real scheduler deployment for stale-job recovery
  (`scripts/recover_stale_ingestion_jobs.py` is manual-trigger only — see "Ingestion retry and
  stale-job recovery (Phase 2.8.3)" above) — those remain future milestones. Retry
  (`POST /documents/{document_id}/ingestion/retry`, Phase 2.8.3) is now implemented; no route
  calls `FileStorage.delete()` or mutates a `Document` row (only `IngestionJob` rows are ever
  written by retry/recovery)
- AWS S3 or Cloudflare R2 as a `FileStorage` implementation, and presigned multipart uploads —
  `MinioFileStorage` covers MinIO/S3-compatible storage generically, but a dedicated S3/R2
  implementation and multipart upload support are future work, not implemented in this phase
- Frontend E2E tests — no frontend exists yet in this repository
- A real-Ollama smoke suite — deliberately kept separate/manual/nightly, not part of the default
  integration run (see "Test architecture" above)
- LangGraph — the LangChain compatibility layer uses only `langchain-core`'s
  `Runnable`/prompt/retriever primitives; see "Why LangGraph is intentionally deferred" under "RAG
  Engine Compatibility Layer" above
- Agents, tool calling, and any LangChain/LangGraph agent packages — neither `RagEngine`
  implementation exposes tool use; both are a fixed decide-then-generate flow
- A LangChain-specific ingestion path, a second Qdrant SDK/collection, or a client-selectable
  `RAG_ENGINE` override — `RAG_ENGINE` is a server-side deployment setting, never a per-request
  parameter
- Kubernetes manifests, Helm charts, ArgoCD Application/Rollout resources, or any
  monitoring/alerting configuration that actually consumes the new `/health/*` endpoints — this
  milestone only establishes the operational health *contract* those future consumers would use
- Auth, rate limiting, an observability/logging pipeline, and Redis actually being used for
  anything (it is only `PING`-checked, not read from or written to, by any code path)

These land in later milestones once the infrastructure is confirmed stable.
