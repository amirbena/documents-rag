# documents-rag

## Goal

Infrastructure scaffold for a local, self-hosted RAG (Retrieval-Augmented Generation) platform.
This milestone contains only the project skeleton: API, config, database wiring, Docker Compose,
and placeholder provider interfaces. No ingestion, embedding, or chat logic is implemented yet.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system overview, service topology, and
environment variable reference.

## Tech stack

- Python 3.11+, FastAPI, Pydantic v2
- SQLAlchemy 2.x (async) + Alembic
- PostgreSQL, Redis, Qdrant
- Ollama (local LLM + embeddings): `llama3.1` for chat, `bge-m3` for multilingual embeddings
- Docker Compose
- pytest, ruff, mypy

## Prerequisites

- Python 3.11+
- Docker
- Docker Compose (bundled with modern Docker Desktop/Docker Engine — check with
  `docker compose version`)
- Git
- [GitHub CLI](https://cli.github.com/) (`gh`) — only needed for the repository/PR workflow
  (opening/reviewing pull requests), not for running the app itself

## Local setup

First-time onboarding, in order — later sections below go into more detail on each step:

1. **Verify prerequisites**: `python3 --version`, `docker --version`, `docker compose version`,
   `git --version` (and `gh --version` / `gh auth status` if you'll be opening PRs).
2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```
4. **Copy the environment file**:
   ```bash
   cp .env.example .env
   ```
5. **Start Docker Compose** (see "Running with Docker Compose" below for the full walkthrough):
   ```bash
   docker compose up --build
   ```
6. **Run Alembic migrations** — Docker Compose starts Postgres but does not apply migrations
   automatically (see "Database migrations" below):
   ```bash
   docker compose exec app alembic upgrade head
   ```
7. **Verify app health**:
   ```bash
   curl http://localhost:8000/health
   # {"status":"ok","service":"documents-rag","version":"0.1.0"}
   ```
   (see "Platform health and readiness" below for the full unversioned health/liveness/readiness
   contract)
8. **Pull the required Ollama models** (see "Running with Docker Compose" below):
   ```bash
   docker compose exec ollama ollama pull llama3.1
   docker compose exec ollama ollama pull bge-m3
   ```
9. **Verify Ollama health**:
   ```bash
   curl http://localhost:8000/api/v1/providers/ollama/health
   ```
10. **Install the Git pre-commit hook** (see "Pre-commit verification" below):
    ```bash
    ./scripts/install-git-hooks.sh
    ```
11. **Run the full verification suite**:
    ```bash
    make verify
    ```

Running the API directly (without Docker) requires reachable Postgres/Redis/Qdrant/Ollama —
easiest to get via `docker compose up postgres redis qdrant ollama` and point `.env` at
`localhost` instead of the service names.

## Running locally

**`python app/main.py` does not start the server.** `app/main.py` only defines the FastAPI `app`
object — running it as a script just imports the module (which builds `app` and exits) and does
nothing else. There is no `if __name__ == "__main__":` block that calls `uvicorn.run(...)`, so
this command produces no running server and no error, which is easy to mistake for "it worked."

The recommended way to run everything (app + Postgres + Redis + Qdrant + Ollama) is:

```bash
docker compose up --build
```

Verify it's up with:

```bash
curl http://localhost:8000/health
# {"status":"ok","service":"documents-rag","version":"0.1.0"}
```

See "Running with Docker Compose" below for the full walkthrough (pulling Ollama models,
checking Ollama health, etc.).

**Optional: running the app process only, without Docker**, once you've completed
[Local setup](#local-setup) above and have Postgres/Redis/Qdrant/Ollama reachable some other way
(e.g. `docker compose up postgres redis qdrant ollama` with `.env` pointed at `localhost`):

```bash
uvicorn app.main:app --reload
```

### PyCharm Run Configuration

- **Prefer Docker Compose for full-stack local development** — PyCharm's Docker Compose run
  configuration (or just running `docker compose up --build` in the terminal) covers the app and
  all its dependencies together.
- **For an app-only run** (no Docker, dependencies reachable separately — see above), create a
  "Python" run configuration with:
  - **Module name:** `uvicorn` (not "Script path")
  - **Parameters:** `app.main:app --reload`
  - **Working directory:** repository root
  - **Python interpreter:** `.venv/bin/python`

## Running with Docker Compose

```bash
docker compose up --build
```

(If you haven't already, copy `.env.example` to `.env` first — see "Local setup" above.)

This starts `app`, `postgres`, `redis`, `qdrant`, and `ollama`. The app is available at
http://localhost:8000, with health check at `GET /health`. Verified working end-to-end: all
five containers start, the health endpoint responds `{"status":"ok","service":"documents-rag","version":"0.1.0"}`
from the host, and the `app` container can reach `postgres:5432`, `redis:6379`, `qdrant:6333`,
and `ollama:11434` over the internal Compose network.

Once the `app`/`postgres` containers are up, run the Alembic migrations — see "Database
migrations" below — before pulling Ollama models or testing document upload.

To pull the required Ollama models after the `ollama` service is up:

```bash
docker compose exec ollama ollama pull llama3.1
docker compose exec ollama ollama pull bge-m3
```

Check whether Ollama is reachable and those models are pulled via:

```bash
curl http://localhost:8000/api/v1/providers/ollama/health
```

Returns `200` when Ollama is reachable and both models are available, or `503` (with the same
JSON body showing which check failed) otherwise.

`OllamaEmbeddingProvider` (`app/rag/providers/ollama_embedding_provider.py`) embeds text via
Ollama's `POST /api/embeddings` with `OLLAMA_EMBEDDING_MODEL`. It's an internal provider only —
no API endpoint exposes it yet, and it doesn't call Ollama's generation endpoint or touch Qdrant.

`OllamaLLMProvider` (`app/rag/providers/ollama_llm_provider.py`) streams completions from
Ollama's `POST /api/generate` (`stream=true`) with the configured chat model, via
`stream_generate(prompt) -> AsyncIterator[str]` (yields chunks as they arrive) and
`generate(prompt) -> str` (joins the streamed chunks). It's an internal provider only — there is
no public chat endpoint or SSE endpoint yet, and it doesn't touch ingestion or Qdrant.

The model it uses is set independently of the provider: set `LLM_MODEL` (e.g. `llama3.1`) to
choose the chat model without touching `LLM_PROVIDER`. If `LLM_MODEL` is unset, it falls back to
`OLLAMA_CHAT_MODEL` for backward compatibility. `OLLAMA_EMBEDDING_MODEL` is never affected by
`LLM_MODEL` — embeddings must stay on one model to keep previously computed vectors valid.
Changing the embedding model deliberately (not via `LLM_MODEL`) is supported through
`EMBEDDING_MODEL` + `EMBEDDING_VERSION` and a re-index — see "Multilingual RAG foundation" below.

`QdrantVectorStore` (`app/rag/providers/qdrant_vector_store.py`) talks to Qdrant's HTTP API
directly under `QDRANT_URL` (no official Qdrant SDK) via async `httpx`, supporting
`create_collection_if_not_exists(collection_name, vector_size)`,
`upsert_vectors(collection_name, points)`, and
`search_similar(collection_name, query_vector, limit)`. Points and results carry payload
metadata: `document_id`, `chunk_id`, `text`, `source`, and optional `page_number`/`sheet_name`.
It's an internal provider only — no document upload, chat, or SSE endpoint touches it yet, but
`IngestionWorker` now upserts into it (see "Ingestion worker" below).

Providers are resolved through `app/rag/providers/provider_factory.py` rather than importing
Ollama/Qdrant classes directly:

```python
from app.rag.providers.provider_factory import (
    get_embedding_provider,
    get_llm_provider,
    get_vector_store,
)

embedding_provider = get_embedding_provider()   # reads EMBEDDING_PROVIDER
llm_provider = get_llm_provider()               # reads LLM_PROVIDER
vector_store = get_vector_store()               # reads VECTOR_STORE_PROVIDER
```

`LLM_PROVIDER`/`EMBEDDING_PROVIDER`/`VECTOR_STORE_PROVIDER` default to `ollama`/`ollama`/`qdrant`
respectively — all currently resolve to real implementations. An unrecognized provider name
raises `UnsupportedProviderError` with a clear message.

Setting `LLM_PROVIDER=openai`, `LLM_PROVIDER=gemini`, or `LLM_PROVIDER=anthropic` is recognized
but raises `ProviderNotImplementedError` immediately — these are explicit stub classes
(`OpenAIProvider`, `GeminiProvider`, `AnthropicProvider`) that make no external API calls and
never silently fall back to Ollama. They exist so future provider support has a place to land
without changing how the factory or callers behave.

`RuleBasedRagDecider` (`app/rag/decision.py`) is a small internal decision layer that classifies
a question — `NEEDS_RETRIEVAL`, `DIRECT_LLM`, `CLARIFICATION_NEEDED`, or `OUT_OF_SCOPE` — using
deterministic keyword/pattern rules, with **no LLM call made to route**:

```python
from app.rag.decision import RuleBasedRagDecider

result = RuleBasedRagDecider().decide("What does the uploaded document say about refunds?")
print(result.decision, result.reason, result.confidence)
```

It's internal-only — no public API endpoint exposes it, and it doesn't perform retrieval,
generation, ingestion, or document upload itself; it only decides what *should* happen next.

## Platform health and readiness

Four **unversioned** endpoints (`app/api/routes/health.py`, registered without an `/api/v1`
prefix) give load balancers, Kubernetes, and monitoring a stable operational contract that
never moves when the business API's version changes:

| Endpoint | Purpose | Calls dependencies? | Status codes |
|---|---|---|---|
| `GET /health` | Lightweight platform summary — "is the process up at all" | No | Always `200` |
| `GET /health/live` | **Liveness** probe | No | Always `200` while the process is alive |
| `GET /health/ready` | **Readiness** probe | Yes | `200` if every *required* dependency check passes, else `503` |
| `GET /health/dependencies` | Detailed diagnostics for every checked dependency | Yes | Always `200` (status is in the body, not the HTTP code) |

**Liveness vs. readiness**: liveness only answers "is this process alive and not deadlocked" —
it never touches Postgres/Redis/Qdrant/Ollama, so it can't go `503` just because a downstream
dependency is having a bad day (which would cause Kubernetes to needlessly kill and restart a
perfectly healthy process). Readiness answers a different question — "can this instance actually
serve traffic right now" — and can legitimately be `503` while liveness stays `200`, telling a
load balancer/Kubernetes to stop routing traffic here *without* restarting the pod.

`GET /health/ready` checks `postgres`, `qdrant`, `ollama`, and the two configured Ollama models
(`ollama_chat_model`, `ollama_embedding_model`) as **required** — any one of these failing makes
readiness `503`. `redis` is checked too but is **not required** for readiness today, since no
application code path reads or writes it yet (see `REDIS_URL` in
[ARCHITECTURE.md](ARCHITECTURE.md)'s environment variable table) — marking it required would make
Kubernetes pull traffic away from a perfectly capable instance over an unused dependency.
`GET /health/dependencies` reports all six checks regardless, each with `required: true/false`,
for full visibility.

Every check is a small, timeout-bounded probe (`SELECT 1` for Postgres, `PING` for Redis, a
lightweight `GET /collections` for Qdrant, the existing `OllamaClient.check_health()` reachability
+ model-availability check for Ollama) — none of them mutate or restart anything, and none retry
beyond their own timeout. Response bodies never include credentials, connection strings, stack
traces, or raw provider response bodies — only a fixed, generic `detail` message per failure mode.

```bash
curl http://localhost:8000/health
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
curl http://localhost:8000/health/dependencies
```

## Database migrations

**Docker Compose currently starts services but does not apply Alembic migrations
automatically** — `docker compose up` brings up a fresh, unmigrated Postgres, so the schema must
be created explicitly before the app can persist anything. The `documents` and `ingestion_jobs`
tables (see [ARCHITECTURE.md](ARCHITECTURE.md)) are required by `POST /api/v1/documents` — that
endpoint will fail against an unmigrated database.

Run migrations after the `postgres`/`app` containers are up and **before** testing document
upload:

```bash
docker compose exec app alembic upgrade head
```

If you're instead running the app locally with an activated virtual environment (see "Local
setup" above), and Postgres is reachable (e.g. via `docker compose up postgres`), run:

```bash
alembic upgrade head
```

See [alembic/README.md](alembic/README.md) for how migrations are structured, how to generate a
new one, and the full list of Alembic commands used in this project.

### Document upload

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@/path/to/handbook.pdf;type=application/pdf"
```

Returns `202 Accepted`:

```json
{"document_id": "...", "job_id": "...", "status": "pending"}
```

This saves the file via the configured `FileStorage` implementation (`local` by default —
`storage/documents/` under a generated object key; `minio` if `FILE_STORAGE_PROVIDER=minio`, see
"Storage abstraction" below), creates a `Document` row (with the original filename preserved
exactly, Hebrew/Unicode included, plus the provider-neutral storage identity), and creates an
`IngestionJob` row with `status=pending`. **Nothing is parsed, chunked, embedded, or upserted
into Qdrant inside the request.** An empty (zero-byte) file is rejected with `400` before any row
is created.

### Storage abstraction

`app/storage/` (see "Storage Abstraction (Phase 2.6/2.7)" in [ARCHITECTURE.md](ARCHITECTURE.md))
is the provider-neutral `FileStorage` contract every upload/ingestion/extraction code path
depends on. Two implementations exist:

- **`local`** (default) — `LocalFileStorage`, storing files under `LOCAL_STORAGE_ROOT`
  (`storage/documents` by default). No extra setup required.
- **`minio`** — `MinioFileStorage`, an S3-compatible object store. Start it locally with:

  ```bash
  docker compose up -d minio
  ```

  Then set `FILE_STORAGE_PROVIDER=minio` (plus `MINIO_ENDPOINT`/`MINIO_ACCESS_KEY`/
  `MINIO_SECRET_KEY`/`MINIO_BUCKET`, already defaulted in `.env.example`/`docker-compose.yml` for
  local dev) in the `app` service's environment. The MinIO console is at
  `http://localhost:9001` (credentials: `minioadmin`/`minioadmin`, local dev only).

Switching `FILE_STORAGE_PROVIDER` is a deployment-time choice, not something a document's own
row carries a preference for beyond its own already-persisted `storage_provider` value — see
"Backward compatibility for pre-migration documents" in ARCHITECTURE.md for how documents
written before this feature remain readable.

### Document read APIs and original download

Five read-only endpoints let you inspect a document's lifecycle and download its original
content — see "Document read APIs and original download (Phase 2.8.2)" in
[ARCHITECTURE.md](ARCHITECTURE.md) for full contract details, the lifecycle-status derivation
table, and the 404/409/503 mapping rationale. None of them mutate anything (delete/re-index are
separate, mutating endpoints, see below; the read-only reconciliation reporting endpoints are
documented further down).

```bash
# List (paginated, newest first)
curl "http://localhost:8000/api/v1/documents?limit=20&offset=0"

# Detail
curl "http://localhost:8000/api/v1/documents/<document_id>"

# Latest ingestion job status (200 with null fields if no job exists yet)
curl "http://localhost:8000/api/v1/documents/<document_id>/ingestion"

# Latest failed ingestion job, with a sanitized error message (404 if it never failed)
curl "http://localhost:8000/api/v1/documents/<document_id>/failure"

# Download the original file
curl -OJ "http://localhost:8000/api/v1/documents/<document_id>/download"
```

`GET .../download` streams the original bytes back with `Content-Type` set from the stored
`content_type` and an RFC 5987/6266-compliant `Content-Disposition: attachment` header (Hebrew/
Unicode filenames survive via a `filename*=UTF-8''...` percent-encoded form alongside an ASCII
fallback). `404` if the document doesn't exist, `409` if the document row exists but its storage
object is missing (a real inconsistency, not "not found"), `503` if the storage backend itself is
unreachable.

### Ingestion retry and stale-job recovery

See "Ingestion retry and stale-job recovery (Phase 2.8.3)" in [ARCHITECTURE.md](ARCHITECTURE.md)
for the full decision table, the one-active-job-per-document Postgres constraint, and the
vector-idempotency reasoning.

```bash
# Retry a FAILED (or stale-PROCESSING) document — 202 + new job if scheduled, 200 if an
# already-active job exists, 404 if the document is missing, 409 if it's already indexed.
curl -X POST "http://localhost:8000/api/v1/documents/<document_id>/ingestion/retry"

# Run one stale-PROCESSING-job recovery batch (manual/optional — never run by make verify/CI):
make recover-stale-ingestion-jobs
```

Retry never deletes or resets an existing `IngestionJob` row — history is append-only, and a new
attempt always means a brand-new `PENDING` row for the existing `IngestionWorker` to pick up.
Retrying doesn't require re-uploading the original file (the same `FileStorage` object is reused)
or any vector-cleanup step (Qdrant point IDs are deterministic per document/chunk, so a retry's
successful upsert naturally overwrites what a first successful attempt would have written).

### Full document deletion

See "Full document deletion (Phase 2.8.4)" in [ARCHITECTURE.md](ARCHITECTURE.md) for the full
scheduling decision table, the deletion-job state machine, the vectors-before-storage cleanup
order, and the lifecycle-precedence rule. The `Document` row and all history (`IngestionJob`/
`VectorCleanupJob`/`DocumentDeletionJob`) are never physically deleted — only a document's
external resources (Qdrant vectors, the stored object) are removed.

```bash
# Schedule deletion — 202 if newly scheduled or already active, 200 if already fully deleted,
# 404 if the document is missing, 409 if it has an active ingestion job or an active re-index job
# (deletion never races an in-flight ingestion or an in-flight re-index build).
curl -X DELETE "http://localhost:8000/api/v1/documents/<document_id>"

# Inspect the latest deletion attempt (404 if none was ever requested).
curl "http://localhost:8000/api/v1/documents/<document_id>/deletion"

# Execute pending deletion jobs against the configured database/Qdrant/storage (manual/optional —
# never run by make verify/CI, mirrors make recover-stale-ingestion-jobs):
make process-pending-document-deletions
```

`DELETE` only ever *schedules* a deletion (inserts a `PENDING` `DocumentDeletionJob` row) — it
never performs the actual cross-system cleanup inline, mirroring how `POST /api/v1/documents`
never runs ingestion inline either. Execution (`DocumentDeletionWorker`, invoked by
`scripts/process_pending_document_deletions.py`) deletes every tracked vector collection
(`delete_all_tracked_document_vectors()`) strictly before deleting the original stored object; a
partial vector-cleanup failure blocks storage deletion entirely and marks the attempt
`PARTIALLY_FAILED` (`deletion_failed` lifecycle) rather than reporting a false success. A deleted
document's `GET .../download` returns `410 Gone` (the row still exists; only its content was
removed), and `POST .../ingestion/retry` on it returns `409` — a document is never implicitly
resurrected once deletion has begun.

### Re-index scheduling and activation

See "Single-document re-index API" and "Job-id-scoped operator activation endpoint" in
[ARCHITECTURE.md](ARCHITECTURE.md) for the full precondition chains, locking behavior, and
rollback guarantees. Every re-index endpoint is manual/operator-triggered — there is no automatic
activation after a build completes, no scheduler, and no activation-history persistence.

```bash
# Inspect staleness + the latest attempt (404 if the document doesn't exist).
curl "http://localhost:8000/api/v1/documents/<document_id>/reindex"

# Schedule a re-index — 202 if newly scheduled, 200 if an already-active attempt exists, 404/409
# for every other blocking condition. Never builds inline (the existing ReindexWorker does, out of
# band).
curl -X POST "http://localhost:8000/api/v1/documents/<document_id>/reindex"

# Activate a completed build — document-scoped: resolves the relevant job through the document
# (defaults to the document's latest attempt; accepts an optional ?job_id= to target a specific
# one):
curl -X POST "http://localhost:8000/api/v1/documents/<document_id>/reindex/activate"

# Activate a completed build — job-scoped: targets one explicit re-index job directly by id, so
# the caller doesn't need to already know the owning document:
curl -X POST "http://localhost:8000/api/v1/reindex/jobs/<job_id>/activate"

# Build one pending re-index job against the configured database/Qdrant/storage (manual/optional —
# never run by make verify/CI, mirrors make process-pending-document-deletions; processes at most
# one job per invocation, invoke repeatedly to make further progress):
make process-pending-reindex-jobs
```

`POST .../reindex` only ever inserts a `PENDING` `ReindexJob` row — it never builds inline. Building
is a separate, explicit, bounded operational step: `make process-pending-reindex-jobs`
(`scripts/process_pending_reindex_jobs.py`) claims and builds **at most one** pending job per
invocation via the existing `ReindexWorker`, then exits — an operator or external scheduler invokes
it repeatedly to make further progress; the script itself never loops, polls, or schedules itself. A
successful build writes the target's vectors into a new collection but **never switches which
collection the document serves from** — the running process keeps serving the document's current
collection untouched, for as long as the operator wants, until an explicit activation call.

Both activation endpoints delegate to the exact same `activate_reindexed_document()` service call
— one atomic, `SELECT ... FOR UPDATE`-locked transaction that switches the document's serving
collection/embedding/chunking metadata, sets `activated_at`, and persists a `VectorCleanupJob` for
the vacated collection — in the same commit. A failure anywhere in that transaction rolls back
entirely; the document is never left pointing at a new collection unless `activated_at` was
actually persisted. Calling activation again on an already-activated job is idempotent — `200`
with `already_activated: true` in the response, no second switch, no second `activated_at`, no
duplicate cleanup job; 404 if the job/document can't be found; 409 if the job isn't `COMPLETED`,
its source collection changed since scheduling, or the document has a blocking deletion in
progress. Activation never executes the `VectorCleanupJob` it creates inline — that remains a
separate, out-of-band operation. `process_next_vector_cleanup_job()` claims one pending/failed
cleanup job at a time (oldest first), refuses to delete a collection that is still the document's
*current* active collection (a defensive safety guard), and marks the job `COMPLETED`/`FAILED` on
completion; processing more than one job per call, and looping, is left to whatever caller invokes
it — there is no dedicated Makefile target or script for this yet in this codebase (unlike full
document deletion's `make process-pending-document-deletions`). The vacated collection's vectors
remain in place — readable, unused — until that cleanup step actually runs.

**Document-scoped vs. job-scoped activation:** the document-scoped route resolves which
`ReindexJob` to activate *through the document* (its latest attempt, or an explicit `?job_id=`
belonging to that document); the job-scoped route targets one explicit re-index job directly by
`job_id` alone, without requiring the caller to already know the owning document — useful when an
operator already has a job id (e.g. from the schedule response) and wants to activate it directly.

**Explicitly not included in this API surface:**
- no automatic activation — a build completing never activates itself
- no reconciliation/audit API of any kind
- no scheduler or cron loop — the worker and cleanup processor are both invoked out-of-band,
  manually or via an external scheduler you control
- no automatic repair of a stale, failed, or partially-cleaned-up state

### Ingestion worker

`IngestionWorker` (`app/services/ingestion/worker.py`) is an internal service — no public API —
that claims and resolves one `pending` `IngestionJob` at a time:

```python
from app.services.ingestion.worker import IngestionWorker

worker = IngestionWorker()
job = await worker.process_next_job(session)  # None if there's nothing pending
```

It claims the oldest pending job with Postgres row-level locking
(`SELECT ... FOR UPDATE SKIP LOCKED`), flips it to `processing`, runs a processing step (its
default: Document → `DocumentTextExtractor` → `DocumentChunker` → `EmbeddingProvider` →
`VectorStore` upsert — see below), then resolves it to `completed` on success or `failed` (with
the error message stored) on any exception. It's idempotent: a job that's already `completed` or
`failed` is never selected again by the claim query, so calling `process_next_job()` repeatedly
never re-processes it. The worker never calls `LLMProvider` — ingestion only embeds and indexes,
it never generates text.

### Document text extraction

`DocumentTextExtractor` (`app/services/documents/text_extractor.py`) reads a document's content
via the injected `FileStorage` and extracts its raw text entirely in memory (no local path, no
temporary file). **It routes by file extension and validates each file's basic
structure/content before extraction** — a mismatched or corrupt file fails clearly instead of
being handed to the wrong parser:

| Extension | Handler | Validated before parsing |
|-----------|---------|----------------------------|
| `.txt`    | UTF-8 plain text | Readable as UTF-8 |
| `.md`     | UTF-8 markdown/plain text (no Markdown parsing) | Readable as UTF-8 |
| `.pdf`    | `pypdf`, page by page, 1-indexed `page_number` preserved | File starts with the `%PDF` header |
| `.docx`   | `python-docx`, plain paragraph text, a single page | Valid ZIP archive containing `word/document.xml` |
| `.xlsx`   | `openpyxl`, sheet by sheet, each sheet's name in `sheet_name` | Valid ZIP archive containing `xl/workbook.xml` |

This is lightweight structural validation, not deep content sanitization — it doesn't check the
upload's `content_type` header or scan for malicious payloads.

```python
from app.services.documents.text_extractor import DocumentTextExtractor

extracted = await DocumentTextExtractor().extract(document)
for page in extracted.pages:
    print(page.page_number, page.sheet_name, page.text)
```

Raises `DocumentTextExtractionError` for a missing stored file, an unsupported extension, or
empty/whitespace-only extracted text — `IngestionWorker` catches this and marks the job
`failed` with the error message stored. UTF-8/Unicode content (Hebrew included) is preserved
exactly across all five file types. This is the ingestion worker's real first processing step;
its output feeds directly into chunking below.

### Document chunking

`DocumentChunker` (`app/services/documents/chunker.py`) takes an `ExtractedDocument` and splits
it into fixed-size, overlapping, word-boundary-aware chunks — no embedding, no Qdrant upsert, no
retrieval:

```python
from app.services.documents.chunker import DocumentChunker

chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)  # or read from Settings
chunks = chunker.chunk(extracted)  # list[DocumentChunk]
for chunk in chunks:
    print(chunk.chunk_id, chunk.page_number, chunk.sheet_name, chunk.text)
```

`chunk_size`/`chunk_overlap` default to the `CHUNK_SIZE`/`CHUNK_OVERLAP` settings (1000/200
characters). Chunks never split inside a word, overlap is built from whole trailing words of the
previous chunk, empty/whitespace-only pages produce zero chunks, and `chunk_id`s
(`f"{document_id}-{chunk_index}"`) are deterministic — the same document always produces the
same chunks in the same order. `page_number`/`sheet_name` are carried over from the source page
(PDF/XLSX respectively; `None` for `.txt`/`.md`/`.docx`).

This is the ingestion worker's second processing step; its output feeds directly into embedding
below.

### Chunk embedding and Qdrant indexing

The ingestion worker's default pipeline continues past chunking: each `DocumentChunk`'s text is
embedded via `get_embedding_provider().embed(...)`, and the resulting vectors are upserted into
Qdrant via `get_vector_store()`:

```python
embedding_provider = get_embedding_provider()   # reads EMBEDDING_PROVIDER
vectors = await embedding_provider.embed([chunk.text for chunk in chunks])

vector_store = get_vector_store()               # reads VECTOR_STORE_PROVIDER
await vector_store.create_collection_if_not_exists(
    settings.qdrant_collection_name, settings.vector_size
)
await vector_store.upsert_vectors(settings.qdrant_collection_name, points)
```

Each `VectorPoint`'s `id` is a deterministic UUIDv5 derived from the chunk's `chunk_id` (so
re-processing a document overwrites the same points rather than duplicating them), and its
payload preserves `document_id`, `chunk_id`, `text`, `source` (the document's
`original_filename`), and optional `page_number`/`sheet_name` carried over from the chunk. The
collection is created (if missing) with `QDRANT_COLLECTION_NAME`/`VECTOR_SIZE` before every
upsert — cheap since Qdrant no-ops when the collection already exists. If embedding a chunk or
upserting into Qdrant raises, `IngestionWorker` catches it and marks the job `failed` with the
error message stored, same as an extraction or chunking failure. A document with zero chunks
(e.g. empty/whitespace-only text) completes without calling the embedding provider or vector
store at all. This is the ingestion worker's final processing step — there is still no retrieval,
chat, or SSE endpoint that reads these vectors back out.

### Retrieval service

`RetrievalService` (`app/rag/retrieval_service.py`) is the internal read-side counterpart to the
ingestion worker's embed/upsert steps: given a query, it embeds it and searches Qdrant for
relevant chunks — no LLM call, no public retrieval/chat/SSE endpoint, no RAG prompt assembly.

```python
results = await RetrievalService().retrieve("what is the refund policy?")
# results: list[VectorSearchResult], ranked by Qdrant score
```

`retrieve(query, limit=None)` rejects an empty/whitespace-only `query` with `EmptyQueryError`
before calling any provider, embeds the query via `get_embedding_provider()`, and searches
`QDRANT_COLLECTION_NAME` via `get_vector_store().search_similar(...)` with `limit` (falling back
to `RETRIEVAL_TOP_K` when omitted). Results come back already ranked by Qdrant's own score
ordering. If `RETRIEVAL_SCORE_THRESHOLD` is set, results scoring below it are filtered out; left
unset, no score filtering happens. Each `VectorSearchResult` preserves `document_id`, `chunk_id`,
`text`, `source`, `page_number`, `sheet_name`, and `score`. An embedding or vector-store failure
propagates as-is rather than being swallowed, and no matches simply means an empty list — nothing
is fabricated.

### RAG prompt builder

`RagPromptBuilder` (`app/rag/prompt_builder.py`) turns a user question and ranked
`VectorSearchResult`s (from `RetrievalService`) into a deterministic, structured prompt — no LLM
call, no public chat/SSE endpoint, no conversation memory, and it never changes retrieval
behavior itself:

```python
from app.rag.prompt_builder import RagPromptBuilder

results = await RetrievalService().retrieve("what is the refund policy?")
built = RagPromptBuilder().build("what is the refund policy?", results)
# built: BuiltRagPrompt(system_prompt, user_prompt, context, sources)
```

`build(question, results)` filters out any result with empty/whitespace-only `text`, then, for
each remaining result **in the given (already-ranked) order**, assigns a stable label —
`[S1]`, `[S2]`, ... — and formats a context block with that label, the result's `source`
filename, `page N` when `page_number` is set, `sheet <name>` when `sheet_name` is set, and the
chunk text itself. `system_prompt` instructs the model to answer only from the supplied context,
never invent missing information, and say explicitly when the answer isn't present. Each context
block has a matching `PromptSource` (`document_id`, `chunk_id`, `source`, `score`,
`page_number`, `sheet_name`) in `sources`, in the same order as the context. If no result has
non-empty text, `context` states plainly that no relevant context was found — no fallback content
is fabricated — and `sources` is an empty list; this no-results path is exactly as deterministic
as the normal path. `RagPromptBuilder` is pure and synchronous: it doesn't mutate the
`VectorSearchResult`s it's given and never imports or calls `LLMProvider`.

### RAG orchestrator

`RagOrchestrator` (`app/rag/orchestrator.py`) composes the decision layer, retrieval service,
prompt builder, and streaming LLM provider into a single call — **no conversation memory, and no
silent fallback between decisions or providers**. It's exposed publicly via `POST /api/v1/chat`
(see "Streaming chat endpoint" below):

```python
from app.rag.orchestrator import RagOrchestrator

async for event in RagOrchestrator().stream_answer("what is the refund policy?"):
    print(event)
# OrchestratorMetadata(decision=..., reason=..., retrieval_used=..., sources=[...])
# OrchestratorToken(text="...")
# OrchestratorToken(text="...")
# ...
```

`stream_answer(question)` routes the question through `RuleBasedRagDecider.decide(...)` first:

- **`CLARIFICATION_NEEDED`/`OUT_OF_SCOPE`**: streams one `OrchestratorMetadata`
  (`retrieval_used=False`) followed by a single fixed `OrchestratorToken` message — no
  `RetrievalService` call and no `LLMProvider` call at all.
- **`NEEDS_RETRIEVAL`**: calls `RetrievalService.retrieve(question)`, builds a prompt via
  `RagPromptBuilder`, streams one `OrchestratorMetadata` (`retrieval_used=True`, `sources` from
  the built prompt), then streams `OrchestratorToken`s from `LLMProvider.stream_generate(...)`
  as they arrive.
- **`DIRECT_LLM`**: streams one `OrchestratorMetadata` (`retrieval_used=False`, no sources), then
  streams `OrchestratorToken`s from the LLM directly, without calling retrieval.

A failure in `RetrievalService` or the LLM provider (`get_llm_provider()`, reads `LLM_PROVIDER`)
propagates to the caller unchanged — `RagOrchestrator` never catches it to fall back from
retrieval to a direct answer, and never silently switches providers.

### RAG engine compatibility layer

`RagOrchestrator` is wrapped behind a small, replaceable `RagEngine` abstraction
(`app/rag/engine.py`) so an alternative RAG execution engine can be swapped in without touching
the public API, the SSE contract, or any existing provider/retrieval/prompt/orchestration code:

```
RagEngine (app/rag/engine.py)
├── CustomRagEngine   (app/rag/engines/custom_engine.py)   — default; wraps RagOrchestrator unchanged
└── LangChainRagEngine (app/rag/engines/langchain_engine.py) — optional
```

Selected via `RAG_ENGINE`:

```bash
RAG_ENGINE=custom      # default — existing installs behave exactly as before
RAG_ENGINE=langchain   # optional — routes RAG execution through LangChain Runnables
```

`get_rag_engine(settings)` (`app/rag/engines/engine_factory.py`) resolves the configured engine —
an unrecognized `RAG_ENGINE` value raises `UnsupportedRagEngineError` immediately; it never
silently falls back to `custom` and never silently switches providers.

**`CustomRagEngine`** is a thin adapter with no logic of its own: it delegates every call directly
to `RagOrchestrator`, so `RuleBasedRagDecider`, `RetrievalService`, `RagPromptBuilder`,
`LLMProvider.stream_generate()`, source metadata, and failure propagation are all completely
unchanged — this remains the platform's reference implementation.

**`LangChainRagEngine`** runs the same four decision paths
(`NEEDS_RETRIEVAL`/`DIRECT_LLM`/`CLARIFICATION_NEEDED`/`OUT_OF_SCOPE`) through LangChain
Runnables/prompt values instead of `RagOrchestrator`'s plain Python composition, while reusing the
platform's existing pieces via three adapters (`app/rag/engines/langchain_adapters.py`):

- `ProviderBackedLLM` — a LangChain `LLM` that streams from whatever `LLMProvider`
  `get_llm_provider()` resolved (never constructs an Ollama/OpenAI/Gemini/Anthropic client itself).
- `ProviderBackedEmbeddings` — a LangChain `Embeddings` wrapping the configured `EmbeddingProvider`.
- `ProviderBackedRetriever` — a LangChain `BaseRetriever` wrapping the existing `RetrievalService`
  (and therefore the existing `QdrantVectorStore`/`QDRANT_COLLECTION_NAME`/`VECTOR_SIZE` — no
  second Qdrant SDK path, no separate collection, no different embedding model).

Retrieved LangChain `Document`s are converted straight back into `VectorSearchResult`s and handed
to the existing, unmodified `RagPromptBuilder`, so source labels (`[S1]`, `[S2]`), rank order, the
"answer only from context / say so if the answer isn't present" instructions, and Hebrew/Unicode
text all behave exactly as they do for `CustomRagEngine`. `CLARIFICATION_NEEDED`/`OUT_OF_SCOPE`/
no-results never invoke the LLM, and stream the exact same fixed, language-appropriate message
text `RagOrchestrator` uses — both resolve it via `PromptProvider` (`app/rag/prompts/`), a small
framework-neutral shared module (no FastAPI/LangChain/engine dependency of its own) so neither
engine imports this text from the other's implementation module. See "Multilingual RAG
foundation" below for the full language-aware prompt design. No LangGraph, no agents, no tool
calling — see "LangChain compatibility layer" in [ARCHITECTURE.md](ARCHITECTURE.md) for the full
engine design.

The generated answer text can legitimately differ between engines (LangChain's own prompt
serialization differs from `RagOrchestrator`'s plain string concatenation), but the public
API/SSE contract, decision routing, retrieval usage, and source attribution are identical either
way — the chat route and the frontend never need to know which engine is configured.

Run the engine-specific tests with:

```bash
make test-rag-engines     # unit + integration + E2E-parity tests for both engines (needs Docker)
make verify-rag-engines   # runs test-rag-engines plus its own checks
```

### Multilingual RAG foundation

Multilingual (Hebrew + English) retrieval and language-aware prompting are shared platform
capabilities, reached identically by both `CustomRagEngine` and `LangChainRagEngine` — neither
engine detects language, selects an embedding model, or owns a prompt catalog itself:

```
Question -> LanguageDetector -> PromptProvider -> PromptCatalog -> ResolvedPrompt -> RagEngine
```

- **Versioned embedding/index configuration** (`app/rag/embedding_config.py`) — `provider`,
  `model`, `dimension`, `EMBEDDING_VERSION`, `CHUNKING_VERSION` together derive a deterministic,
  sanitized Qdrant collection name. Changing any one of them always produces a different
  collection — incompatible vectors can never land in the same collection, and `IngestionWorker`/
  `RetrievalService` always resolve the same active configuration.
- **Collection safety** (`app/services/indexing/collection_registry.py`) — an existing collection with the
  wrong vector dimension is rejected explicitly (`IncompatibleIndexConfigurationError`), never
  silently reused, recreated, or deleted.
- **Document indexing metadata** — `Document` rows record exactly which embedding
  provider/model/dimension/version and chunking version they were indexed with, and when; a
  document is "stale" whenever that stored configuration no longer matches the active one — not
  merely because vectors exist somewhere in Qdrant.
- **Real embedding-vector validation** (`app/rag/embedding_validation.py`) — every embedding
  batch (ingestion, re-index, and the single query vector at retrieval time) is checked against
  the active configuration's expected count/dimension *before* any Qdrant write/search, catching a
  misconfigured `EMBEDDING_MODEL`/`VECTOR_SIZE` pair immediately instead of only once an existing
  collection happens to disagree.
- **Re-index** (`app/services/indexing/reindex_service.py`) — re-derives a document's vectors from its
  already-persisted stored file (no new upload) when its configuration changes; idempotent, and
  returns a typed `ReindexResult`/`ReindexOutcome` (`ALREADY_CURRENT`/`REINDEXED`/
  `REINDEXED_EMPTY`/`REINDEXED_WITH_CLEANUP_PENDING`) rather than a plain bool, since a bool can't
  represent "zero-chunk document" or "the new collection committed but cleaning up the old one
  failed" distinctly. A Postgres commit failure after a successful Qdrant write rolls back and
  expires the `Document` — see "Re-index (`app/services/indexing/reindex_service.py`)" in
  [ARCHITECTURE.md](ARCHITECTURE.md) for the full non-atomic-transaction contract. A failed
  legacy-collection cleanup is tracked as a retryable `VectorCleanupJob`, never silently dropped.
- **Language detection** (`app/rag/language.py`) — deterministic, word-level Hebrew/Latin
  script-dominance counting (not an ML model), so a few Latin-script technical identifiers
  (Kafka, Qdrant, Kubernetes, LangChain) embedded in a Hebrew question never override the
  surrounding language, and vice versa.
- **Multilingual decision routing** (`app/rag/decision.py`) — `RuleBasedRagDecider` has Hebrew
  equivalents of every English pattern (document/file references, extraction-verb-near-
  sensitive-noun), so a natural Hebrew question (e.g. `לפי הקובץ שהעליתי, מה מדיניות השמירה?`)
  routes to retrieval without needing an English trigger phrase, and both engines see the same
  decision by construction (one shared decider, no per-engine decision logic).
- **PromptCatalog/PromptProvider** (`app/rag/prompts/`) — five prompt types
  (`grounded_answer`/`direct_answer`/`clarification`/`no_results`/`out_of_scope`) x two languages
  (`he`/`en`); both engines resolve all fixed/governed text through `PromptProvider`, never a
  private constant. The two generation-backed types share **one English-authored governance
  instruction** (never duplicated per language) plus an explicit response-language directive
  (`"Respond directly and naturally in Hebrew (he)."`/`"...in English (en)."`) — never "answer in
  English and translate." Instructions require answering only from context, preserving quoted
  source text and `[S1]`/`[S2]` labels untranslated, and never translating code/API names/class
  names/filenames/commands/environment variables/error messages.
- **Multilingual embedding model** — `OLLAMA_EMBEDDING_MODEL` defaults to `bge-m3` (1024-dim,
  BAAI's embedding model supporting 100+ languages including Hebrew); requires
  `ollama pull bge-m3`. `EMBEDDING_VERSION` defaults to `v2` alongside this default-model change,
  so any pre-existing installation built on Phase 2.5's `v1`/`nomic-embed-text` (768-dim) config
  never silently reuses that collection — it re-indexes into a new one. `.env.example` documents
  pinning back to the legacy English-only `nomic-embed-text` (768-dim, NOT recommended for
  Hebrew content) via `EMBEDDING_MODEL=nomic-embed-text` + `VECTOR_SIZE=768` +
  `EMBEDDING_VERSION=v1`. **Changing the embedding model always creates a new versioned
  collection and requires document re-indexing** — the previous collection's vectors are never
  deleted automatically. Automated tests never depend on a real embedding model — they use a
  deterministic fake (`tests/multilingual_fixtures.py`) with a small Hebrew/English
  concept-synonym table, which proves the retrieval *wiring* works cross-language, not real model
  retrieval quality; see "Real multilingual runtime smoke" below for an optional, manual
  real-`bge-m3` check — broader recall/ranking evaluation on a larger corpus remains future work.

See "Multilingual RAG Foundation" in [ARCHITECTURE.md](ARCHITECTURE.md) for the full design, and
run the multilingual-specific tests with:

```bash
make test-multilingual-rag     # unit + integration + E2E matrix tests (needs Docker)
make verify-multilingual-rag   # runs test-multilingual-rag plus its own checks
```

#### Real multilingual runtime smoke (optional, manual)

`make smoke-multilingual-real` exercises the real, configured embedding model (default `bge-m3`)
against five Hebrew/English scenarios (Hebrew doc/query, cross-language both directions, English
doc/query, mixed Hebrew+English with embedded technical identifiers), asserting the correct
source scores higher than an unrelated distractor and the vector dimension matches configuration.
It requires a locally reachable Ollama with the model already pulled, fails clearly (non-zero
exit, explicit message) if the model isn't installed, and is never run by `make verify`/
`make test*`/CI — it's a small illustrative corpus, not a production-scale retrieval-quality
evaluation.

### Streaming chat endpoint

`POST /api/v1/chat` (`app/api/v1/routes/chat.py`) is a thin route that streams the configured
`RagEngine.stream_answer(question)` back as Server-Sent Events — no decision, retrieval, or
prompt-building logic in the route itself, no direct provider calls, and no branch on which engine
is selected:

```bash
curl -N -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"What does the uploaded document say?"}'
```

Request body: `{"question": "..."}` — a Pydantic validator rejects an empty/whitespace-only
`question` with `422` before the route is even reached. There is no `model` field: the
orchestrator/LLM provider interface doesn't support a validated per-request model override yet,
and the embedding model is never client-selectable (fixed via `OLLAMA_EMBEDDING_MODEL`, same as
everywhere else in this project).

Response: `Content-Type: text/event-stream`, one event per line-pair, each followed by a blank
line:

```
event: <event-name>
data: <JSON>

```

Four deterministic event types, always in this shape:

| Event | Emitted when | JSON body |
|---|---|---|
| `metadata` | Always first, once, from the run's `OrchestratorMetadata` | `decision`, `reason`, `retrieval_used`, `sources` (each: `document_id`, `chunk_id`, `source`, `score`, plus `page_number`/`sheet_name` when present) |
| `token` | Once per `OrchestratorToken`, in generation order | `text` |
| `done` | Exactly once, after the last token, only on normal completion | `status: "completed"` |
| `error` | At most once, if the configured `RagEngine.stream_answer(...)` raises after streaming has started | `message` (a fixed, safe string — never a stack trace, prompt, secret, credential, internal URL, or provider response body), `status: "failed"` |

An `error` event ends the stream — no `done` event follows it. The route does not buffer the
full answer: each `OrchestratorToken` is written to the response as soon as the engine yields it,
so `curl -N` (no-buffer mode, shown above) prints tokens as they arrive rather than all at once at
the end. This event shape is identical regardless of which `RagEngine` produced it.

### Reconciliation reporting APIs (read-only)

Three read-only diagnostic endpoints (`app/api/v1/routes/reconciliation.py`) over the
reconciliation service layer — one document, a bounded page of documents, and one collection.
None of them mutate anything: no repair, no scheduler, no CLI, no persisted audit/report history,
no automatic activation, no cleanup execution, no orphan-ID discovery. See "Reconciliation:
bounded batch lifecycle audit" / "Single-document lifecycle audit API" / "Collection
reconciliation report API" in [ARCHITECTURE.md](ARCHITECTURE.md) for full contracts.

```bash
# One document, by id — always 200, even for a document that doesn't exist (see below).
curl "http://localhost:8000/api/v1/reconciliation/documents/<document_id>/audit"

# A bounded, oldest-first page of documents (opaque cursor pagination).
curl "http://localhost:8000/api/v1/reconciliation/documents/audit?limit=20"

# One collection's consistency report, by name — always 200, even for a collection that
# doesn't exist (see below).
curl "http://localhost:8000/api/v1/reconciliation/collections/<collection_name>/report"
```

**Single-document audit** delegates to `audit_document_lifecycle()` exactly once and is always
`200` — a missing document is represented as `classification: "not_found"` with
`database.document_exists: false`, never a `404` (the service already returns this as a typed
result, not an exception, and the route preserves that). Response:

```json
{
  "document_id": "...",
  "overall_status": "consistent",
  "classification": "warning",
  "issues": [{"code": "...", "severity": "warning", "summary": "...", "...": "..."}],
  "database": {
    "document_exists": true, "collection_name": "documents-v2",
    "document_created_at": "2026-01-01T00:00:00Z", "latest_ingestion_status": "completed",
    "latest_deletion_status": null, "latest_reindex_status": null,
    "latest_reindex_activated": false, "pending_cleanup_collections": []
  },
  "file_storage": {"inspected": true, "source_file_exists": true},
  "vector_store": {
    "inspected": true, "collection_name": "documents-v2",
    "collection_exists": true, "has_vectors": true, "vector_count": 3
  }
}
```

**Batch document audit** pages through documents oldest-first via an opaque keyset cursor.
`limit` (optional, 1-50, default 20; out-of-range is a `422`) and `cursor` (optional, opaque — pass
back the previous page's `next_cursor` verbatim; a malformed cursor is a `400`). Response:

```json
{
  "items": [
    {
      "document_id": "...", "original_filename": "report.pdf",
      "created_at": "2026-01-01T00:00:00+00:00", "overall_status": "consistent",
      "classification": "warning",
      "issues": [{"code": "...", "severity": "warning", "summary": "...", "...": "..."}]
    }
  ],
  "summary": {
    "total": 20, "consistent": 15, "transitional": 2, "warning": 2,
    "inconsistent": 1, "not_found": 0, "dependency_unavailable": 0,
    "finding_counts": {"stale_ingestion_job": 1}
  },
  "limit": 20,
  "next_cursor": "opaque-cursor-or-null"
}
```

Documents are always returned oldest-first (deterministic keyset pagination), audited
sequentially (one shared `AsyncSession`, never concurrently); an empty repository returns `200`
with empty `items`/zeroed `summary`/`next_cursor: null`, never `404`.

**Collection report** delegates to `build_collection_reconciliation_report()` exactly once and is
always `200` — a missing collection is `classification: "missing"`, `exists: false`,
`actual_vector_count: 0`, never a `404`; `collection_name` is validated (`400` if malformed) but
otherwise never transformed. `is_active` compares against the platform's single currently-desired
collection (`get_active_embedding_config()`), independent of `index_collection_status`
(`IndexCollection.status`'s own `active`/`retired` bookkeeping flag) — an inactive collection can
still be perfectly consistent. `expected_vector_count` is a **document-count-based proxy**, not a
tracked chunk count (this schema never persists one) — only a *deficit*
(`actual_vector_count < expected_vector_count`) is ever flagged `inconsistent`; a surplus (the
normal case for multi-chunk documents) is not. Response:

```json
{
  "collection_name": "documents-v2",
  "classification": "healthy",
  "exists": true,
  "is_active": true,
  "index_collection_status": "active",
  "embedding_provider": "ollama", "embedding_model": "bge-m3",
  "embedding_dimension": 1024, "embedding_version": "v1", "chunking_version": "v1",
  "document_count": 1540,
  "expected_vector_count": 1540,
  "actual_vector_count": 1540,
  "difference": 0,
  "issues": [],
  "generated_at": "2026-07-19T10:00:00Z"
}
```

## Verification

A `Makefile` wraps all quality gates behind one command:

```bash
make test        # pytest -m "not integration and not e2e and not slow" -q (the fast unit suite)
make test-unit    # alias for 'make test'
make lint         # ruff check .
make typecheck    # mypy app
make compose      # docker compose config
make verify       # runs test, lint, typecheck, compose, in order — stops at the first failure
```

`make verify` is the standard pre-commit/pre-PR check, and stays fast and Docker-independent
(beyond `docker compose config`, which only validates the compose file — it starts nothing). If
`make` isn't available, run the underlying commands directly:

```bash
pytest -m "not integration and not e2e and not slow" -q
ruff check .
ruff check --fix .    # lint + autofix
mypy app
docker compose config
```

All four gates (`pytest`, `ruff check .`, `mypy app`, `docker compose config`) must pass cleanly
before committing. `make verify` never runs the Testcontainers-based integration suite — see
"Integration tests" below for that.

Run `make help` any time for a quick summary of these commands.

## Unit tests

Every unit test lives under `tests/unit/` — nothing sits directly under `tests/*.py` anymore.
The layout mirrors `app/`'s own package structure, one directory per top-level concern:

```
tests/unit/
├── configuration/   # Settings/.env.example consistency
├── core/            # app.core.config
├── api/             # route-level tests (dependency-override style, fake DB session)
├── services/
│   ├── documents/   # mirrors app/services/documents/ 1:1
│   ├── ingestion/   # mirrors app/services/ingestion/ 1:1
│   └── indexing/    # mirrors app/services/indexing/ 1:1
├── rag/             # app.rag.decision/orchestrator/prompt_builder/retrieval_service/etc.
│   ├── engines/     # mirrors app/rag/engines/ 1:1
│   ├── prompts/     # mirrors app/rag/prompts/ 1:1
│   └── providers/   # mirrors app/rag/providers/ 1:1
├── storage/         # mirrors app/storage/ 1:1
└── scripts/         # tests for scripts/*.py contracts
```

Fakes/mocks only — no Docker, no Testcontainers, no real Postgres/Qdrant/MinIO/Ollama. This is
what `make test`/`make verify` run (`pytest -m "not integration and not e2e and not slow" -q`,
which still discovers everything under `tests/unit/` via `testpaths = ["tests"]`).

## Integration tests

A separate, Testcontainers-based integration suite lives under `tests/integration/` (marked
`@pytest.mark.integration`) — it is **not** part of `make test`/`make verify`, so the normal
fast unit suite never needs Docker beyond `docker compose config`.

- **Docker is required** to run this suite — Testcontainers for Python starts real, isolated,
  temporary Postgres and Qdrant containers on ephemeral (dynamically assigned) ports.
- **The repository's main `docker-compose.yml` is not used for integration tests** — no fixed
  ports, no shared/persistent Compose volumes, no reuse of local Compose state. Each test session
  gets its own fresh containers, started and torn down entirely by Testcontainers.
- **Real Postgres/Qdrant, fake deterministic AI providers**: migrations, `IngestionWorker`
  transaction/locking behavior, Qdrant's actual HTTP contract, and platform readiness
  (`GET /health/ready` against real Postgres/Qdrant, with Redis/Ollama checks faked
  deterministically) are all exercised against real services, but embeddings come from a small
  fake, deterministic provider — **no real Ollama container runs and no model is pulled** as
  part of this first suite.
- **MinIO integration coverage** — `tests/integration/test_minio_storage.py` and
  `tests/integration/ingestion/test_worker_minio.py` run against a real, ephemeral MinIO
  container (Testcontainers, dynamic port, no persistent volume — same pattern as
  Postgres/Qdrant): bucket initialization, save/read/delete/exists/metadata, presigned download
  URLs, missing-object/error-translation behavior, and a full upload → Postgres → MinIO →
  extraction → chunking → fake-embeddings → Qdrant chain. A real-Ollama smoke suite is still not
  part of this first suite — real-Ollama verification is left for a future manual/nightly smoke
  suite, not the everyday integration run.
- **Document read API coverage (Phase 2.8.2)** — `tests/integration/documents/read/test_postgres.py`
  (real Postgres: ordering/pagination, latest-job selection with multiple jobs, a document with
  no job, latest-failed-job selection, isolation between documents) and
  `tests/integration/documents/download/test_minio.py` (real MinIO: exact-byte download,
  missing-object → 409) — see `make test-document-read-integration`.
- **Containers and all state are removed after the test session** — nothing persists between
  runs, and nothing is written outside the ephemeral containers themselves.

Run it with:

```bash
make test-integration     # pytest -m integration -q
make verify-integration   # runs the integration suite (room for future integration-specific checks)
make test-storage          # storage-abstraction unit tests only (no Docker)
make test-storage-integration  # MinIO integration suite only (needs Docker)
make test-minio             # MinIO unit + integration tests (needs Docker for the latter)
make test-document-read     # document read/download API unit tests (no Docker)
make test-document-read-integration  # document read/download Postgres + MinIO + E2E coverage (needs Docker)
make test-ingestion-retry   # retry/stale-recovery unit tests (no Docker)
make test-ingestion-retry-integration  # retry/stale-recovery Postgres + Qdrant coverage (needs Docker)
make test-document-deletion  # full-document-deletion unit tests (no Docker)
make test-document-deletion-integration  # deletion Postgres + Qdrant + storage + E2E coverage (needs Docker)
```

- **Ingestion retry/stale-recovery coverage (Phase 2.8.3)** —
  `tests/integration/ingestion/test_retry_postgres.py` (real Postgres: the partial unique index
  actually rejecting a second active job, history preservation after a retry, plus one
  real-Postgres-and-Qdrant test forcing a first attempt to fail, confirming zero Qdrant points
  exist, then retrying to a real success) and `tests/integration/ingestion/test_concurrency.py`
  (two genuinely concurrent retry requests via `asyncio.gather` over independent sessions
  producing exactly one new active job, two concurrent stale-recovery calls never recovering
  the same row twice) — see `make test-ingestion-retry-integration`.
- **Full document deletion coverage (Phase 2.8.4)** — `tests/integration/documents/deletion/`:
  `test_postgres.py` (real Postgres: the partial unique index, append-only history, lifecycle
  derivation after completion/partial failure, migration correctness),
  `test_concurrency.py` (concurrent delete requests via `asyncio.gather` producing exactly one
  active job, concurrent worker claims never double-processing a row — kept separate from
  ordinary persistence tests, matching `tests/integration/ingestion/test_concurrency.py`'s
  concurrency-stress convention), `test_qdrant.py` (real Qdrant: full tracked-collection cleanup
  including a historical pending/failed `VectorCleanupJob` collection, an unrelated document's
  vectors surviving, idempotent re-deletion, a real forced partial-collection failure blocking
  storage deletion), and `test_storage.py` (real LocalFileStorage and real Testcontainers MinIO:
  exact-object deletion, already-missing-object idempotency, provider-failure partial state,
  identical Local/MinIO contract) — see `make test-document-deletion-integration`.

## Backend E2E tests

A separate suite lives under `tests/e2e/backend/` (marked `@pytest.mark.e2e`) — it is **not**
part of `make test`/`make verify`, and it is a distinct suite from `tests/integration/`.

- **Covers the complete backend user flow through real HTTP**: document upload
  (`POST /api/v1/documents`) → ingestion (the real `IngestionWorker`: extraction, chunking,
  embedding, Qdrant upsert) → retrieval/orchestration → the streaming chat SSE endpoint
  (`POST /api/v1/chat`), consumed incrementally so event order (`metadata` → `token`(s) → `done`)
  is genuinely exercised, not just inspected as one buffered string. It also covers validation
  errors, the decision layer's clarification/out-of-scope/direct-LLM/no-relevant-results paths,
  a mid-stream LLM failure, an ingestion failure, and liveness staying up when readiness fails.
- **Docker is required** — like the integration suite, Testcontainers for Python starts real,
  isolated, temporary Postgres and Qdrant containers on ephemeral (dynamically assigned) ports,
  with an isolated database and Qdrant collection per test.
- **The repository's main `docker-compose.yml` is not used** — no fixed ports, no
  shared/persistent Compose volumes. Every container is started and torn down by Testcontainers.
- **Real Postgres and real Qdrant, deterministic fake AI providers**: the FastAPI app runs for
  real behind a real ASGI HTTP client, with real extraction/chunking/decision/prompt-building/
  Qdrant code paths — only the embedding model and the chat LLM are swapped for deterministic
  fakes (`FakeEmbeddingProvider`, `FakeStreamingLLMProvider`), via monkeypatching the provider
  factory each consuming module already imports, never a production-code branch on `APP_ENV`.
  **No real Ollama container runs and no model is pulled.**
- **Containers and all state are removed after the test session.**
- **MinIO backend E2E coverage** — `tests/e2e/backend/test_minio_e2e.py` runs the same
  upload → ingestion → retrieval → streaming chat flow through the real HTTP boundary with
  `FILE_STORAGE_PROVIDER=minio`, against a real, ephemeral MinIO container (Testcontainers,
  dynamic port, unique bucket per test, no persistent volume), selected purely through the app's
  real `Settings`/`create_file_storage()` dependency chain — never a hand-substituted storage
  instance. It verifies the uploaded object exists in MinIO under the `Document` row's real
  `storage_key` with byte-identical content, that Hebrew/Unicode filenames and content survive the
  full round trip, that citation/source identity and the SSE event contract are unaffected by the
  storage provider, and that no MinIO implementation detail (bucket name, endpoint, credentials)
  leaks into the public response. Runs under both `RAG_ENGINE=custom` and `RAG_ENGINE=langchain`.
- **Document read API backend E2E coverage (Phase 2.8.2)** —
  `tests/e2e/backend/documents/read/test_local.py` (local storage) and
  `tests/e2e/backend/documents/read/test_minio.py` (real MinIO) drive
  upload → ingestion → list → detail → ingestion-status → download over the real HTTP boundary,
  including a forced-failure scenario asserted through `GET .../failure`, exact-byte download
  comparison, and a Hebrew filename round-trip through `Content-Disposition`. These are read-only
  document APIs (they touch only `Document`/`IngestionJob`/`FileStorage`, never `RagEngine`), so
  — unlike the RAG-engine-parity/multilingual E2E suites — they do not need to run under both
  `RAG_ENGINE` settings.
- **Ingestion retry/stale-recovery E2E coverage (Phase 2.8.3)** —
  `tests/e2e/backend/ingestion/test_retry_recovery.py` drives
  `POST .../ingestion/retry` over real HTTP after a real forced/transient failure and a
  manufactured stale-`PROCESSING` row, confirming both the retry contract and that history stays
  visible through the existing read APIs.
- **Full document deletion E2E coverage (Phase 2.8.4)** — `tests/e2e/backend/documents/deletion/`,
  organized by user-visible workflow rather than infrastructure: `test_successful_deletion.py`
  drives `DELETE /api/v1/documents/{id}` and `GET .../deletion` over real HTTP, then executes the
  scheduled job with a real `DocumentDeletionWorker` against the real ephemeral Qdrant container
  and a real `LocalFileStorage` (vectors and object removed, lifecycle becomes `deleted`, download
  returns `410`, chunks no longer searchable via a real `search_similar()` call);
  `test_partial_failures.py` covers a forced real Qdrant delete failure (lifecycle becomes
  `deletion_failed`, the object stays downloadable, no false success) and a forced storage failure
  followed by a successful retry (vectors removed once, storage cleanup completes on the second
  attempt); `test_concurrent_requests.py` covers two genuinely concurrent `DELETE` requests
  (`asyncio.gather`) converging on exactly one active job; `test_deleted_document_behavior.py`
  covers a deleted document rejecting `POST .../ingestion/retry` with `409`. Shared
  upload-and-ingest/execute-deletion helpers live in `support.py` within that same directory.

Run it with:

```bash
make test-e2e-backend     # pytest -m e2e tests/e2e/backend -q (includes the MinIO E2E test)
make verify-e2e-backend   # runs the backend E2E suite (room for future E2E-specific checks)
make test-e2e-backend-minio    # the MinIO backend E2E test only (needs Docker)
make verify-e2e-backend-minio  # runs the MinIO backend E2E test (room for future checks)
make test-document-read-integration  # document read/download Postgres + MinIO + E2E coverage
```

## Pre-commit verification

Install a git hook that runs `make verify` automatically before every commit, so a broken
build/test/lint/type-check can never be committed by accident — and so this doesn't depend on
remembering to run checks manually (including when Claude Code is making the change):

```bash
./scripts/install-git-hooks.sh
```

This copies `.githooks/pre-commit` into `.git/hooks/pre-commit` (git hooks live outside version
control, so they must be installed locally — this script is the one-time setup step). Once
installed, every `git commit` runs `make verify` first and **blocks the commit if it fails**. The
hook only checks — it never auto-fixes files (e.g. it runs `ruff check .`, not
`ruff check --fix .`) and never stages or commits anything on its own.

You can always run the same check manually, without committing:

```bash
make verify
```

If you ever need to skip the hook in an emergency, `git commit --no-verify` bypasses it — but
prefer fixing the underlying issue over skipping the check.

## Troubleshooting

- **`app` fails to start / connection refused to postgres|redis|qdrant|ollama**: those services
  take a few seconds to become ready. `docker-compose.yml` uses `depends_on` (start order only, not
  a readiness check) — if the app crashes on startup, retry with
  `docker compose up --build app` after confirming the dependency logs show it's ready.
- **Port already in use**: another local process is bound to `8000`, `5432`, `6379`, `6333`, or
  `11434`. Stop it, or change the host-side port mapping in `docker-compose.yml`
  (`"HOST:CONTAINER"`).
- **Checking service logs**: `docker compose logs <service> --tail 50`.
- **Verifying internal networking** (from inside the `app` container):
  ```bash
  docker compose exec app python -c "import socket; socket.create_connection(('postgres', 5432), timeout=5)"
  docker compose exec app python -c "import urllib.request; urllib.request.urlopen('http://ollama:11434', timeout=5)"
  ```
- **Rebuilding after dependency changes**: `docker compose up --build app` (Python deps are
  installed at image build time, not at container start).
- **Full reset** (drops Postgres/Qdrant/Ollama volumes — deletes local data):
  `docker compose down -v`.

## GitHub CLI / PR workflow

Pull requests are created from the terminal with the [GitHub CLI](https://cli.github.com/)
(`gh`), not the web UI. Before opening a PR:

```bash
gh --version       # verify the CLI is installed
gh auth status     # verify you're authenticated
```

PRs should be small and focused (one milestone per PR) and their description should include
verification results (test/lint/type-check output), not just a claim that checks passed. This
repository uses a PR template at
[.github/pull_request_template.md](.github/pull_request_template.md) — the web UI picks it up
automatically, and PRs opened via `gh pr create` from the terminal should follow that same
template (e.g. via `gh pr create --body-file <filled-template>`) rather than an ad-hoc
description. PR titles and the full description format (Summary, Why, Changes, Verification,
Explicit exclusions, Next recommended milestone) are defined in [CLAUDE.md](CLAUDE.md) under
"Pull Request Workflow" — follow that format for every PR.

## Current milestone status

Infrastructure scaffold complete and verified: FastAPI app, Docker Compose topology (app,
postgres, redis, qdrant, ollama), configuration, async DB wiring, Alembic scaffold, and abstract
provider interfaces. On top of that, Ollama reachability and model-availability checks are
implemented (`GET /api/v1/providers/ollama/health`), a concrete `OllamaEmbeddingProvider` can
embed text via `/api/embeddings`, a concrete `OllamaLLMProvider` can stream completions via
`/api/generate`, and a concrete `QdrantVectorStore` can create collections, upsert vectors, and
run similarity search over Qdrant's HTTP API — all resolved through a configuration-driven
provider factory (`app/rag/providers/provider_factory.py`) instead of being hardcoded to a single
backend — covered by tests with mocked HTTP transports (and a manual end-to-end smoke test
against a real Qdrant container). Explicit future-provider stubs (`OpenAIProvider`,
`GeminiProvider`, `AnthropicProvider`) exist so those `LLM_PROVIDER` values fail clearly instead
of falling back to Ollama. `LLM_MODEL` selects the chat model independently of `LLM_PROVIDER`
(falling back to `OLLAMA_CHAT_MODEL`), while `OLLAMA_EMBEDDING_MODEL` stays fixed for embeddings.
A rule-based RAG decision layer (`RuleBasedRagDecider`) can classify a question as needing
retrieval, going direct to the LLM, needing clarification, or being out of scope — deterministic
routing, no LLM call involved. `POST /api/v1/documents` accepts a file upload (Hebrew/Unicode
filenames included), stores it locally under a generated safe filename, and creates `Document` +
`pending` `IngestionJob` rows — returning `202` without parsing, chunking, embedding, or
upserting anything. `IngestionWorker` claims and resolves those pending jobs (Postgres row-level
locking, idempotent by construction), and its real first processing step,
`DocumentTextExtractor`, extracts text from `.txt`/`.md`/`.pdf`/`.docx`/`.xlsx` files
(page-by-page with page numbers for PDFs, sheet-by-sheet with sheet names for XLSX,
Hebrew/Unicode preserved throughout), and its second step, `DocumentChunker`, splits that text
into fixed-size, overlapping, word-boundary-aware, deterministic chunks (`CHUNK_SIZE`/
`CHUNK_OVERLAP`). Its third and fourth steps now embed each chunk via `get_embedding_provider()`
and upsert the resulting vectors into Qdrant via `get_vector_store()`
(`QDRANT_COLLECTION_NAME`/`VECTOR_SIZE`, created if missing), preserving `document_id`,
`chunk_id`, `text`, `source`, and `page_number`/`sheet_name` as payload metadata — marking the
job `completed` on success or `failed` with the error stored on failure at any step (extraction,
chunking, embedding, or upsert). An internal `RetrievalService` closes the read side of the same
loop — given a query, it embeds it via `get_embedding_provider()` and searches
`QDRANT_COLLECTION_NAME` via `get_vector_store()`, returning ranked `VectorSearchResult`s
(`RETRIEVAL_TOP_K`/`RETRIEVAL_SCORE_THRESHOLD`). On top of that, an internal `RagPromptBuilder`
turns a question and those ranked results into a deterministic `BuiltRagPrompt`
(`system_prompt`/`user_prompt`/`context`/`sources`) with stable `[S1]`/`[S2]`/... source labels,
filename/page/sheet attribution, and instructions to answer only from context and say when the
answer isn't present. An internal `RagOrchestrator` wires the decision layer, retrieval service,
prompt builder, and streaming `LLMProvider` together: `stream_answer(question)` routes via
`RuleBasedRagDecider`, and for `NEEDS_RETRIEVAL`/`DIRECT_LLM` streams the LLM's answer token by
token (for `CLARIFICATION_NEEDED`/`OUT_OF_SCOPE` it streams a fixed message with no retrieval and
no LLM call) — with no silent fallback between decisions or providers on failure. `POST
/api/v1/chat` now exposes it publicly as Server-Sent Events (`metadata`/`token`/`done`/`error`),
via a thin route with no orchestration logic of its own — see "Streaming chat endpoint" above.
Conversation memory/multi-turn context and a model-override parameter are not implemented. A
Testcontainers-based integration suite (`tests/integration/`, `make test-integration`) backs
migrations, `IngestionWorker`'s real Postgres locking behavior, and `QdrantVectorStore`'s real
HTTP contract with genuine ephemeral containers — separate from the fast unit suite and from
`docker-compose.yml` — while still using fake, deterministic AI providers instead of a real
Ollama model. On top of the business API, four **unversioned** platform endpoints (`GET /health`,
`/health/live`, `/health/ready`, `/health/dependencies`) now give Kubernetes/load
balancers/monitoring a stable operational contract independent of API versioning — readiness
checks Postgres, Qdrant, Ollama, and its two configured models for real, with a short timeout and
no secrets/connection details ever in the response — see "Platform health and readiness" above.
A backend E2E suite (`tests/e2e/backend/`, `make test-e2e-backend`) now drives the full flow —
upload → ingestion → retrieval/orchestration → streaming chat — through real HTTP against the
same kind of ephemeral Postgres/Qdrant containers, with deterministic fake embedding/LLM
providers swapped in via dependency/provider-factory overrides rather than any `APP_ENV` branch
in production code; see "Backend E2E tests" above. A `RagEngine` abstraction
(`app/rag/engine.py`) now sits behind `POST /api/v1/chat`, with `CustomRagEngine` (wrapping
`RagOrchestrator` unchanged) as the default and an optional `LangChainRagEngine` selectable via
`RAG_ENGINE=langchain` — both share the same provider factory, `RetrievalService`,
`RagPromptBuilder`, Qdrant collection, and embedding model, and both produce an identical public
API/SSE contract; see "RAG engine compatibility layer" above. Multilingual (Hebrew + English)
retrieval and language-aware prompting are now shared platform capabilities: a versioned
`EmbeddingIndexConfig` derives a deterministic Qdrant collection identity so incompatible
embeddings can never share a collection, `Document` rows track exactly which configuration they
were indexed with (staleness detection + a backend re-index capability), a deterministic
`ScriptBasedLanguageDetector` resolves Hebrew/English from a question, and a shared
`PromptCatalog`/`PromptProvider` resolves every fixed/governed prompt through both `RagEngine`
implementations — see "Multilingual RAG foundation" above. A provider-neutral `FileStorage`
abstraction (`app/storage/`) now sits behind upload/ingestion/extraction/re-index: `save`/`read`/
`delete`/`exists`/`get_metadata`/`generate_download_url`, backed by `LocalFileStorage` (default)
or `MinioFileStorage` (S3-compatible, selected via `FILE_STORAGE_PROVIDER=minio`), resolved
through one factory (`create_file_storage()`) exactly like the AI provider factory — see "Storage
abstraction" above and "Storage Abstraction (Phase 2.6/2.7)" in ARCHITECTURE.md. MinIO is now
covered end to end: real adapter tests (`tests/integration/test_minio_storage.py`), a real
ingestion-pipeline test (`tests/integration/ingestion/test_worker_minio.py`), and a focused
public backend E2E test (`tests/e2e/backend/test_minio_e2e.py`, `make test-e2e-backend-minio`)
that drives `POST /api/v1/documents` → real MinIO → ingestion → `POST /api/v1/chat` through real
HTTP with `FILE_STORAGE_PROVIDER=minio`, under both `RAG_ENGINE=custom` and
`RAG_ENGINE=langchain` — see "Backend E2E tests" above. Five read-only document APIs (Phase
2.8.2) now let a client inspect a document's lifecycle and download its original content —
`GET /api/v1/documents`, `GET /api/v1/documents/{id}`, `GET /api/v1/documents/{id}/ingestion`,
`GET /api/v1/documents/{id}/failure`, `GET /api/v1/documents/{id}/download` — backed by a new
`app/services/documents/query_service.py` query layer and a derived `DocumentLifecycleStatus`
(`uploaded`/`pending`/`processing`/`indexed`/`failed`, plus `deleting`/`deletion_failed`/`deleted`
as of Phase 2.8.4 below); see "Document read APIs and original download (Phase 2.8.2)" above and
in ARCHITECTURE.md. **These five endpoints remain strictly read-only.** Document lifecycle
*mutation* has two deliberate exceptions: `POST /api/v1/documents/{id}/ingestion/retry` (Phase
2.8.3) schedules a new ingestion attempt for a FAILED or stale-PROCESSING document, backed by a
real Postgres partial unique index (`ix_ingestion_jobs_one_active_per_document`) enforcing at
most one active job per document, and `app/services/ingestion/stale_recovery_service.py`'s
`recover_stale_ingestion_jobs()` (triggered manually via `scripts/recover_stale_ingestion_jobs.py`
/ `make recover-stale-ingestion-jobs`, no HTTP endpoint) recovers `PROCESSING` jobs abandoned by a
crashed worker — see "Ingestion retry and stale-job recovery" above and in ARCHITECTURE.md.
Retry never re-uploads content or touches Qdrant directly — vector idempotency is free by
construction (deterministic chunk/point IDs), verified against a real ephemeral Qdrant container.
`DELETE /api/v1/documents/{id}` and `GET /api/v1/documents/{id}/deletion` (Phase 2.8.4) are the
other exception: full, asynchronous, cross-system document deletion, backed by
`app/models/document_deletion_job.py`'s append-only `DocumentDeletionJob` ledger, its own real
Postgres partial unique index (`ix_document_deletion_jobs_one_active_per_document`), and
`DocumentDeletionWorker` (execution triggered by `scripts/process_pending_document_deletions.py`
/ `make process-pending-document-deletions`, mirroring `IngestionWorker`'s out-of-band execution
model exactly) — see "Full document deletion" above and in ARCHITECTURE.md. The `Document` row
and all `IngestionJob`/`VectorCleanupJob`/`DocumentDeletionJob` history are never physically
removed; only Qdrant vectors and the stored object are. Bulk re-index/reconciliation,
orphan-object cleanup, hash-based deduplication, frontend E2E, a real-Ollama smoke suite, a real
multilingual-model evaluation run, and a real scheduler deployment for stale
ingestion-recovery/deletion-execution remain future milestones — see "Integration tests" above,
and "Test architecture"/"What is intentionally not implemented yet" in
[ARCHITECTURE.md](ARCHITECTURE.md) for the full list of what's intentionally deferred.
