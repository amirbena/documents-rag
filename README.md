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
- Ollama (local LLM + embeddings): `llama3.1` for chat, `nomic-embed-text` for embeddings
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
   curl http://localhost:8000/api/v1/health
   # {"status":"ok","environment":"local"}
   ```
8. **Pull the required Ollama models** (see "Running with Docker Compose" below):
   ```bash
   docker compose exec ollama ollama pull llama3.1
   docker compose exec ollama ollama pull nomic-embed-text
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
curl http://localhost:8000/api/v1/health
# {"status":"ok","environment":"local"}
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
http://localhost:8000, with health check at `GET /api/v1/health`. Verified working end-to-end:
all five containers start, the health endpoint responds `{"status":"ok","environment":"local"}`
from the host, and the `app` container can reach `postgres:5432`, `redis:6379`, `qdrant:6333`,
and `ollama:11434` over the internal Compose network.

Once the `app`/`postgres` containers are up, run the Alembic migrations — see "Database
migrations" below — before pulling Ollama models or testing document upload.

To pull the required Ollama models after the `ollama` service is up:

```bash
docker compose exec ollama ollama pull llama3.1
docker compose exec ollama ollama pull nomic-embed-text
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
`OLLAMA_CHAT_MODEL` for backward compatibility. `OLLAMA_EMBEDDING_MODEL` is separate and fixed —
it's never affected by `LLM_MODEL`, since embeddings must stay on one model to keep previously
computed vectors valid.

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

This saves the file under `storage/documents/` (with a generated, filesystem-safe stored
filename — never the raw original name), creates a `Document` row (with the original filename
preserved exactly, Hebrew/Unicode included), and creates an `IngestionJob` row with
`status=pending`. **Nothing is parsed, chunked, embedded, or upserted into Qdrant inside the
request.** An empty (zero-byte) file is rejected with `400` before any row is created.

### Ingestion worker

`IngestionWorker` (`app/services/ingestion_worker.py`) is an internal service — no public API —
that claims and resolves one `pending` `IngestionJob` at a time:

```python
from app.services.ingestion_worker import IngestionWorker

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

`DocumentTextExtractor` (`app/services/document_text_extractor.py`) loads a document's stored
file and extracts its raw text. **It routes by file extension and validates each file's basic
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
from app.services.document_text_extractor import DocumentTextExtractor

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

`DocumentChunker` (`app/services/document_chunker.py`) takes an `ExtractedDocument` and splits
it into fixed-size, overlapping, word-boundary-aware chunks — no embedding, no Qdrant upsert, no
retrieval:

```python
from app.services.document_chunker import DocumentChunker

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

### Streaming chat endpoint

`POST /api/v1/chat` (`app/api/v1/routes/chat.py`) is a thin route that streams
`RagOrchestrator.stream_answer(question)` back as Server-Sent Events — no decision, retrieval,
or prompt-building logic in the route itself, and no direct provider calls:

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
| `error` | At most once, if `RagOrchestrator.stream_answer(...)` raises after streaming has started | `message` (a fixed, safe string — never a stack trace, prompt, secret, credential, internal URL, or provider response body), `status: "failed"` |

An `error` event ends the stream — no `done` event follows it. The route does not buffer the
full answer: each `OrchestratorToken` is written to the response as soon as the orchestrator
yields it, so `curl -N` (no-buffer mode, shown above) prints tokens as they arrive rather than
all at once at the end.

## Verification

A `Makefile` wraps all quality gates behind one command:

```bash
make test        # pytest -q
make lint         # ruff check .
make typecheck    # mypy app
make compose      # docker compose config
make verify       # runs test, lint, typecheck, compose, in order — stops at the first failure
```

`make verify` is the standard pre-commit/pre-PR check. If `make` isn't available, run the
underlying commands directly:

```bash
pytest -q
ruff check .
ruff check --fix .    # lint + autofix
mypy app
docker compose config
```

All four gates (`pytest`, `ruff check .`, `mypy app`, `docker compose config`) must pass cleanly
before committing.

Run `make help` any time for a quick summary of these commands.

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
Conversation memory/multi-turn context and a model-override parameter are not implemented — see
[ARCHITECTURE.md](ARCHITECTURE.md) for the full list of what's intentionally deferred.
