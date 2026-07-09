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

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
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
cp .env.example .env
docker compose up --build
```

This starts `app`, `postgres`, `redis`, `qdrant`, and `ollama`. The app is available at
http://localhost:8000, with health check at `GET /api/v1/health`. Verified working end-to-end:
all five containers start, the health endpoint responds `{"status":"ok","environment":"local"}`
from the host, and the `app` container can reach `postgres:5432`, `redis:6379`, `qdrant:6333`,
and `ollama:11434` over the internal Compose network.

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
metadata: `document_id`, `chunk_id`, `text`, `source`, and an optional `page_number`. It's an
internal provider only — no document ingestion, upload, chat, or SSE endpoint touches it yet.

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
default: `DocumentTextExtractor` — see below; still no chunking, embedding, or Qdrant upsert),
then resolves it to `completed` on success or `failed` (with the error message stored) on any
exception. It's idempotent: a job that's already `completed` or `failed` is never selected again
by the claim query, so calling `process_next_job()` repeatedly never re-processes it. The worker
never calls `EmbeddingProvider`, `LLMProvider`, `VectorStore`, or the provider factory.

### Document text extraction

`DocumentTextExtractor` (`app/services/document_text_extractor.py`) loads a document's stored
file and extracts its raw text. **Routing is currently by file extension only — this is MVP
behavior, not final validation:**

| Extension | Handler |
|-----------|---------|
| `.txt`    | UTF-8 plain text |
| `.md`     | UTF-8 markdown/plain text (no Markdown parsing) |
| `.pdf`    | `pypdf`, page by page, 1-indexed `page_number` preserved |
| `.docx`   | `python-docx`, plain paragraph text, a single page |
| `.xlsx`   | `openpyxl`, sheet by sheet, each sheet's name in `sheet_name` |

There's no `content_type`/MIME validation and no content sniffing yet — a mismatched extension
(e.g. an `.xlsx` renamed to `.txt`) is parsed as whatever the extension claims. **Future
hardening**: `content_type` validation, real MIME sniffing, extension/content mismatch
detection, and generally safer file validation ahead of parsing untrusted uploads.

```python
from app.services.document_text_extractor import DocumentTextExtractor

extracted = await DocumentTextExtractor().extract(document)
for page in extracted.pages:
    print(page.page_number, page.sheet_name, page.text)
```

Raises `DocumentTextExtractionError` for a missing stored file, an unsupported extension, or
empty/whitespace-only extracted text — `IngestionWorker` catches this and marks the job
`failed` with the error message stored. UTF-8/Unicode content (Hebrew included) is preserved
exactly across all five file types. This is the ingestion worker's real first processing step,
but the extracted text isn't stored anywhere yet — chunking and persistence come in a later
milestone.

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
Hebrew/Unicode preserved throughout) — marking the job `completed` on success or `failed` with
the error stored on failure. Chunking, embedding generation, and Qdrant upsert are not yet
wired in — extracted text is discarded once the step returns. A public chat/query endpoint and
any pipeline wiring the decision layer, providers, vector store, and ingestion worker together
into a full RAG flow are not yet implemented — see [ARCHITECTURE.md](ARCHITECTURE.md) for the
full list of what's intentionally
deferred.
