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
Document ingestion, a public chat/query endpoint, and any pipeline wiring these providers
together into a full RAG flow are not yet implemented — see
[ARCHITECTURE.md](ARCHITECTURE.md) for the full list of what's
intentionally deferred.
