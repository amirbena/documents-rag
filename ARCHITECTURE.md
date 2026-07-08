# Architecture

## System overview

`documents-rag` is a local-first RAG (Retrieval-Augmented Generation) platform. Everything runs on
the user's machine via Docker Compose, with Ollama providing local LLM and embedding inference —
no external API calls or cloud dependencies.

This milestone is infrastructure plus three vertical slices: a FastAPI app wired to Postgres,
Redis, Qdrant, and Ollama, with a health endpoint, an Ollama health/model-availability check, a
concrete Ollama-backed embedding provider, a concrete streaming Ollama-backed LLM provider, and
placeholder interfaces for the rest. No ingestion, public chat endpoint, or Qdrant indexing logic
exists yet.

## Services

| Service    | Image                    | Purpose (current)                                             | Purpose (future) |
|------------|--------------------------|------------------------------------------------------------------|-------------------|
| `app`      | built from `Dockerfile`  | FastAPI process: `/api/v1/health`, `/api/v1/providers/ollama/health` | RAG API: ingestion, retrieval, chat |
| `postgres` | `postgres:16-alpine`     | Relational store, wired via async SQLAlchemy                     | Document/session/metadata storage |
| `redis`    | `redis:7-alpine`         | Available on the network                                         | Caching, task queues |
| `qdrant`   | `qdrant/qdrant:latest`   | Available on the network                                         | Vector storage/search for embeddings |
| `ollama`   | `ollama/ollama:latest`   | Health/model checks + embeddings (`nomic-embed-text`) + streaming generation (`llama3.1`) | Backing a future public chat endpoint |

The app queries Ollama's `/api/tags` endpoint (via `app/services/ollama_client.py`) to check
reachability and whether the configured models are pulled, calls `/api/embeddings` (via
`app/rag/providers/ollama_embedding_provider.py`) to embed text with `OLLAMA_EMBEDDING_MODEL`,
and calls `/api/generate` with `stream=true` (via `app/rag/providers/ollama_llm_provider.py`) to
stream completions from `OLLAMA_CHAT_MODEL`. The LLM provider is internal-only — there is no
public chat or SSE endpoint yet. Callers resolve these providers through
`app/rag/providers/provider_factory.py` rather than importing Ollama classes directly — see
"Provider factory" below. Postgres, Redis, and Qdrant are still unused beyond connection
configuration and abstract interfaces (`app/rag/providers/`).

## Provider factory

`app/rag/providers/provider_factory.py` resolves which concrete provider class to construct,
based on three config variables, so the rest of the codebase depends on the `EmbeddingProvider` /
`LLMProvider` / `VectorStore` interfaces rather than being coupled to Ollama or Qdrant directly:

- `get_embedding_provider()` — `EMBEDDING_PROVIDER` (`"ollama"` → `OllamaEmbeddingProvider`)
- `get_llm_provider()` — `LLM_PROVIDER` (`"ollama"` → `OllamaLLMProvider`)
- `get_vector_store()` — `VECTOR_STORE_PROVIDER` (`"qdrant"` is recognized but raises
  `NotImplementedError` — no concrete `VectorStore` exists yet)

An unrecognized provider name raises `UnsupportedProviderError` (a `ValueError`) with a message
naming the offending value and the supported provider(s). All Ollama-specific logic (HTTP calls,
error handling) stays inside the Ollama provider classes — the factory only selects and
constructs; it never reimplements provider behavior, and business/service code should resolve
providers through it rather than importing `OllamaEmbeddingProvider`/`OllamaLLMProvider` directly.

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
| `DATABASE_URL`             | `postgresql+asyncpg://postgres:postgres@postgres:5432/rag_db`         | Async SQLAlchemy engine |
| `REDIS_URL`                | `redis://redis:6379/0`                                                | Not yet consumed |
| `QDRANT_URL`               | `http://qdrant:6333`                                                  | Not yet consumed |
| `OLLAMA_BASE_URL`          | `http://ollama:11434`                                                 | Used by `OllamaClient` for health/model checks |
| `OLLAMA_CHAT_MODEL`        | `llama3.1`                                                             | Checked for availability; used by `OllamaLLMProvider` to stream completions |
| `OLLAMA_EMBEDDING_MODEL`   | `nomic-embed-text`                                                     | Checked for availability; used by `OllamaEmbeddingProvider` to embed |
| `LLM_PROVIDER`             | `ollama`                                                               | Selects the `LLMProvider` implementation via the provider factory |
| `EMBEDDING_PROVIDER`       | `ollama`                                                               | Selects the `EmbeddingProvider` implementation via the provider factory |
| `VECTOR_STORE_PROVIDER`    | `qdrant`                                                               | Recognized but not yet implemented — `get_vector_store()` raises `NotImplementedError` |

## Current boundaries

- `app/api` — FastAPI routers: `/health` and `/providers/ollama/health`.
- `app/core` — configuration and cross-cutting concerns.
- `app/db` — SQLAlchemy async engine/session setup.
- `app/models` — ORM models (empty for now).
- `app/schemas` — Pydantic request/response schemas.
- `app/services` — business logic layer. Currently just `OllamaClient`
  (`app/services/ollama_client.py`), a thin async HTTP client scoped strictly to reachability and
  model-availability checks — it intentionally does not call generation or embedding endpoints.
- `app/rag/providers` — abstract interfaces for embedding, LLM, and vector store providers, a
  `provider_factory.py` that resolves the configured implementation for each (see "Provider
  factory" above), and two concrete implementations:
  - `OllamaEmbeddingProvider` (`app/rag/providers/ollama_embedding_provider.py`) — calls
    `POST /api/embeddings` for `OLLAMA_EMBEDDING_MODEL` only.
  - `OllamaLLMProvider` (`app/rag/providers/ollama_llm_provider.py`) — calls
    `POST /api/generate` with `stream=true` for `OLLAMA_CHAT_MODEL`, exposing
    `stream_generate(prompt) -> AsyncIterator[str]` (yields text chunks as Ollama streams them)
    and `generate(prompt) -> str` (joins the streamed chunks, satisfying the abstract
    `LLMProvider` contract). Internal-only — no ingestion, no Qdrant writes, no public chat/SSE
    endpoint.

  `VectorStore` remains abstract-only. `OllamaClient` (health checks) is deliberately kept
  separate from these provider interfaces so health checks don't get entangled with the
  generation/embedding contracts.
- `app/workers` — background job placeholders.

## What is intentionally not implemented yet

- Document ingestion/upload endpoints
- A public chat/query endpoint (including SSE streaming to clients)
- A public API endpoint for embeddings (the provider is internal-only for now)
- Concrete `VectorStore` implementation (Qdrant client)
- Qdrant collection creation
- Database models/migrations beyond the empty Alembic scaffold
- Auth, rate limiting, observability/logging pipeline

These land in later milestones once the infrastructure is confirmed stable.
