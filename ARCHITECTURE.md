# Architecture

## System overview

`documents-rag` is a local-first RAG (Retrieval-Augmented Generation) platform. Everything runs on
the user's machine via Docker Compose, with Ollama providing local LLM and embedding inference —
no external API calls or cloud dependencies.

This milestone is infrastructure plus one vertical slice: a FastAPI app wired to Postgres, Redis,
Qdrant, and Ollama, with a health endpoint, an Ollama health/model-availability check, and
placeholder provider interfaces. No ingestion, embedding, chat, or Qdrant indexing logic exists
yet.

## Services

| Service    | Image                    | Purpose (current)                                             | Purpose (future) |
|------------|--------------------------|------------------------------------------------------------------|-------------------|
| `app`      | built from `Dockerfile`  | FastAPI process: `/api/v1/health`, `/api/v1/providers/ollama/health` | RAG API: ingestion, retrieval, chat |
| `postgres` | `postgres:16-alpine`     | Relational store, wired via async SQLAlchemy                     | Document/session/metadata storage |
| `redis`    | `redis:7-alpine`         | Available on the network                                         | Caching, task queues |
| `qdrant`   | `qdrant/qdrant:latest`   | Available on the network                                         | Vector storage/search for embeddings |
| `ollama`   | `ollama/ollama:latest`   | Queried for reachability and model-availability only              | Local chat (`llama3.1`) and embedding (`nomic-embed-text`) inference |

The app queries Ollama's `/api/tags` endpoint (via `app/services/ollama_client.py`) to check
reachability and whether the configured models are pulled — it does not yet call
`/api/generate` or `/api/embeddings`. Postgres, Redis, and Qdrant are still unused beyond
connection configuration and abstract interfaces (`app/rag/providers/`).

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
| `OLLAMA_CHAT_MODEL`        | `llama3.1`                                                             | Checked for availability, not yet used to generate |
| `OLLAMA_EMBEDDING_MODEL`   | `nomic-embed-text`                                                     | Checked for availability, not yet used to embed |

## Current boundaries

- `app/api` — FastAPI routers: `/health` and `/providers/ollama/health`.
- `app/core` — configuration and cross-cutting concerns.
- `app/db` — SQLAlchemy async engine/session setup.
- `app/models` — ORM models (empty for now).
- `app/schemas` — Pydantic request/response schemas.
- `app/services` — business logic layer. Currently just `OllamaClient`
  (`app/services/ollama_client.py`), a thin async HTTP client scoped strictly to reachability and
  model-availability checks — it intentionally does not call generation or embedding endpoints.
- `app/rag/providers` — abstract interfaces for embedding, LLM, and vector store providers.
  Concrete implementations (Ollama, Qdrant) will be added in a later milestone. `OllamaClient` is
  deliberately kept separate from these interfaces so health checks don't get entangled with the
  generation/embedding contracts.
- `app/workers` — background job placeholders.

## What is intentionally not implemented yet

- Document ingestion/upload endpoints
- Chat/query endpoints
- Ollama generation calls (`/api/generate`) and embedding calls (`/api/embeddings`)
- Concrete `EmbeddingProvider`, `LLMProvider`, `VectorStore` implementations (Ollama/Qdrant clients)
- Qdrant collection creation
- Database models/migrations beyond the empty Alembic scaffold
- Auth, rate limiting, observability/logging pipeline

These land in later milestones once the infrastructure is confirmed stable.
