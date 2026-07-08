# Architecture

## System overview

`documents-rag` is a local-first RAG (Retrieval-Augmented Generation) platform. Everything runs on
the user's machine via Docker Compose, with Ollama providing local LLM and embedding inference —
no external API calls or cloud dependencies.

This milestone is infrastructure only: a FastAPI app wired to Postgres, Redis, Qdrant, and Ollama,
with a health endpoint and placeholder provider interfaces. No ingestion, embedding, or chat logic
exists yet.

## Services

| Service    | Image                    | Purpose (current)                         | Purpose (future) |
|------------|--------------------------|---------------------------------------------|-------------------|
| `app`      | built from `Dockerfile`  | FastAPI process, `/api/v1/health`            | RAG API: ingestion, retrieval, chat |
| `postgres` | `postgres:16-alpine`     | Relational store, wired via async SQLAlchemy | Document/session/metadata storage |
| `redis`    | `redis:7-alpine`         | Available on the network                     | Caching, task queues |
| `qdrant`   | `qdrant/qdrant:latest`   | Available on the network                     | Vector storage/search for embeddings |
| `ollama`   | `ollama/ollama:latest`   | Available on the network                     | Local chat (`llama3.1`) and embedding (`nomic-embed-text`) inference |

The app does not yet query Postgres, Redis, Qdrant, or Ollama — it only holds connection
configuration and abstract interfaces (`app/rag/providers/`) for them.

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
| `OLLAMA_BASE_URL`          | `http://ollama:11434`                                                 | Not yet consumed |
| `OLLAMA_CHAT_MODEL`        | `llama3.1`                                                             | Not yet consumed |
| `OLLAMA_EMBEDDING_MODEL`   | `nomic-embed-text`                                                     | Not yet consumed |

## Current boundaries

- `app/api` — FastAPI routers (currently just `/health`).
- `app/core` — configuration and cross-cutting concerns.
- `app/db` — SQLAlchemy async engine/session setup.
- `app/models` — ORM models (empty for now).
- `app/schemas` — Pydantic request/response schemas.
- `app/services` — business logic layer (empty for now).
- `app/rag/providers` — abstract interfaces for embedding, LLM, and vector store providers.
  Concrete implementations (Ollama, Qdrant) will be added in a later milestone.
- `app/workers` — background job placeholders.

## What is intentionally not implemented yet

- Document ingestion/upload endpoints
- Chat/query endpoints
- Concrete `EmbeddingProvider`, `LLMProvider`, `VectorStore` implementations (Ollama/Qdrant clients)
- Qdrant collection creation
- Database models/migrations beyond the empty Alembic scaffold
- Auth, rate limiting, observability/logging pipeline

These land in later milestones once the infrastructure is confirmed stable.
