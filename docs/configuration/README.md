# Configuration

Environment variables, defaults, and required-vs-optional configuration. `app/core/config.py`
(`Settings`) is the single source of truth for defaults — this table must be kept in sync with it,
not the other way around. Set via `docker-compose.yml` for containers, or `.env` (copied from
`.env.example`) for local runs outside Docker.

No secrets or real credentials are included below — every credential-shaped value shown is a
local-development-only default (e.g. MinIO's `minioadmin`/`minioadmin`), never reused elsewhere.

## Core / runtime

| Variable | Default | Required? | Notes |
|---|---|---|---|
| `APP_ENV` | `local` | Optional | Read by application/provider configuration |
| `LOG_LEVEL` | `INFO` | Optional | Not yet wired to a logger |
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@postgres:5432/rag_db` | **Required for readiness** | Async SQLAlchemy engine |
| `REDIS_URL` | `redis://redis:6379/0` | Optional | Checked but not required for readiness — no application code path reads/writes it yet |

## Providers

| Variable | Default | Required? | Notes |
|---|---|---|---|
| `LLM_PROVIDER` | `ollama` | Optional | `openai`/`gemini`/`anthropic` recognized but stub-only |
| `LLM_MODEL` | *(unset)* | Optional | Falls back to `OLLAMA_CHAT_MODEL` if unset |
| `EMBEDDING_PROVIDER` | `ollama` | Optional | |
| `VECTOR_STORE_PROVIDER` | `qdrant` | Optional | |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | **Required for readiness** | |
| `OLLAMA_CHAT_MODEL` | `llama3.1` | **Required for readiness** | |
| `OLLAMA_EMBEDDING_MODEL` | `bge-m3` | **Required for readiness** | Fixed, not selectable via `LLM_MODEL` — always used by `OllamaEmbeddingProvider` directly |
| `QDRANT_URL` | `http://qdrant:6333` | **Required for readiness** | |

## Indexing / embedding configuration

| Variable | Default | Required? | Notes |
|---|---|---|---|
| `EMBEDDING_MODEL` | *(unset)* | Optional | Generic override; falls back to `OLLAMA_EMBEDDING_MODEL` |
| `VECTOR_SIZE` | `1024` | Optional | Must match the embedding provider's output size (`bge-m3`=1024, `nomic-embed-text`=768) |
| `EMBEDDING_VERSION` | `v2` | Optional | Bump whenever the embedding model/dimension changes meaningfully |
| `CHUNKING_VERSION` | `v1` | Optional | Bump whenever chunking parameters/algorithm change meaningfully |
| `QDRANT_COLLECTION_NAME` | `documents` | Optional | A **prefix**, not the literal collection name — see [docs/storage/](../storage/README.md) |
| `CHUNK_SIZE` | `1000` | Optional | Characters |
| `CHUNK_OVERLAP` | `200` | Optional | Characters |

## Retrieval

| Variable | Default | Required? | Notes |
|---|---|---|---|
| `RETRIEVAL_TOP_K` | `5` | Optional | Default result count when no explicit `limit` is passed |
| `RETRIEVAL_SCORE_THRESHOLD` | *(unset)* | Optional | Unset disables score filtering |
| `RAG_ENGINE` | `custom` | Optional | `langchain` is the only other recognized value |

## Multilingual

| Variable | Default | Required? | Notes |
|---|---|---|---|
| `DEFAULT_RESPONSE_LANGUAGE` | `en` | Optional | Must be `he` or `en` |
| `PROMPT_CATALOG_VERSION` | `v2` | Optional | Stamped onto every resolved prompt |

## Object storage

| Variable | Default | Required? | Notes |
|---|---|---|---|
| `FILE_STORAGE_PROVIDER` | `local` | Optional | `minio` is the only other recognized value |
| `LOCAL_STORAGE_ROOT` | `storage/documents` | Optional | Only read when `local` |
| `MINIO_ENDPOINT` | *(unset)* | **Required when `minio`** | |
| `MINIO_ACCESS_KEY` | *(unset)* | **Required when `minio`** | Never logged |
| `MINIO_SECRET_KEY` | *(unset)* | **Required when `minio`** | Never logged |
| `MINIO_BUCKET` | *(unset)* | **Required when `minio`** | |
| `MINIO_SECURE` | `false` | Optional | |
| `MINIO_REGION` | *(unset)* | Optional | |
| `MINIO_PRESIGNED_URL_EXPIRY_SECONDS` | `3600` | Optional | |
| `MINIO_CREATE_BUCKET_IF_MISSING` | `true` | Optional | |

## Ingestion recovery

| Variable | Default | Required? | Notes |
|---|---|---|---|
| `INGESTION_STALE_AFTER_SECONDS` | `900` | Optional | Approximation, not a liveness proof — see [docs/document-lifecycle/](../document-lifecycle/README.md) |
| `INGESTION_RECOVERY_BATCH_SIZE` | `50` | Optional | Max stale jobs recovered per call/script run |

## Provider/storage configuration cross-reference

- Provider selection detail: [docs/providers/](../providers/README.md)
- Storage provider detail: [docs/storage/](../storage/README.md)
- Which of the above are actually checked by `/health/ready`: [docs/deployment/](../deployment/README.md)

## Test configuration

Tests never read `.env` — integration/E2E fixtures configure their own ephemeral
Postgres/Qdrant/MinIO connection strings dynamically per test session (Testcontainers-assigned
ports), and a guard in `tests/integration/conftest.py`/`tests/e2e/backend/conftest.py` prevents
ever pointing at a production `APP_ENV`/`DATABASE_URL`/`QDRANT_URL`. `tests/unit/configuration/`
verifies `Settings` and `.env.example` stay consistent with each other.

## Current Limitations

- `LOG_LEVEL` is read but not yet wired to an actual logger.
- `REDIS_URL` is checked for readiness but not required — no application code path reads or
  writes Redis yet.

## Deferred Behavior

- Redis actually being used for anything (caching, task queues) — currently connection-only.
- A structured logging pipeline consuming `LOG_LEVEL`.
