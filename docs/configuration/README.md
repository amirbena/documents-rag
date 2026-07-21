# Configuration

Environment variables, defaults, and required-vs-optional configuration. `app/core/config.py`
(`Settings`) is the single source of truth for defaults — this table must be kept in sync with it,
not the other way around. Set via `docker-compose.yml` for containers, or `.env` (copied from
`.env.example`) for local runs outside Docker.

No secrets or real credentials are included below — every credential-shaped value shown is a
local-development-only default (e.g. MinIO's `minioadmin`/`minioadmin`), never reused elsewhere.

## Fail-fast validation (Phase 2.10)

`Settings()` validates everything below **at construction time** — effectively at process import,
since `get_settings()` is called eagerly in `app/main.py`/`app/db/session.py` — rather than
surfacing a bad value later on the first request that happens to exercise it. Covers: well-formed
URLs (scheme + host present) for `DATABASE_URL`/`REDIS_URL`/`QDRANT_URL`/`OLLAMA_BASE_URL`;
supported provider names for `LLM_PROVIDER`/`EMBEDDING_PROVIDER`/`VECTOR_STORE_PROVIDER`/
`RAG_ENGINE`/`FILE_STORAGE_PROVIDER`; positive values for every timeout/pool/retry setting below;
`PROVIDER_RETRY_MAX_DELAY_SECONDS >= PROVIDER_RETRY_BASE_DELAY_SECONDS`; and MinIO cross-field
completeness (all four `MINIO_*` required fields present when `FILE_STORAGE_PROVIDER=minio`).
Error messages name only the offending field — `hide_input_in_errors=True` on `Settings.model_config`
ensures a validation failure on one field never echoes another field's raw value (e.g.
`minio_secret_key`) into the exception's string representation. A well-formed but unreachable URL
still fails later, at the relevant health check or first call — this validation only catches
structurally malformed values (typos, missing scheme), never attempts a connection.

## Core / runtime

| Variable | Default | Required? | Notes |
|---|---|---|---|
| `APP_ENV` | `local` | Optional | Read by application/provider configuration |
| `LOG_LEVEL` | `INFO` | Optional | Wired to `configure_logging()` (Phase 2.10) — see [docs/operations/](../operations/README.md#structured-logging) |
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@postgres:5432/rag_db` | **Required for readiness** | Async SQLAlchemy engine; validated as a well-formed URL at startup, never for reachability |
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

## Provider HTTP timeouts (Phase 2.10)

Every value below replaced a previously hardcoded literal in its provider module — defaults are
unchanged from those literals until overridden. See
[docs/providers/](../providers/README.md#timeout-and-retry-policy-phase-210) for which client each timeout
applies to and how it composes with the retry policy below.

| Variable | Default | Applies to |
|---|---|---|
| `OLLAMA_EMBEDDING_TIMEOUT_SECONDS` | `30.0` | `OllamaEmbeddingProvider`'s `httpx.AsyncClient` |
| `OLLAMA_LLM_TIMEOUT_SECONDS` | `60.0` | `OllamaLLMProvider`'s `httpx.AsyncClient` (streaming generation) |
| `OLLAMA_HEALTH_TIMEOUT_SECONDS` | `5.0` | `OllamaClient`'s health/reachability check |
| `QDRANT_TIMEOUT_SECONDS` | `30.0` | `QdrantVectorStore`'s `httpx.AsyncClient` |
| `MINIO_TIMEOUT_SECONDS` | `30.0` | `MinioFileStorage`'s `urllib3.PoolManager` (connect + read) |

No provider client in this codebase makes an HTTP call without one of these timeouts — there is no
remaining unbounded-wait literal.

## Provider retry policy (Phase 2.10)

Bounded exponential backoff with full jitter, applied inside the Ollama embedding, Qdrant, and
MinIO adapters — see [docs/providers/](../providers/README.md#timeout-and-retry-policy-phase-210) for exactly
which failures are retried, which are permanent, and which call paths are excluded entirely
(streaming LLM generation, MinIO's `response.read()`).

| Variable | Default | Notes |
|---|---|---|
| `PROVIDER_RETRY_MAX_ATTEMPTS` | `3` | Total attempts, including the first — not "3 retries after the first try" |
| `PROVIDER_RETRY_BASE_DELAY_SECONDS` | `0.5` | Backoff base; actual delay is `random.uniform(0, min(max_delay, base * 2**attempt))` |
| `PROVIDER_RETRY_MAX_DELAY_SECONDS` | `5.0` | Hard cap on any single backoff delay; validated `>= PROVIDER_RETRY_BASE_DELAY_SECONDS` |

## PostgreSQL connection pool (Phase 2.10)

| Variable | Default | Wired? |
|---|---|---|
| `DB_POOL_SIZE` | `5` | **Yes** — passed to `create_async_engine` as `pool_size` |
| `DB_MAX_OVERFLOW` | `10` | **Yes** — passed as `max_overflow` |
| `DB_POOL_RECYCLE` | `1800` | **Yes** — passed as `pool_recycle` |
| `DB_POOL_TIMEOUT` | `30.0` | **No** — validated (must be positive) but not currently passed to `create_async_engine`; dead configuration, see [Current Limitations](#current-limitations) |
| `DB_POOL_PRE_PING` | `true` | **No** — same as above |

See [docs/operations/](../operations/README.md#connection-pool-ownership) for engine ownership and
disposal, and `app/db/session.py`'s `_pool_kwargs()` for the dialect gate that skips these kwargs
entirely for a backend whose pool doesn't accept them (defensive only — this codebase never points
`DATABASE_URL` at anything but `postgresql+asyncpg`).

## CORS (Phase 2.10)

| Variable | Default | Notes |
|---|---|---|
| `CORS_ALLOW_ORIGINS` | *(empty)* | Comma-separated allowed origins; empty means no cross-origin request is permitted. A literal `*` is passed through to `CORSMiddleware` as a real wildcard (Starlette itself treats it specially) even though it isn't a documented "shorthand" — see [docs/deployment/](../deployment/README.md#cors) for the full policy (methods, credentials, exposed headers) |

## Startup dependency check (defined, not wired)

| Variable | Default | Wired? |
|---|---|---|
| `STARTUP_DEPENDENCY_CHECK` | `false` | **No** — defined and validated as a boolean, but no code path reads it. The application lifespan (`app/core/lifespan.py`) never probes dependency reachability at startup regardless of this value — see [docs/architecture/](../architecture/README.md#process-lifecycle-phase-210) and [Current Limitations](#current-limitations) |

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

- `DB_POOL_TIMEOUT` and `DB_POOL_PRE_PING` are validated but not consumed — the shared engine
  (`app/db/session.py`) does not pass them to `create_async_engine`.
- `STARTUP_DEPENDENCY_CHECK` is validated but not consumed — no code path reads it; the lifespan
  never probes dependency reachability regardless of its value.
- `REDIS_URL` is checked for readiness but not required — no application code path reads or
  writes Redis yet.
- Provider clients remain per-operation (constructed and closed on each call), never
  application-owned shared clients — only the shared PostgreSQL engine is process-lifetime-scoped.

## Deferred Behavior

- Redis actually being used for anything (caching, task queues) — currently connection-only.
- Wiring `DB_POOL_TIMEOUT`/`DB_POOL_PRE_PING` into the shared engine, or removing them if they
  turn out not to be needed.
- Wiring `STARTUP_DEPENDENCY_CHECK` into the lifespan, or removing it if the deliberate
  no-startup-gate design (see [docs/architecture/](../architecture/README.md#process-lifecycle-phase-210))
  makes it permanently unnecessary.
