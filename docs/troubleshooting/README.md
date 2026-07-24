# Troubleshooting

Common failures and verified recovery steps.

## `app` fails to start — invalid configuration (Phase 2.10)

`Settings()` validates fail-fast at process import (see
[docs/configuration/](../configuration/README.md#fail-fast-validation-phase-210)) — a malformed
`DATABASE_URL`, an unrecognized `LLM_PROVIDER`, an incomplete MinIO config, or a non-positive
timeout/pool/retry value raises immediately, before the app finishes importing, with a message
naming only the offending field (never a secret value). This is **not** the same failure as
"dependency unreachable" below — a startup crash with a Pydantic `ValidationError` in the logs
means fix `.env`/the environment; a running process that later returns `503` from
`/health/ready` means a dependency, not configuration, is the problem.

## `POST /documents`/other requests fail while a dependency is still coming up

`docker-compose.yml` uses `depends_on` (start order only, not a readiness check), and (Phase
2.10) the `app` process itself never gates startup on Postgres/Qdrant/Ollama/Redis/MinIO
reachability — it starts and stays up even while a dependency is still coming up. Check
`GET /health/ready` first; it reports `503` with the specific failing check(s) until every
required dependency is reachable:

```bash
curl http://localhost:8000/health/ready
docker compose logs <service> --tail 50   # confirm the specific dependency is actually ready
```

If the `app` process itself crashed or won't start at all (not merely a `503` from
`/health/ready`), that's a configuration or resource-initialization failure, not a dependency
readiness issue — see "invalid configuration" above.

## Port already in use

Another local process is bound to `8000`, `5432`, `6379`, `6333`, or `11434`. Stop it, or change
the host-side port mapping in `docker-compose.yml` (`"HOST:CONTAINER"`).

## Checking service logs

```bash
docker compose logs <service> --tail 50
```

## Verifying internal networking

From inside the `app` container:

```bash
docker compose exec app python -c "import socket; socket.create_connection(('postgres', 5432), timeout=5)"
docker compose exec app python -c "import urllib.request; urllib.request.urlopen('http://ollama:11434', timeout=5)"
```

## Database connectivity

`GET /health/ready` and `GET /health/dependencies` report Postgres reachability directly
(`SELECT 1`) — check those first before assuming a connectivity problem:

```bash
curl http://localhost:8000/health/dependencies
```

If Postgres itself won't start, check `docker compose logs postgres --tail 50` for a corrupted
volume; a full reset (drops data) is the last resort — see below.

## Migration problems / multiple Alembic heads

```bash
alembic heads
```

**Must show exactly one head.** More than one means two migration branches were created without
merging — resolve with an explicit merge migration
(`alembic merge -m "merge heads" <rev1> <rev2>`), never by deleting a migration file. See
[docs/deployment/](../deployment/README.md#migration-sequencing) for when migrations must run, and
[alembic/README.md](../../alembic/README.md) for authoring mechanics.

If `docker compose up` was run without ever applying migrations, `POST /documents` and similar
endpoints will fail against an unmigrated schema — run
`docker compose exec app alembic upgrade head`.

## Vector-store (Qdrant) connectivity

`GET /health/dependencies` checks Qdrant via a lightweight `GET /collections` call. If it fails:

```bash
docker compose logs qdrant --tail 50
curl http://localhost:6333/collections   # from the host, if the port is published
```

An `IncompatibleIndexConfigurationError` (not a connectivity issue) means an existing collection's
vector dimension doesn't match the active `EmbeddingIndexConfig` — see
[docs/storage/](../storage/README.md); this is never auto-resolved, an operator must resolve it
deliberately.

## Object-storage connectivity

For `local` (default): check `LOCAL_STORAGE_ROOT` is writable inside the container. For `minio`:
`GET /health/dependencies` reports the `file_storage` check (bucket reachability/creation); check
`docker compose logs minio --tail 50` and confirm `MINIO_ENDPOINT`/`MINIO_ACCESS_KEY`/
`MINIO_SECRET_KEY`/`MINIO_BUCKET` are set correctly (see
[docs/configuration/](../configuration/README.md)).

## Provider configuration

`GET /api/v1/providers/ollama/health` returns `200` when Ollama is reachable and both configured
models are pulled, or `503` (same body, showing which check failed) otherwise:

```bash
curl http://localhost:8000/api/v1/providers/ollama/health
docker compose exec ollama ollama pull llama3.1
docker compose exec ollama ollama pull bge-m3
```

An unrecognized `LLM_PROVIDER`/`EMBEDDING_PROVIDER`/`VECTOR_STORE_PROVIDER`/`FILE_STORAGE_PROVIDER`
value raises a clear, named exception at resolution time (`UnsupportedProviderError`/
`StorageConfigurationError`) — check application logs for the exact offending value; there is no
silent fallback to a default provider.

## Provider retry exhaustion

A provider call that fails repeatedly with a transient error (connection failure, timeout, or
429/502/503/504) retries up to `PROVIDER_RETRY_MAX_ATTEMPTS` times with bounded backoff, then
raises that provider's own existing error type unchanged (`OllamaEmbeddingError`/
`QdrantVectorStoreError`/a `StorageError` subclass) — never a generic "retries exhausted" wrapper.
A **permanent** error (4xx other than 429, or a malformed response) is never retried at all — it
fails on the first attempt. See
[docs/providers/](../providers/README.md#timeout-and-retry-policy-phase-210) for the exact
transient/permanent classification and the two call paths (streaming LLM generation, MinIO
`response.read()`) that are never retried by design. If a call fails faster or slower than
expected, check `PROVIDER_RETRY_MAX_ATTEMPTS`/`PROVIDER_RETRY_BASE_DELAY_SECONDS`/
`PROVIDER_RETRY_MAX_DELAY_SECONDS` before assuming a bug.

## Missing CORS response headers

If a browser-based frontend reports a blocked cross-origin request (no
`Access-Control-Allow-Origin` in the response), the calling origin is almost certainly missing
from `CORS_ALLOW_ORIGINS` — the server still answers the request normally (CORS enforcement is
browser-side), it just omits the header. Check:

```bash
echo $CORS_ALLOW_ORIGINS   # comma-separated; must exactly match the frontend's Origin header
curl -i -H "Origin: http://localhost:3000" http://localhost:8000/health
```

See [docs/deployment/](../deployment/README.md#cors) for the full fixed policy (methods,
credentials, exposed headers) — only `allow_origins` is configurable.

## PostgreSQL pool exhaustion / connection issues

The shared engine's pool size is `DB_POOL_SIZE` + `DB_MAX_OVERFLOW` connections
(`app/db/session.py`) — see [docs/operations/](../operations/README.md#connection-pool-ownership).
A pool that's genuinely exhausted under real concurrent load raises a standard SQLAlchemy
`TimeoutError` from the checkout call; there is no custom pool-exhaustion handling. Note
`DB_POOL_TIMEOUT`/`DB_POOL_PRE_PING` are validated but currently **not** wired into the engine
(see [docs/configuration/](../configuration/README.md#postgresql-connection-pool-phase-210)) — do
not expect changing them to have any effect yet.

## Stale local Alembic history (pre-Phase-2.10 database)

`Can't locate revision identified by '<old-id>'` means your local database predates the Phase 2.10
migration history reset (see
[alembic/README.md](../../alembic/README.md#migration-history-reset-phase-210)) — it's stamped at
one of the 9 deleted incremental revisions. Recreate it; do not attempt to bridge or
`alembic stamp head` your way past this:

```bash
dropdb rag_db && createdb rag_db
alembic upgrade head
```

See [docs/development/](../development/README.md#recreating-your-local-database-after-the-alembic-history-reset-phase-210)
for the Docker Compose equivalent.

## Testcontainers failures (integration / backend E2E)

- **Docker not running** — Testcontainers needs a working Docker daemon; `docker ps` should
  succeed before running `make test-integration`/`make test-e2e-backend`.
- **A test hangs at container startup** — check for stale containers/networks from a previous
  interrupted run (`docker ps -a`, `docker network ls`); remove any leftover Testcontainers-labeled
  resources.
- See [docs/backend-e2e/](../backend-e2e/README.md#failure-diagnosis) for backend-E2E-specific
  diagnosis.

## Stale local containers / full reset

Drops Postgres/Qdrant/Ollama volumes — **deletes local data**:

```bash
docker compose down -v
```

Rebuild after dependency changes (Python deps are installed at image build time, not container
start):

```bash
docker compose up --build app
```

## Lifecycle jobs stuck in a non-terminal state

| Stuck state | Operational recovery available? | Command |
|---|---|---|
| Ingestion job stuck `PROCESSING` (stale) | **Yes** | `make recover-stale-ingestion-jobs` |
| Deletion job stuck `PROCESSING` | **No** — no recovery mechanism exists (documented limitation) | — |
| Re-index job stuck `PROCESSING` | **No** — no recovery mechanism exists (documented limitation) | — |
| Deletion job `PARTIALLY_FAILED` | Yes — retryable | `DELETE /documents/{id}` again |
| Vector-cleanup job `PENDING`/`FAILED` | Yes | `make process-pending-vector-cleanups` |

See [docs/document-lifecycle/](../document-lifecycle/README.md) for the full state machines and
[docs/operations/](../operations/README.md) for every recovery command. Do not treat the "No"
rows above as a bug to fix here — they are documented, deliberate scope boundaries.
