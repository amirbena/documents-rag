# Troubleshooting

Common failures and verified recovery steps.

## `app` fails to start / connection refused to postgres|redis|qdrant|ollama

Those services take a few seconds to become ready. `docker-compose.yml` uses `depends_on` (start
order only, not a readiness check) — if the app crashes on startup, confirm the dependency logs
show it's ready, then retry:

```bash
docker compose up --build app
```

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
