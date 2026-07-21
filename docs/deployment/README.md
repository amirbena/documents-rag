# Deployment

Supported deployment model, container topology, migration sequencing, and the operational health
contract. See [docs/configuration/](../configuration/README.md) for the full environment variable
reference.

## Supported deployment model

**Local-first, Docker Compose only.** There is no production deployment target, Kubernetes
manifest, or cloud infrastructure configuration in this repository — Docker Compose is both the
development and the only currently-supported runtime topology.

## Container topology

```
host:8000 ──► app ──► postgres:5432
                 ├──► redis:6379
                 ├──► qdrant:6333
                 ├──► ollama:11434
                 └──► minio:9000 (only when FILE_STORAGE_PROVIDER=minio)
```

All services join the default Compose network (`documents-rag_default`) and address each other by
service name via Docker's embedded DNS. Only the ports needed for host-side debugging are
published (`8000`, `5432`, `6379`, `6333`, `11434`, `9000`, `9001`) — a real production deployment
would typically expose only `app`'s port.

`minio` is a local/dev-only service — the app does not require it while
`FILE_STORAGE_PROVIDER=local` (the default). Start it explicitly with
`docker compose up -d minio`. Credentials in `docker-compose.yml`/`.env.example`
(`minioadmin`/`minioadmin`) are for local development only.

## Startup sequence

```bash
cp .env.example .env
docker compose up --build
docker compose exec app alembic upgrade head   # see "Migration sequencing" below
docker compose exec ollama ollama pull llama3.1
docker compose exec ollama ollama pull bge-m3
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/providers/ollama/health
```

## Migration sequencing

**Docker Compose starts Postgres but does not apply Alembic migrations automatically.**
`docker compose up` brings up a fresh, unmigrated Postgres — the schema must be created explicitly
before the app can persist anything.

```bash
docker compose exec app alembic upgrade head        # containerized
alembic upgrade head                                 # local venv, Postgres reachable directly
```

Run this after the `app`/`postgres` containers are up and **before** testing document upload.
Verify exactly one head:

```bash
alembic heads   # expected: a single "<revision> (head)" line — never more than one
```

See [alembic/README.md](../../alembic/README.md) for migration-authoring mechanics (autogenerate,
`env.py` wiring) — this page covers only sequencing at deployment time.

## Infrastructure dependencies

**The process itself starts regardless of any remote dependency's reachability (Phase 2.10)** —
`app/core/lifespan.py` never probes PostgreSQL/Qdrant/Ollama/Redis/MinIO at startup; the only
things that can prevent the process from starting are invalid configuration (fail-fast `Settings`
validation, see [docs/configuration/](../configuration/README.md#fail-fast-validation-phase-210))
and failure to initialize an application-owned resource (the shared SQLAlchemy engine object
itself — never a live connection). `GET /health/ready` below is what actually gates whether an
instance is fit to receive traffic, and it can report `503` at any point after the process has
already started and is otherwise running fine.

| Dependency | Required for the process to start? | Required for `/health/ready`? |
|---|---|---|
| PostgreSQL | No (migrations must still be applied before upload/etc. work) | Yes |
| Qdrant | No | Yes |
| Ollama (+ both configured models pulled) | No | Yes |
| Redis | No | No — checked but not required |
| MinIO | No | Only if `FILE_STORAGE_PROVIDER=minio` |

## Readiness assumptions

Four **unversioned** endpoints (`app/api/routes/health.py`, no `/api/v1` prefix — deliberately
independent of business API versioning):

| Endpoint | Purpose | Calls dependencies? | Status codes |
|---|---|---|---|
| `GET /health` | Static process-up summary | No | Always `200` |
| `GET /health/live` | Liveness — never touches a downstream dependency | No | Always `200` while alive |
| `GET /health/ready` | Readiness — can this instance serve traffic | Yes | `200` if every *required* check passes, else `503` |
| `GET /health/dependencies` | Full per-dependency diagnostic detail | Yes | Always `200` (status is in the body) |

Required-for-readiness checks: `postgres`, `qdrant`, `ollama`, `ollama_chat_model`,
`ollama_embedding_model`. `redis` and `file_storage` are checked but only `file_storage` is
required (Redis is unused by any code path today). Every check is timeout-bounded (3s), runs
concurrently, never mutates or restarts the dependency it probes, and every response is a fixed,
generic message — never a raw exception, connection string, or credential.

**Future consumers this contract is designed for (none wired up in this repository):** Kubernetes
liveness/readiness probes, load balancer health checks, ArgoCD rollout gates, monitoring/alerting
polling `/health/dependencies`.

## Graceful shutdown and process termination

On SIGTERM/process exit, FastAPI's ASGI lifespan runs `app/core/lifespan.py`'s shutdown path: it
disposes the shared SQLAlchemy engine (closing pooled connections) via an `AsyncExitStack`, with
deterministic `app_shutdown_begin`/`app_shutdown_complete` structured log lines around it. No
provider client is closed here — every Ollama/Qdrant/MinIO client is already created and closed
per operation, so there is nothing else process-lifetime-scoped to release. This governs the API
process only.

The standalone `scripts/process_pending_*.py` batch scripts are a **separate process model** with
their own SIGINT/SIGTERM handling — see
[docs/operations/](../operations/README.md#worker-signal-handling-phase-210) for stop-before-next-claim
semantics, exit codes, and the force-kill limitation for an already-claimed deletion/re-index job.

## CORS

`CORSMiddleware` (`app/core/cors.py`, registered in `app/main.py`) is always installed; its effect
is governed entirely by `CORS_ALLOW_ORIGINS` (empty by default — no cross-origin request is
permitted until a frontend origin is named). Fixed policy, not configuration surface:

| Aspect | Value |
|---|---|
| Allowed origins | `CORS_ALLOW_ORIGINS` (comma-separated); empty = none allowed |
| Allowed methods | `GET`, `POST`, `DELETE` — exactly the verbs this API's routes use |
| Allowed headers | Starlette's safelisted set (`Accept`, `Accept-Language`, `Content-Language`, `Content-Type`) — covers JSON and multipart bodies; no custom header is allow-listed |
| Credentials | **Disabled** (`allow_credentials=False`) — this backend has no cookie/session/bearer-token authentication, so there is no credentialed-CORS use case to support |
| Exposed response headers | `X-Correlation-ID` — so browser-based frontend JS can read it (otherwise browsers hide non-safelisted response headers from cross-origin scripts) |

Registered before `correlation_id_middleware` in `app/main.py`, so correlation ID stays the
outermost middleware layer — see
[docs/architecture/](../architecture/README.md#process-lifecycle-phase-210). A wildcard
`CORS_ALLOW_ORIGINS=*`, if ever set, is honored as a real wildcard (Starlette reflects it as a
literal `Access-Control-Allow-Origin: *`) and stays spec-safe specifically because credentials are
disabled — the CORS spec forbids combining a wildcard origin with credentialed requests.

## Current deployment limitations

- **No production deployment target exists** — no Kubernetes manifests, Helm charts, or cloud
  infrastructure-as-code.
- **No deployed background-worker process** — every worker (ingestion, deletion, re-index,
  cleanup) is invoked by a script or test, never a long-running managed process. See
  [docs/operations/](../operations/README.md).
- **No auth or rate limiting.** Structured JSON logging and correlation IDs exist (Phase 2.10 —
  see [docs/operations/](../operations/README.md#structured-logging)), but no metrics/tracing
  platform.
- **Alembic migrations are a manual step** — not automated as part of container startup. See
  [docs/development/](../development/README.md#recreating-your-local-database-after-the-alembic-history-reset-phase-210)
  if your local database predates the Phase 2.10 baseline reset.
- **No automated stale-`PROCESSING` recovery for deletion/re-index jobs**, and **no safe
  mid-operation cancellation** of a worker's active work unit — see
  [docs/operations/](../operations/README.md#current-limitations).
- **No CI workflow exists in this repository** (`.github/workflows/` is absent) — verification is
  run manually / via the local pre-commit hook only.

## Deferred Behavior

- Kubernetes manifests, Helm charts, ArgoCD Application/Rollout resources — this repository only
  establishes the operational health *contract* those future consumers would use.
- A real scheduler deployment (cron, Kubernetes CronJob) for any of the operational scripts in
  [docs/operations/](../operations/README.md).
- Auth, rate limiting, and a structured observability/logging pipeline.
- Automated Alembic migration application as part of container startup.
