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

| Dependency | Required for the app to start? | Required for readiness? |
|---|---|---|
| PostgreSQL | Yes (migrations must be applied first) | Yes |
| Qdrant | Yes | Yes |
| Ollama (+ both configured models pulled) | Yes | Yes |
| Redis | No — app starts without it | No — checked but not required |
| MinIO | Only if `FILE_STORAGE_PROVIDER=minio` | Only if configured |

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

## Current deployment limitations

- **No production deployment target exists** — no Kubernetes manifests, Helm charts, or cloud
  infrastructure-as-code.
- **No deployed background-worker process** — every worker (ingestion, deletion, re-index,
  cleanup) is invoked by a script or test, never a long-running managed process. See
  [docs/operations/](../operations/README.md).
- **No auth, rate limiting, or observability/logging pipeline.**
- **Alembic migrations are a manual step** — not automated as part of container startup.
- **No CI workflow exists in this repository** (`.github/workflows/` is absent) — verification is
  run manually / via the local pre-commit hook only.

## Deferred Behavior

- Kubernetes manifests, Helm charts, ArgoCD Application/Rollout resources — this repository only
  establishes the operational health *contract* those future consumers would use.
- A real scheduler deployment (cron, Kubernetes CronJob) for any of the operational scripts in
  [docs/operations/](../operations/README.md).
- Auth, rate limiting, and a structured observability/logging pipeline.
- Automated Alembic migration application as part of container startup.
