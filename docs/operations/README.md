# Operations

Worker execution, lifecycle operations, reconciliation, cleanup, and operational recovery. This is
the canonical reference for **which command to run for which situation** — for the underlying
state machines each command drives, see [docs/document-lifecycle/](../document-lifecycle/README.md).

## Design principle: bounded, explicit, one-job-at-a-time

Every operational script in this repository processes **at most one job per invocation** (except
`process_pending_document_deletions.py`, which drains up to 100 in a bounded loop) and **never
loops, polls, schedules itself, or runs as a daemon**. An operator (or an external scheduler you
control, outside this repository) is responsible for repeated invocation. This is a deliberate,
repository-wide convention — every script listed below follows it.

## Worker execution commands

| Situation | Command | Bound |
|---|---|---|
| Process pending ingestion jobs | *(no script — `IngestionWorker.process_next_job()` is invoked only by tests today; see [Current Limitations](#current-limitations))* | — |
| Recover stale `PROCESSING` ingestion jobs | `make recover-stale-ingestion-jobs` | One batch (`INGESTION_RECOVERY_BATCH_SIZE`, default 50) |
| Process pending document deletions | `make process-pending-document-deletions` | Up to 100 per run, looped internally |
| Build one pending re-index job | `make process-pending-reindex-jobs` | Exactly one job |
| Process/retry one vector-cleanup job | `make process-pending-vector-cleanups` | Exactly one job |

None of these are part of `make verify`/`make test*`/CI — all are manual/optional, run against the
real configured database/Qdrant/storage.

```bash
make recover-stale-ingestion-jobs        # scripts/recover_stale_ingestion_jobs.py
make process-pending-document-deletions  # scripts/process_pending_document_deletions.py
make process-pending-reindex-jobs        # scripts/process_pending_reindex_jobs.py
make process-pending-vector-cleanups     # scripts/process_pending_vector_cleanups.py
```

Each exits `0` when the invocation completed — whether a job was processed or none was
eligible/pending — including when the underlying worker/service recorded a legitimate failure
(e.g. a build failure, or the active-collection safety guard refusing a cleanup). Exit `1` only
when the invocation itself failed unexpectedly (database/Qdrant/storage unreachable); such
failures are logged, never printed as a raw stack trace to stdout.

## Worker signal handling (Phase 2.10)

The three looping/claiming `process_pending_*.py` scripts (not `recover_stale_ingestion_jobs.py`,
which is a single bounded batch with no per-job claim loop to interrupt) install SIGINT/SIGTERM
handlers for the duration of their own run, via `scripts._shutdown.install_stop_signal_handlers()`.
This is a **separate process model from the API's FastAPI lifespan** (see
[docs/architecture/](../architecture/README.md#process-lifecycle-phase-210)) — nothing here starts,
stops, or is stopped by the API process.

- A received signal sets a process-local stop flag; it never raises inside the running work unit.
- The flag is checked **before claiming the next job**, never mid-job —
  `process_pending_document_deletions.py`'s loop (the only one that claims more than once per
  invocation) checks it before each iteration; the two single-job scripts check it once, before
  their only claim.
- A job already claimed (and its worker's own claim → process → commit sequence) always reaches
  its own next commit before the loop re-checks the flag — the current work unit is allowed to
  finish.
- Exit behavior is unchanged by a signal-driven stop: exit code `0`, with the same
  processed-count summary line printed as an unsignaled run (e.g. `Processed 2 document deletion
  job(s).`). A genuine unexpected exception still propagates and exits non-zero, exactly as before
  this change.
- **Force-kill limitation**: if the process is killed outright (not signaled) after a
  `DocumentDeletionJob`/`ReindexJob` is claimed (status → `PROCESSING`) but before its terminal
  commit, that row stays `PROCESSING` indefinitely — see [Current Limitations](#current-limitations).
  `VectorCleanupJob` has no `PROCESSING` status at all (its claim commits with no status mutation),
  so it has no equivalent stuck-state risk.

## Structured logging

`app/core/logging_config.py`'s `configure_logging()` installs a single JSON-formatting stream
handler on the root logger, called once from `app/main.py` at import time. Every log record
carries `timestamp`, `level`, `logger`, `message`, and `correlation_id` (via
`app.core.correlation.get_correlation_id()`, "-" outside any request); any caller-supplied
`extra={...}` fields (`event`, `operation`, `provider`, `document_id`, `job_id`,
`collection_name`, `signal`, `processed`, `error_category`, etc.) ride along verbatim — no fixed
field set is enforced. `LOG_LEVEL` sets the root logger's threshold. No raw exception body,
connection string, or credential is ever logged.

**Standalone scripts never call `configure_logging()`** — only `app/main.py` does. A script's own
`logger.info(...)` calls (e.g. `worker_stop_signal_received`, `worker_stop_before_claim`,
`worker_run_complete`) go through Python's unconfigured default logging, not the JSON formatter —
see [Current Limitations](#current-limitations).

Named event types currently emitted:

| Event | Emitted by | When |
|---|---|---|
| `app_startup_begin` / `app_startup_complete` | `app/core/lifespan.py` | Application startup |
| `app_shutdown_begin` / `app_shutdown_complete` | `app/core/lifespan.py` | Application shutdown |
| `app_error_fallback` | `app/core/exception_handlers.py` | An `AppError` reached the fallback handler |
| `unhandled_exception_fallback` | `app/core/exception_handlers.py` | Any other unhandled exception reached the fallback handler |
| `worker_stop_signal_received` | `scripts/_shutdown.py` | SIGINT/SIGTERM received by a batch script |
| `worker_stop_before_claim` | `scripts/process_pending_*.py` | A stop was requested; the next claim was skipped |
| `worker_run_complete` | `scripts/process_pending_document_deletions.py` | The batch loop finished (signaled or not) |

## Correlation IDs

Header: **`X-Correlation-ID`**. `app/core/middleware.py`'s `correlation_id_middleware` (the
outermost registered middleware — see [docs/architecture/](../architecture/README.md#process-lifecycle-phase-210))
accepts a non-empty incoming header value or generates a UUID4 via
`app.core.correlation.generate_correlation_id()`, stores it in a request-scoped `ContextVar`
(`app/core/correlation.py` — the middleware is the ContextVar's only writer), and echoes it on
every response, including error responses from the fallback exception handlers. Outbound calls
from the Ollama and Qdrant HTTP providers attach the current correlation ID via
`correlation_headers()`; MinIO (SDK-based, not raw `httpx`) does not propagate it.

**Standalone scripts have no correlation context.** They run outside any HTTP request, so
`get_correlation_id()` always returns the fixed placeholder (`"-"`) for their entire process
lifetime — a script's own log lines carry no correlation ID, and there is no mechanism to derive
or inject one for a background script's outbound provider calls.

## Connection pool ownership

The shared SQLAlchemy async engine (`app/db/session.py`) is a module-level singleton, constructed
once at import with `pool_size`/`max_overflow`/`pool_recycle` from `Settings` (see
[docs/configuration/](../configuration/README.md#postgresql-connection-pool-phase-210)) — every
`AsyncSession` the app creates borrows a connection from this one pool. `app/core/lifespan.py`
disposes it on application shutdown via an `AsyncExitStack`. The isolated per-check engine
`platform_health.check_postgres()` creates for `GET /health/ready` is **intentionally separate** —
it is created and disposed within that single check, never shares the app's pool, and this has not
been changed without first demonstrating a concrete defect in that isolation (see
[docs/deployment/](../deployment/README.md#readiness-assumptions)).

## Reconciliation (read-only diagnostics)

Three read-only endpoints surface lifecycle findings — **reconciliation never repairs anything**:

```bash
curl "http://localhost:8000/api/v1/reconciliation/documents/<document_id>/audit"
curl "http://localhost:8000/api/v1/reconciliation/documents/audit?limit=20"
curl "http://localhost:8000/api/v1/reconciliation/collections/<collection_name>/report"
```

See [docs/document-lifecycle/README.md#7-reconciliation-audit-lifecycle](../document-lifecycle/README.md#7-reconciliation-audit-lifecycle)
for classification semantics and full API contracts.

## Mapping a reconciliation finding to a repair action

Repair is provided entirely through the pre-existing, domain-specific, bounded commands above —
**never** a generic "repair this finding" endpoint, and none should ever be added.

| Finding you observed | Repair action |
|---|---|
| Stale or failed ingestion job | `POST /documents/{id}/ingestion/retry` |
| Incomplete/partially-failed deletion | `DELETE /documents/{id}` again (append-only retry) |
| Stale/outdated active index (embedding/chunking version drift) | Schedule (`POST /documents/{id}/reindex`) → `make process-pending-reindex-jobs` → activate (`POST .../reindex/activate`) |
| Leftover vectors in a superseded collection | `make process-pending-vector-cleanups` |

All four are independently operator-reachable today. This mapping is deliberate design, not a
workaround — see [docs/architecture/](../architecture/README.md)'s invariants for why no generic
repair engine exists.

## Observability and diagnostics currently available

- **Operational health** — `GET /health`, `/health/live`, `/health/ready`, `/health/dependencies`
  (unversioned, no `/api/v1` prefix). See [docs/deployment/](../deployment/README.md) for the full
  liveness/readiness contract.
- **Reconciliation reporting** — the three read-only endpoints above are the only structured,
  queryable diagnostic surface; there is no persisted audit history, dashboard, or alerting hook.
- **Sanitized failure messages** — `GET /documents/{id}/failure` and `GET /documents/{id}/deletion`
  return a fixed, safe message; the raw exception stays in Postgres for operator/log inspection
  only (never returned by any API).

## Current Limitations

- **No operational entrypoint processes a normal `PENDING` ingestion job at all.**
  `IngestionWorker.process_next_job()` is invoked only by test fixtures (unit/integration/E2E
  `conftest.py`) — there is no `scripts/process_pending_*_ingestions.py` equivalent to the three
  scripts above. `recover_stale_ingestion_jobs.py` only recovers already-*stale* `PROCESSING` rows
  (converting them to a fresh `PENDING` replacement); it does not process a normal `PENDING` row
  either. In a real deployment, something outside this repository must drive
  `IngestionWorker.process_next_job()` directly today.
- **No automated stale-`PROCESSING` recovery for deletion or re-index jobs** — only ingestion has
  this (`recover_stale_ingestion_jobs.py`). A worker crash, or a force-kill of a
  `process_pending_document_deletions.py`/`process_pending_reindex_jobs.py` process, between a
  job's claim and its terminal commit leaves that row stuck `PROCESSING` indefinitely, with no
  automated or scripted recovery path — see [Worker signal handling](#worker-signal-handling-phase-210).
- **No safe mid-operation cancellation of an active worker work unit.** SIGINT/SIGTERM (Phase
  2.10) only stops a script from claiming its *next* job — a job already claimed always runs to
  its own completion; there is no mechanism to safely interrupt a job partway through processing.
- **No deployed background-worker process for any job type** — every worker is invoked directly by
  a script or test, never by a long-running process or scheduler this repository manages.
- **No persisted reconciliation/audit run history** — every audit call is a live snapshot; nothing
  is stored for trend analysis.
- **No CLI, dashboard, or alerting integration** consumes any of the above yet.
- **Standalone scripts produce unstructured, correlation-less logs** — `configure_logging()` is
  never called by any `scripts/*.py` entrypoint, so their log calls bypass the JSON formatter and
  carry no correlation ID (see [Structured logging](#structured-logging) and
  [Correlation IDs](#correlation-ids)).

## Deferred Behavior

- **A real scheduler deployment** (cron, Kubernetes CronJob, etc.) wiring any of the four
  operational scripts to run periodically — intentionally out of scope; an operator or external
  infrastructure you control invokes them.
- **Stale-deletion/re-index-job recovery mechanisms** — would extend the existing ingestion
  convention, not invent a new one; not built yet.
- **A generic reconciliation "repair" or "apply fix" API** — must never be built; the bounded
  per-domain commands above are the intended, permanent design, not a placeholder for a future
  unified mechanism.
- **Automatic activation, automatic cleanup scheduling, or any background daemon** — every
  mutating lifecycle action remains explicit and operator-triggered.
