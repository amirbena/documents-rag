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

## Reconciliation (read-only diagnostics)

Three read-only endpoints surface lifecycle findings — **reconciliation never repairs anything**:

```bash
curl "http://localhost:8000/api/v1/reconciliation/documents/<document_id>/audit"
curl "http://localhost:8000/api/v1/reconciliation/documents/audit?limit=20"
curl "http://localhost:8000/api/v1/reconciliation/collections/<collection_name>/report"
```

See [docs/document-lifecycle/README.md#7-reconciliation--audit-lifecycle](../document-lifecycle/README.md#7-reconciliation--audit-lifecycle)
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
- **No stale-`PROCESSING` recovery for deletion or re-index jobs** — only ingestion has this. A
  worker crash mid-`PROCESSING` for deletion/re-index leaves that row stuck indefinitely with no
  automated or scripted recovery path.
- **No deployed background-worker process for any job type** — every worker is invoked directly by
  a script or test, never by a long-running process or scheduler this repository manages.
- **No persisted reconciliation/audit run history** — every audit call is a live snapshot; nothing
  is stored for trend analysis.
- **No CLI, dashboard, or alerting integration** consumes any of the above yet.

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
