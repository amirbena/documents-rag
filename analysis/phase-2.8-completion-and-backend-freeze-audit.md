# Phase 2.8 Completion and Backend Lifecycle Freeze Audit

Analysis-only document. No production code, tests, migrations, or configuration was modified to
produce or update this report — only this file. Every claim below is grounded in direct inspection
of the repository, not in prior task descriptions, commit messages, or PR summaries.

**Status: FINALIZED.** This document was originally written against `main` at commit `bd1c29b`
(PRs #37–#40 merged) and recorded a `FREEZE_BLOCKED_BY_FINITE_GAPS` verdict with three named
finite blockers. Those three blockers have since been closed and merged
([PR #41](https://github.com/amirbena/documents-rag/pull/41), squash commit `e498cb3`). This
revision re-verifies all three directly against the newly merged `main` and updates the verdict to
**`FREEZE_APPROVED`**. The original findings for the five domains that were already COMPLETE
(Observe, Recover, Delete's non-route-layer behavior, Deduplicate, Reconcile) are preserved
unchanged below — they were not re-audited from scratch, per this task's explicit instruction not
to repeat the full audit.

---

## 1. Executive Verdict

**Conclusion: `FREEZE_APPROVED`.**

Phase 2.8 is complete. All five domains that were previously found COMPLETE (Observe, Recover,
Delete, Deduplicate, Reconcile) remain COMPLETE, and the three finite gaps this audit originally
identified as blocking an unqualified freeze have now been closed, merged to `main`, and directly
re-verified:

1. **The `REINDEX_ACTIVE` deletion-route bug is fixed.** `DELETE /documents/{document_id}` now maps
   `DeletionRequestOutcome.REINDEX_ACTIVE` to a deterministic, sanitized HTTP 409 through the same
   outcome-to-error-response table already used for `DOCUMENT_NOT_FOUND`/`INGESTION_ACTIVE` — it no
   longer reaches the `assert result.job is not None` that previously raised an uncaught
   `AssertionError` (HTTP 500). Verified directly in `app/api/v1/routes/documents.py` on merged
   `main`, and covered by 4 new unit tests plus 1 new E2E regression test, all passing.
2. **The re-index build worker now has an operator-facing entrypoint.**
   `scripts/process_pending_reindex_jobs.py` (`make process-pending-reindex-jobs`) claims and builds
   at most one pending `ReindexJob` per invocation via the existing `ReindexWorker`, then exits — no
   loop, no daemon, no scheduler. Verified present on merged `main`; 8 unit tests passing.
3. **The vector-cleanup worker now has an operator-facing entrypoint.**
   `scripts/process_pending_vector_cleanups.py` (`make process-pending-vector-cleanups`) claims and
   processes/retries at most one eligible (PENDING or FAILED) `VectorCleanupJob` per invocation via
   the existing `process_next_vector_cleanup_job()`/`retry_cleanup_job()`, preserving the
   active-serving-collection safety guard entirely inside the service layer. Verified present on
   merged `main`; 7 unit tests passing.

**Remaining blocking gap count: zero.** No migration, new model, new lifecycle status, scheduler,
daemon, or generalized repair framework was introduced to close these gaps — exactly as the
original audit's "Minimum Remaining Work" (§8, historical) anticipated: one route-layer branch and
two thin scripts reusing already-tested code.

---

## 2. Repository State

**Current (post-closure), verified directly:**

```
git switch main; git pull --ff-only origin main
git log -1 --oneline           -> e498cb3 Close Phase 2.8 operational lifecycle gaps (#41)
git rev-parse main             -> e498cb3bab1d0e823131d8f0dbc605ad84ce7fef
git rev-parse origin/main      -> e498cb3bab1d0e823131d8f0dbc605ad84ce7fef
alembic heads                   -> fb63f21089ca (head)
```

`main` now additionally contains the squash-merged
[PR #41 "Close Phase 2.8 operational lifecycle gaps"](https://github.com/amirbena/documents-rag/pull/41)
(3 commits squashed: `8d0b152` "Fix deletion conflict while re-index is active", `3638563` "Add
bounded re-index worker command", `b5690d0` "Add bounded vector-cleanup worker command"), merged via
`gh pr merge --squash --delete-branch=false` with auto-merge disabled and the source branch
preserved. No CI is configured in this repository (no `.github/workflows/`), confirmed again at
merge time via `gh pr checks`, `statusCheckRollup`, and the GitHub check-runs API all reporting
zero checks.

**Original (pre-closure) state, for historical reference:** `main` at `bd1c29b223bf57e5ba0f7321be4a8f400f301953`
("Add reconciliation audit and reporting APIs (#40)"), single Alembic head `fb63f21089ca` — the
same migration head as today; **no migration was added anywhere in this closure work.**

---

## 3. Capability Matrix

| Domain | Sub-capability | Status | Evidence anchor |
|---|---|---|---|
| **1. Observe** | List documents | COMPLETE | `app/services/documents/query_service.py:130-235`; offset pagination, total count, deterministic `created_at DESC, id DESC` |
| | Document detail | COMPLETE | `query_service.py:246-273`; deletion-aware lifecycle derivation |
| | Ingestion state | COMPLETE (latest-only by design) | `query_service.py:152-199`; no history-enumeration endpoint |
| | Sanitized failures | COMPLETE | `query_service.py:76-127`; fixed-constant sanitization, leak-proven at unit/route/E2E |
| | Download | COMPLETE (buffers in memory, not streamed) | `app/services/documents/download_service.py:48-83`; byte-for-byte + Hebrew filename + CRLF-injection tests |
| **2. Recover** | Ingestion retry | COMPLETE | `app/services/ingestion/retry_service.py:62-184`; append-only, DB partial unique index, 409 on deleted doc (E2E-proven) |
| | Stale ingestion recovery | COMPLETE | `app/services/ingestion/stale_recovery_service.py`; no HTTP route, not in `make verify`/`test*`, atomic flip+replace |
| | Attempt/history visibility | PARTIAL (latest-only by design; `NOT_REQUIRED_BY_DEFINED_SCOPE`) | all read paths are `LIMIT 1`; history preserved in DB, never enumerable via API |
| | Stale deletion-job recovery | MISSING (confirmed intentional; matches CLAUDE.md; `NOT_REQUIRED_BY_DEFINED_SCOPE`) | no service/script exists; a crashed-mid-`PROCESSING` `DocumentDeletionJob` is permanently stuck |
| | Failed-deletion retry | PARTIAL (`NOT_REQUIRED_BY_DEFINED_SCOPE`) | only `PARTIALLY_FAILED` is retryable via re-`DELETE`; stuck `PROCESSING` is not |
| **3. Delete** | Deletion request/scheduling | **COMPLETE — route bug fixed** | `deletion_service.py:105-252`; `documents.py`'s `_DELETION_OUTCOME_ERRORS` now maps `REINDEX_ACTIVE` → 409 (was: uncaught `AssertionError` → 500) |
| | Deletion execution (worker) | COMPLETE | `deletion_worker.py:102-128`; vectors-before-storage, partial failure blocks storage, append-only |
| | Post-deletion contract (410/409/soft-delete) | COMPLETE | 410 (not 404) on download after completed deletion, 409 on ingestion retry, `Document` row survives |
| | Vector deletion completeness | COMPLETE | `app/services/indexing/vector_deletion_service.py:109-134`; all 3 sources genuinely enumerated and deduplicated |
| | Result-type honesty | COMPLETE | `VectorDeletionResult.fully_deleted` gates worker progression; proven via unit + real-Qdrant-failure E2E |
| **4. Deduplicate** | Upload-time hash + storage | COMPLETE | SHA-256, `documents.content_hash`, DB-level unique index `uq_documents_content_hash` |
| | Duplicate detection on upload | COMPLETE | `app/services/documents/dedup_service.py:181-223`; `REUSED_ACTIVE`/`REUSED_INDEXED`/`REUSED_FAILED` outcomes |
| | Deleted-document re-upload | COMPLETE | hash released to `NULL` only on `COMPLETED` deletion, same commit |
| | Concurrent-upload race | COMPLETE | commit-time `IntegrityError` disambiguated by Postgres `constraint_name`; real 5-way-parallel Postgres test converges on exactly 1 row |
| | Hash-based reconciliation | MISSING (`NOT_REQUIRED_BY_DEFINED_SCOPE`; never in scope) | zero references to `content_hash` in `app/services/reconciliation/` |
| **5. Upgrade (re-index)** | Scheduling | COMPLETE | `app/services/indexing/reindex_scheduling_service.py`; deterministic outcome ladder, race-safe |
| | Build | **COMPLETE — operational entrypoint added** | `ReindexWorker.process_next_job()` now reachable via `scripts/process_pending_reindex_jobs.py` / `make process-pending-reindex-jobs`; 8 unit tests |
| | Activation | COMPLETE | `app/services/indexing/reindex_activation.py`; document-scoped and job-scoped HTTP endpoints, atomic outcome ladder |
| | Historical-collection cleanup | **COMPLETE — operational entrypoint added** | `process_next_vector_cleanup_job()`/`retry_cleanup_job()` now reachable via `scripts/process_pending_vector_cleanups.py` / `make process-pending-vector-cleanups`; 7 unit tests |
| **6. Reconcile** | Single-document audit | COMPLETE | `app/services/reconciliation/document_audit_service.py`; read-only, dependency failures become WARNING findings, never absence-proof |
| | Batch/keyset audit | COMPLETE | `document_audit_batch_service.py`; bounded limit (1–50), Base64 keyset cursor, aggregate classification counts |
| | Collection reconciliation report | COMPLETE | `collection_reconciliation_report_service.py`; HEALTHY/INCONSISTENT/MISSING/UNMANAGED, deficit-only flagging |
| | Repair / dry-run | **MISSING BY DESIGN — deliberately never added, still true** | zero matches for `dry.run` anywhere in `app`/`tests`/README/ARCHITECTURE; reconciliation remains strictly read-only (see §6) |

---

## 4. Product Questions Matrix

All 12 product questions from the original audit are now directly answerable, with two answers
updated to reflect the closed gaps (Q6, Q8) and one clarified (Q12):

| # | Product question | Answer | Evidence |
|---|---|---|---|
| 1 | What documents exist? | Yes — `GET /documents` with offset pagination, total count, deterministic ordering | §3 Observe/List |
| 2 | Can I inspect a document's details? | Yes — lifecycle status, active collection, timestamps; not re-index status (separate endpoint, deliberate) | `query_service.py:246-273` |
| 3 | Can I see why ingestion failed, safely? | Yes — sanitized to a fixed constant message, proven not to leak internals at 3 test levels | `query_service.py:76-127` |
| 4 | Can I retry a failed ingestion? | Yes — append-only, blocked (409) on an already-deleted document, DB-enforced single-active-job invariant | `retry_service.py` |
| 5 | Can a stuck/stale ingestion be recovered? | Yes, via a manually-run standalone script (`scripts/recover_stale_ingestion_jobs.py`), never automatically | `stale_recovery_service.py` |
| 6 | **Can the document be removed safely?** | **Yes, fully — including while a re-index is active.** The 500-bug on `REINDEX_ACTIVE` is fixed; the route now returns 409, matching the documented contract. | `deletion_service.py`, fixed `documents.py`, `test_document_deletion_routes.py` |
| 7 | Is duplicate content prevented on upload? | Yes — DB-enforced unique hash, race-safe, correctly allows re-upload of content whose prior document was fully deleted | `dedup_service.py` |
| 8 | **Can a document's embeddings be rebuilt without downtime (re-index)?** | **Yes, fully, end-to-end through operator-invocable commands only** — schedule via HTTP, build via `make process-pending-reindex-jobs`, activate via HTTP, historical cleanup via `make process-pending-vector-cleanups`. No ad hoc Python is required anywhere in the cycle anymore. | `reindex_scheduling_service.py`, `reindex_worker.py` + new script, `reindex_activation.py`, `cleanup_job_service.py` + new script |
| 9 | Are there orphaned objects or vectors? | Discoverable via the reconciliation single-document/batch audit and the collection report (deficit detection) | `document_audit_service.py`, `collection_reconciliation_report_service.py` |
| 10 | Can I audit many documents at once, boundedly? | Yes — keyset-paginated batch audit, limit 1–50, aggregate counts, never unbounded | `document_audit_batch_service.py` |
| 11 | Can I see whether a specific Qdrant collection is healthy vs. the DB's expectation? | Yes — HEALTHY/INCONSISTENT/MISSING/UNMANAGED classification, deficit-only | `collection_reconciliation_report_service.py` |
| 12 | **Can a finding actually be repaired, with explicit bounded execution?** | **Yes, for all four domain-specific repair actions** — see §6. Reconciliation itself remains observational only; it was never made mutating and gained no generic repair engine. Repair is provided entirely through pre-existing, independently-invocable domain commands. | See §6 |

---

## 5. Detailed Findings

### 5.1–5.2, 5.4, 5.7 — unchanged from the original audit

The findings for Observe, Recover, Deduplicate, and Reconcile were not re-examined in this
revision — they were already COMPLETE (with documented, deliberate scope exclusions for
ingestion-attempt-history and stale-deletion-recovery) and this closure work did not touch any of
their files. See §3's Capability Matrix for the current status of each; the reasoning is unchanged
from the original audit.

### 5.3 — Delete: `REINDEX_ACTIVE` bug — **RESOLVED**

**Original finding:** `app/api/v1/routes/documents.py`'s `delete_document_route` only special-cased
`DOCUMENT_NOT_FOUND` and `INGESTION_ACTIVE` before `assert result.job is not None`, so
`DeletionRequestOutcome.REINDEX_ACTIVE` (which returns `job=None`) reached the assertion and raised
an uncaught `AssertionError` — FastAPI's default unhandled-exception path, HTTP 500 — instead of the
documented 409.

**Resolution, verified directly on merged `main`:** the three-outcome error mapping now reads:

```python
_DELETION_OUTCOME_ERRORS = {
    DeletionRequestOutcome.DOCUMENT_NOT_FOUND: (status.HTTP_404_NOT_FOUND, "Document not found."),
    DeletionRequestOutcome.INGESTION_ACTIVE: (status.HTTP_409_CONFLICT, "..."),
    DeletionRequestOutcome.REINDEX_ACTIVE: (status.HTTP_409_CONFLICT, "..."),
}
...
if result.outcome in _DELETION_OUTCOME_ERRORS:
    status_code, detail = _DELETION_OUTCOME_ERRORS[result.outcome]
    raise HTTPException(status_code=status_code, detail=detail)

assert result.job is not None
```

`REINDEX_ACTIVE` is now resolved before the assertion, so the assertion is unreachable for that
outcome. `deletion_service.py` (the outcome-producing side) was not modified — it already correctly
returned `REINDEX_ACTIVE`; only the route's incomplete handling was the defect.

**Test evidence (all passing on merged `main`, re-run during this revision):**
- `tests/unit/api/test_document_deletion_routes.py` — 4 new tests: `test_delete_active_reindex_returns_409_not_500`
  (asserts 409, a `detail` string present, no `AssertionError`/`Traceback` substrings — i.e. no leaked
  internal error text), `test_delete_active_reindex_processing_also_returns_409`,
  `test_delete_active_reindex_creates_no_deletion_job` (asserts `session.deletion_jobs == {}` and
  `session.commit_count == 0`), `test_delete_completed_reindex_does_not_block_deletion` (a
  `COMPLETED`, no-longer-active `ReindexJob` correctly does not block). 13/13 passing.
- `tests/e2e/backend/indexing/test_reindex_lifecycle.py` — new
  `test_delete_returns_409_while_reindex_job_is_active`: schedules a real re-index via the HTTP API
  (creating a genuine PENDING `ReindexJob` row), then issues a real `DELETE` request and asserts
  `409`, no `AssertionError` text in the response, and no deletion side effect on the document's
  lifecycle status. Runs against real Postgres + Qdrant. Passing.

### 5.5 — Upgrade: re-index build worker had no operational entrypoint — **RESOLVED**

**Original finding:** `ReindexWorker.process_next_job()` was fully implemented and unit-tested but
never invoked outside its own tests — no script, no Makefile target, no route (its own module
docstring admitted: *"not wired into anything yet"*).

**Resolution, verified directly on merged `main`:** `scripts/process_pending_reindex_jobs.py`
exists, constructs `ReindexWorker(file_storage=create_file_storage(settings))`, and calls
`worker.process_next_job(session, settings)` **exactly once** per invocation — no loop, no polling.
`make process-pending-reindex-jobs` wraps it. The worker module's docstring was corrected to state
the actual wiring instead of the stale "not wired into anything yet" claim.

Exit-code contract: returns 0 for `NO_JOB`/`COMPLETED`/`FAILED`/`SKIPPED_DELETED` (all are outcomes
the worker itself already recorded — legitimate, already-persisted results, not script failures);
returns 1 only if the invocation itself raises unexpectedly, with the exception logged (never
printed as a raw traceback) and a bounded, generic message on stdout.

**Test evidence:** `tests/unit/scripts/test_process_pending_reindex_jobs.py` — 8 tests, all mocking
`ReindexWorker`/`async_session_factory`/`create_file_storage` (no real Postgres/Qdrant/storage
needed): no-job exits 0, completed job exits 0 and reports identity, both `FAILED` and
`SKIPPED_DELETED` outcomes still exit 0 (parametrized), an unexpected exception exits 1 without
leaking the raw exception text (`"internal-host"`/`"6333"`/`"Traceback"`/`"RuntimeError"` all
asserted absent from stdout), `process_next_job()` is called exactly once (no loop), the session
context manager is always entered and exited exactly once (no dependency leak), and the worker is
constructed with the actual `create_file_storage()` result. 8/8 passing.

### 5.6 — Upgrade: vector-cleanup worker had no operational entrypoint — **RESOLVED**

**Original finding:** `process_next_vector_cleanup_job()`/`retry_cleanup_job()` were fully
implemented and unit-tested but had zero references outside `cleanup_job_service.py` itself and its
tests — no API route, no script, no Makefile target.

**Resolution, verified directly on merged `main`:** `scripts/process_pending_vector_cleanups.py`
exists, calls `process_next_vector_cleanup_job(session, vector_store)` **exactly once** per
invocation. `make process-pending-vector-cleanups` wraps it. No `--job-id` explicit-retry mode was
added: the existing claim query already selects both `PENDING` *and* `FAILED` rows, so a single
argument-less invocation already covers "process a fresh cleanup or retry a previously-failed one"
— there was no service-contract gap an extra CLI mode would have needed to close. The
active-serving-collection safety guard remains entirely inside `retry_cleanup_job()`, untouched;
the script never calls Qdrant directly and never re-implements any cleanup-state logic.

Same exit-code contract as the re-index script: 0 for `NO_JOB`/`COMPLETED`/`FAILED` (the FAILED case
— including the safety guard refusing to delete — is printed as `FAILED`, never silently reported as
success), 1 only for an unexpected invocation-level exception, sanitized output.

**Test evidence:** `tests/unit/scripts/test_process_pending_vector_cleanups.py` — 7 tests: no-job
exits 0, completed cleanup exits 0 and reports job/document/collection identity, a `FAILED` outcome
is represented in the output (never hidden as success) and still exits 0, an unexpected exception
exits 1 without leaking raw exception text, the service is called exactly once (no loop), the
session context manager is always exited, and the service is invoked with the actual configured
vector store. 7/7 passing.

---

## 6. Reconciliation Repair Decision

The original audit evaluated whether reconciliation's role — surface findings, repair via
**existing bounded domain actions**, never a new generalized repair framework — was actually
fulfilled end-to-end. Two of four mappings were already fully reachable; two were blocked purely by
a missing operational entrypoint. That gap is now closed:

| Reconciliation finding | Bounded repair action | Reachable by an operator today? |
|---|---|---|
| Stale/failed ingestion job | Ingestion retry (`POST .../ingestion/retry`) | **Yes** — unchanged, already reachable |
| Incomplete/partially-failed deletion | Deletion retry (re-`DELETE`) | **Yes** — unchanged, already reachable |
| Stale/outdated active index (embedding/chunking version drift) | Re-index build (`make process-pending-reindex-jobs`) + activation (`POST .../reindex/activate`) | **Yes, now** — build entrypoint added this closure |
| Leftover vectors in a superseded collection | Vector cleanup processing/retry (`make process-pending-vector-cleanups`) | **Yes, now** — entrypoint added this closure |

**All four domain-specific bounded repair actions are now independently operator-reachable.**
Reconciliation itself was not modified in any way by this closure work — it remains **strictly
read-only and observational by design**: no mutation capability was added to
`document_audit_service.py`, `document_audit_batch_service.py`, or
`collection_reconciliation_report_service.py`; no dry-run flag exists anywhere in the codebase (still
zero matches for `dry.run`); no reconciliation route gained a write path; and no generic "apply fix"
or "repair this finding" endpoint was introduced. Repair is, and remains, entirely a matter of an
operator (or an operator's own external scheduler) invoking the correct pre-existing,
domain-specific command for the finding they observed — reconciliation's job is only ever to make
that finding visible.

---

## 7. Freeze Decision

**`FREEZE_APPROVED`**

Rationale: at the time of the original audit, five of six domains were already COMPLETE and the
sixth (Reconcile) did exactly what it claimed, but a confirmed 500-error bug on a mutating endpoint
and two operationally-unreachable "bounded repair" actions blocked an unqualified freeze. Both
categories of defect were finite, mechanically scoped, and have now been directly re-verified as
closed on merged `main` — including running the actual test suites (unit, integration with real
Postgres/Qdrant, and E2E with real Postgres/Qdrant) rather than trusting the merge alone. No new
lifecycle status, migration, scheduler, daemon, or generalized repair framework was introduced to
close them, so this closure work did not reopen any of the freeze's own guardrails against
speculative architecture.

The remaining PARTIAL/MISSING items in the Capability Matrix (ingestion-attempt-history enumeration,
stale-deletion-job recovery, hash-based reconciliation) are all classified
`NOT_REQUIRED_BY_DEFINED_SCOPE` — they are pre-existing, already-documented, deliberate exclusions
(see `CLAUDE.md` §6, §8) that this audit explicitly does not treat as freeze blockers.

---

## 8. Minimum Remaining Work

**None.** No remaining Phase 2.8 lifecycle work is required. The three items the original audit's
§8 identified — the route-layer fix and the two thin operational scripts — have all been
implemented, tested, merged to `main`, and re-verified directly against the merged state in this
revision.

---

## 9. What Must Not Be Built

Unchanged from the original audit — still binding, and this closure work did not violate any of it:

- No generalized "repair" or "apply fix" API endpoint for reconciliation findings.
- No dry-run mode/flag for any operation (confirmed still absent everywhere).
- No new lifecycle statuses on any model (confirmed: zero model/schema/migration changes in the
  closure PR).
- No stale-deletion-job recovery service (still an explicit, documented non-goal).
- No scheduler, cron, or background daemon process for any worker — both new scripts execute
  directly, exactly once, exactly like the two pre-existing operational scripts they mirror; an
  operator or external scheduler outside this repository is responsible for repeated invocation.
- No new abstraction layer (`repositories/`, `domain/`, `application/`).
- No attempt-history-listing endpoint for ingestion.
- No content-hash-based storage-drift reconciliation feature.
- No Phase 2.9 planning, broad refactor, or redesign of any of the five previously-COMPLETE domains.

**Going forward:** no additional speculative backend lifecycle phases, micro-state-machines, or
recovery mechanisms should be added before UI validation. Backend lifecycle work may resume only
when driven by one of: (1) a bug reproduced through the UI, (2) a browser End-to-End failure, (3) an
actual broken integration flow, (4) an observed operational issue, or (5) a concrete production
requirement — never by further speculative completeness audits of already-COMPLETE domains.

---

## 10. Backend Lifecycle Freeze

```text
FREEZE_APPROVED

Phase 2.8 is complete.

No additional speculative backend lifecycle phases, micro-state-machines, or recovery
mechanisms should be added before UI validation.

Backend lifecycle work may resume only when driven by:
1. a bug reproduced through the UI;
2. a browser End-to-End failure;
3. an actual broken integration flow;
4. an observed operational issue;
5. a concrete production requirement.
```

This freeze covers all six domains audited in this document (Observe, Recover, Delete, Deduplicate,
Upgrade, Reconcile) as they exist on `main` at commit `e498cb3bab1d0e823131d8f0dbc605ad84ce7fef`.
The four domain-specific bounded repair actions documented in §6 remain the only sanctioned
operator-facing repair mechanisms; reconciliation remains read-only by design; no generic repair
engine exists or should be built.

## Final Recommendation

**Phase 2.8 is complete. Freeze backend lifecycle architecture and move to UI and browser
End-to-End validation.**
