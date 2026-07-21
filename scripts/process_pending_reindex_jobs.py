"""Standalone operational script: process at most one pending ReindexJob's build and print a
bounded summary.

Run via `make process-pending-reindex-jobs` or `python scripts/process_pending_reindex_jobs.py`.
Not part of `make verify`/`make test*`/CI — mirrors `scripts/recover_stale_ingestion_jobs.py`'s and
`scripts/process_pending_document_deletions.py`'s style: a plain script invoked directly against the
app's real `get_settings()`/DB session/storage-factory machinery, intended for manual or
future-scheduled invocation against a real deployment, never the test suite.

Unlike `process_pending_document_deletions.py` (which drains up to 100 jobs in one run), this script
processes **at most one** pending job per invocation — an operator or external scheduler is expected
to invoke it repeatedly to make further progress; the script itself never loops, polls, or schedules
itself. This is the explicit, bounded operational entrypoint `ReindexWorker.process_next_job()` was
previously missing (see `app.services.indexing.reindex_worker`'s module docstring: "not wired into
anything yet").

Reuses the existing `ReindexWorker` entirely — never re-implements claim/build/commit logic, never
queries or mutates `ReindexJob` state directly. Build-only: never activates the built target (see
`scripts/process_pending_document_deletions.py`'s analogue in the deletion domain, and
`app.services.indexing.reindex_activation` for the separate activation step) and never touches
historical-collection cleanup (see `process_pending_vector_cleanups.py`).

## SIGINT/SIGTERM (Phase 2.10)

`scripts._shutdown.install_stop_signal_handlers()` is checked once, before the script's only claim
— this process's own asyncio loop and FastAPI's lifespan (`app/core/lifespan.py`) are entirely
separate lifecycles. If the process is force-killed (not merely signaled) after a job is claimed
(status -> PROCESSING) but before its terminal commit, that `ReindexJob` row remains PROCESSING
indefinitely — `app.services.indexing.reindex_worker`'s own module docstring already documents
"No stale-PROCESSING recovery yet" as an existing, unresolved gap mirroring `IngestionJob`'s/
`DocumentDeletionJob`'s. This script does not change that; it only makes a *requested*
(signal-driven) stop predictable, not a forced kill safe.
"""

import asyncio
import logging

from app.core.config import get_settings
from app.db.session import async_session_factory
from app.services.indexing.reindex_worker import ReindexWorker, ReindexWorkerOutcome
from app.storage.factory import create_file_storage
from scripts._shutdown import install_stop_signal_handlers

logger = logging.getLogger(__name__)


async def main() -> int:
    """Claim and build at most one pending ReindexJob; print a bounded summary.

    Returns 0 when the invocation completed — whether a job was processed or none was pending —
    including when the worker itself recorded a job as FAILED/SKIPPED_DELETED (an expected,
    already-persisted outcome, not a script failure). Returns 1 only if the invocation itself
    failed unexpectedly (e.g. the database or storage backend is unreachable); such failures are
    logged via the existing logging mechanism, never printed as a raw stack trace. Also returns 0,
    without claiming anything, if a stop was requested (SIGINT/SIGTERM) before the claim — see
    this module's docstring.
    """
    settings = get_settings()
    worker = ReindexWorker(file_storage=create_file_storage(settings))

    with install_stop_signal_handlers() as stop:
        if stop:
            logger.info(
                "Stop requested; not claiming a re-index job.",
                extra={"event": "worker_stop_before_claim", "signal": stop.signal_name},
            )
            print("Stop requested before claiming a job; exiting without processing.")
            return 0

        try:
            async with async_session_factory() as session:
                result = await worker.process_next_job(session, settings)
        except Exception:
            logger.exception("Unexpected failure while processing a pending re-index job.")
            print("Re-index worker invocation failed unexpectedly. See server logs for details.")
            return 1

    if result.outcome == ReindexWorkerOutcome.NO_JOB:
        print("No pending re-index job to process.")
        return 0

    print(f"job={result.job_id} document={result.document_id} -> {result.outcome.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
