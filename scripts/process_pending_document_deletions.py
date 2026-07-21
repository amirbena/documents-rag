"""Standalone operational script: process pending DocumentDeletionJob rows and print a summary.

Run via `make process-pending-document-deletions` or
`python scripts/process_pending_document_deletions.py`. Not part of `make verify`/`make test*`/CI
— mirrors `scripts/recover_stale_ingestion_jobs.py`'s style exactly: a plain script invoked
directly against the app's real `get_settings()`/DB session/provider-factory machinery, intended
for manual or future-scheduled invocation against a real deployment, never the test suite.
Processes jobs one at a time in a loop until no PENDING job remains (bounded by `batch_size`),
using `DocumentDeletionWorker` — the same execution logic Backend E2E tests drive directly.

## SIGINT/SIGTERM (Phase 2.10)

`scripts._shutdown.install_stop_signal_handlers()` is checked before each claim, never mid-job —
`DocumentDeletionWorker.process_next_job()`'s own claim/process/commit sequence for one job is
never interrupted once started, so a signal received while a job is in flight only takes effect
before the *next* claim. This process's own asyncio loop and FastAPI's lifespan
(`app/core/lifespan.py`) are entirely separate lifecycles — nothing here is wired to the API
process.

If the process is force-killed (not merely signaled) after a job is claimed (status ->
PROCESSING) but before its terminal commit, that `DocumentDeletionJob` row remains PROCESSING
indefinitely — there is currently no stale-recovery mechanism for `DocumentDeletionJob` in this
codebase (see CLAUDE.md's High-Risk Invariants and `app/services/documents/deletion_worker.py`'s
own module docstring). This script does not change that; it only makes a *requested*
(signal-driven) stop predictable, not a forced kill safe.
"""

import asyncio
import logging

from app.core.config import get_settings
from app.db.session import async_session_factory
from app.rag.providers.provider_factory import get_vector_store
from app.services.documents.deletion_worker import DocumentDeletionWorker
from app.storage.factory import create_file_storage
from scripts._shutdown import install_stop_signal_handlers

logger = logging.getLogger(__name__)

_MAX_JOBS_PER_RUN = 100


async def main() -> int:
    """Process up to `_MAX_JOBS_PER_RUN` pending document deletions and print a summary.

    Stops cooperatively on SIGINT/SIGTERM: checked before each claim, never mid-job. See this
    module's docstring for what a force-kill (rather than a signal) leaves behind.
    """
    settings = get_settings()
    worker = DocumentDeletionWorker(
        vector_store=get_vector_store(settings), file_storage=create_file_storage()
    )

    processed = 0
    with install_stop_signal_handlers() as stop:
        for _ in range(_MAX_JOBS_PER_RUN):
            if stop:
                logger.info(
                    "Stop requested; not claiming another document deletion job.",
                    extra={"event": "worker_stop_before_claim", "signal": stop.signal_name},
                )
                break

            async with async_session_factory() as session:
                job = await worker.process_next_job(session)
            if job is None:
                break
            processed += 1
            print(f"  job={job.id} document={job.document_id} -> {job.status}")

    logger.info(
        "Document deletion batch run finished.",
        extra={
            "event": "worker_run_complete",
            "processed": processed,
            "stopped_by_signal": stop.signal_name,
        },
    )
    print(f"Processed {processed} document deletion job(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
