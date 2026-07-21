"""Standalone operational script: process (or retry) at most one eligible VectorCleanupJob and
print a bounded summary.

Run via `make process-pending-vector-cleanups` or
`python scripts/process_pending_vector_cleanups.py`. Not part of `make verify`/`make test*`/CI —
mirrors `scripts/process_pending_reindex_jobs.py`'s style exactly: a plain script invoked directly
against the app's real `get_settings()`/DB session/provider-factory machinery, intended for manual
or future-scheduled invocation against a real deployment, never the test suite.

Processes **at most one** job per invocation — an operator or external scheduler is expected to
invoke it repeatedly to make further progress; the script itself never loops, polls, or schedules
itself. This is the explicit, bounded operational entrypoint
`app.services.indexing.cleanup_job_service.process_next_vector_cleanup_job()` was previously
missing (see that module's own docstring: it existed only as an importable function nothing ever
called outside its own tests).

No `--job-id` retry mode is offered: `process_next_vector_cleanup_job()` already claims from both
PENDING *and* FAILED rows (`VectorCleanupStatus.PENDING`/`FAILED`, oldest first) via the existing
service's own claim query, so a single argument-less invocation already covers "process a fresh
cleanup or retry a previously-failed one" — there is no service-contract gap an extra CLI mode
would need to close. Reuses the existing service entirely — never deletes vectors directly, never
calls Qdrant directly, never re-implements the active-serving-collection safety guard
(`retry_cleanup_job()` already refuses to delete from a document's current collection; this script
never bypasses that).

## SIGINT/SIGTERM (Phase 2.10)

`scripts._shutdown.install_stop_signal_handlers()` is checked once, before the script's only claim
— this process's own asyncio loop and FastAPI's lifespan (`app/core/lifespan.py`) are entirely
separate lifecycles. `VectorCleanupJob` has no `PROCESSING` status (see
`app.services.indexing.cleanup_job_service`'s module docstring): the claim commits with no status
mutation, so a force-kill after the claim but before `retry_cleanup_job()` finishes simply leaves
the row PENDING/FAILED exactly as it was — the next invocation picks it up again unchanged. There
is no stuck-PROCESSING risk for this job type to document here.
"""

import asyncio
import logging

from app.core.config import get_settings
from app.db.session import async_session_factory
from app.rag.providers.provider_factory import get_vector_store
from app.services.indexing.cleanup_job_service import (
    VectorCleanupWorkerOutcome,
    process_next_vector_cleanup_job,
)
from scripts._shutdown import install_stop_signal_handlers

logger = logging.getLogger(__name__)


async def main() -> int:
    """Claim and process at most one eligible VectorCleanupJob; print a bounded summary.

    Returns 0 when the invocation completed — whether a job was processed or none was eligible —
    including when the underlying service recorded the attempt as FAILED (an expected,
    already-persisted outcome such as the active-collection safety guard refusing to delete, not a
    script failure — partial/failed cleanup is never hidden or reported as success). Returns 1 only
    if the invocation itself failed unexpectedly (e.g. the database or vector store is
    unreachable); such failures are logged via the existing logging mechanism, never printed as a
    raw stack trace. Also returns 0, without claiming anything, if a stop was requested
    (SIGINT/SIGTERM) before the claim — see this module's docstring.
    """
    settings = get_settings()
    vector_store = get_vector_store(settings)

    with install_stop_signal_handlers() as stop:
        if stop:
            logger.info(
                "Stop requested; not claiming a vector cleanup job.",
                extra={"event": "worker_stop_before_claim", "signal": stop.signal_name},
            )
            print("Stop requested before claiming a job; exiting without processing.")
            return 0

        try:
            async with async_session_factory() as session:
                result = await process_next_vector_cleanup_job(session, vector_store)
        except Exception:
            logger.exception("Unexpected failure while processing a pending vector cleanup job.")
            print("Vector cleanup worker invocation failed unexpectedly. See server logs for details.")
            return 1

    if result.outcome == VectorCleanupWorkerOutcome.NO_JOB:
        print("No pending or retry-eligible vector cleanup job to process.")
        return 0

    print(
        f"job={result.job_id} document={result.document_id} "
        f"collection={result.collection_name} -> {result.outcome.value}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
