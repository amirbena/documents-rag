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

logger = logging.getLogger(__name__)


async def main() -> int:
    """Claim and process at most one eligible VectorCleanupJob; print a bounded summary.

    Returns 0 when the invocation completed — whether a job was processed or none was eligible —
    including when the underlying service recorded the attempt as FAILED (an expected,
    already-persisted outcome such as the active-collection safety guard refusing to delete, not a
    script failure — partial/failed cleanup is never hidden or reported as success). Returns 1 only
    if the invocation itself failed unexpectedly (e.g. the database or vector store is
    unreachable); such failures are logged via the existing logging mechanism, never printed as a
    raw stack trace.
    """
    settings = get_settings()
    vector_store = get_vector_store(settings)

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
