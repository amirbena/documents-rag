"""Standalone operational script: process pending DocumentDeletionJob rows and print a summary.

Run via `make process-pending-document-deletions` or
`python scripts/process_pending_document_deletions.py`. Not part of `make verify`/`make test*`/CI
— mirrors `scripts/recover_stale_ingestion_jobs.py`'s style exactly: a plain script invoked
directly against the app's real `get_settings()`/DB session/provider-factory machinery, intended
for manual or future-scheduled invocation against a real deployment, never the test suite.
Processes jobs one at a time in a loop until no PENDING job remains (bounded by `batch_size`),
using `DocumentDeletionWorker` — the same execution logic Backend E2E tests drive directly.
"""

import asyncio

from app.core.config import get_settings
from app.db.session import async_session_factory
from app.rag.providers.provider_factory import get_vector_store
from app.services.document_deletion_service import DocumentDeletionWorker
from app.storage.factory import create_file_storage

_MAX_JOBS_PER_RUN = 100


async def main() -> int:
    """Process up to `_MAX_JOBS_PER_RUN` pending document deletions and print a summary."""
    settings = get_settings()
    worker = DocumentDeletionWorker(
        vector_store=get_vector_store(settings), file_storage=create_file_storage()
    )

    processed = 0
    for _ in range(_MAX_JOBS_PER_RUN):
        async with async_session_factory() as session:
            job = await worker.process_next_job(session)
        if job is None:
            break
        processed += 1
        print(f"  job={job.id} document={job.document_id} -> {job.status}")

    print(f"Processed {processed} document deletion job(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
