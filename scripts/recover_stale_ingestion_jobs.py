"""Standalone operational script: run one stale-PROCESSING-job recovery batch and print a summary.

Run via `make recover-stale-ingestion-jobs` or `python scripts/recover_stale_ingestion_jobs.py`.
Not part of `make verify`/`make test*`/CI — mirrors `scripts/smoke_multilingual_real.py`'s style
(a plain script invoked directly, not a `python -m app.cli...` package) since this repo has no
existing `app/cli/` convention to extend; introducing one for a single operation would be more
machinery than the task warrants. Connects via the app's real `get_settings()`/DB session
machinery (`app.db.session`) against whatever DATABASE_URL is configured — intended for manual or
future-scheduled invocation against a real deployment, not the test suite.
"""

import asyncio

from app.core.config import get_settings
from app.db.session import async_session_factory
from app.services.ingestion_retry_service import recover_stale_ingestion_jobs


async def main() -> int:
    """Run one recover_stale_ingestion_jobs() batch against the configured database and print it."""
    settings = get_settings()

    async with async_session_factory() as session:
        result = await recover_stale_ingestion_jobs(
            session,
            batch_size=settings.ingestion_recovery_batch_size,
            stale_after_seconds=settings.ingestion_stale_after_seconds,
        )

    print(f"Recovered {result.count} stale ingestion job(s).")
    for entry in result.recovered:
        print(f"  stale_job={entry.stale_job_id} -> replacement_job={entry.replacement_job_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
