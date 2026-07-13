"""add one-active-ingestion-job-per-document partial unique index

Revision ID: b7e2f6a1c9d4
Revises: a3f9c7d2e1b5
Create Date: 2026-07-13 13:00:00.000000

Enforces, at the database level, that a document has at most one "active" (pending or
processing) IngestionJob row at a time — see app/services/ingestion/retry_service.py and
app/services/ingestion/stale_recovery_service.py (Phase 2.8.3). `IngestionJob.status` is stored
as a plain VARCHAR (native_enum=False, see
app/models/ingestion_job.py), so the WHERE clause below matches the lowercase string values
directly ('pending'/'processing'), not a native Postgres enum type.

Duplicate-active-row backfill: checked whether two active (pending/processing) IngestionJob rows
could already exist for the same document_id in this codebase before this migration.
`app.services.documents.upload_service.upload_document()` creates exactly one PENDING job per
upload, and prior to this PR there was no retry/re-index path that could create a second job
while an existing one was still pending/processing — `app.services.indexing.reindex_service` (unrelated,
pre-existing) does not go through IngestionJob at all. So there is no reachable path to duplicate
active rows in any installation that only ever ran code up to this PR, and no data backfill/
resolution step is needed here. The `upgrade()` step below still runs a defensive cleanup query
first (deterministic: keep the newest active row per document, mark any older active duplicates
FAILED with a fixed migration-reason message) so the index creation cannot fail even if this
assumption turns out to be wrong for some out-of-band data, without silently dropping any row.
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = 'b7e2f6a1c9d4'
down_revision: str | None = 'a3f9c7d2e1b5'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INDEX_NAME = "ix_ingestion_jobs_one_active_per_document"
_MIGRATION_DUPLICATE_RESOLUTION_MESSAGE = (
    "STALE_PROCESSING_RECOVERED: superseded by a newer active ingestion job during the "
    "one-active-job-per-document migration (b7e2f6a1c9d4)."
)


def upgrade() -> None:
    # Defensive, idempotent cleanup: for any document with more than one active
    # (pending/processing) job, keep only the most recently created one active and mark the
    # rest FAILED. See module docstring — this is believed unreachable given the current
    # codebase, but the migration does not assume that at the cost of a failed index build.
    op.execute(
        sa.text(
            """
            UPDATE ingestion_jobs
            SET status = 'failed',
                error_message = :message
            WHERE status IN ('pending', 'processing')
              AND id NOT IN (
                  SELECT DISTINCT ON (document_id) id
                  FROM ingestion_jobs
                  WHERE status IN ('pending', 'processing')
                  ORDER BY document_id, created_at DESC, id DESC
              )
            """
        ).bindparams(message=_MIGRATION_DUPLICATE_RESOLUTION_MESSAGE)
    )

    op.create_index(
        _INDEX_NAME,
        "ingestion_jobs",
        ["document_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('pending', 'processing')"),
    )


def downgrade() -> None:
    op.drop_index(_INDEX_NAME, table_name="ingestion_jobs")
