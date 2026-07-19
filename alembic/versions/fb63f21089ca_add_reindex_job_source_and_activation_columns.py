"""add reindex_jobs source_collection_name and activated_at columns

Revision ID: fb63f21089ca
Revises: a8685da857f3
Create Date: 2026-07-16 09:00:00.000000

Phase 2.8.6, subtask 5 — adds the two columns atomic reindex activation
(`app/services/indexing/reindex_activation.py`) requires:

- `source_collection_name` (mandatory foreign key into `index_collections`, mirroring
  `target_collection_name`'s existing FK exactly): the collection the document was actually
  serving from at the moment this job was scheduled. Captured once by
  `reindex_scheduling_service.schedule_reindex()` and never changed afterward — activation compares
  it against the document's *current* `collection_name` to detect staleness (a different job for
  the same document activated first, moving the document to a third collection this job never knew
  about) before ever overwriting anything.
- `activated_at` (nullable): the durable marker that cutover actually happened, deliberately
  separate from `completed_at` (build success) — a job may be `COMPLETED` with `activated_at IS
  NULL` indefinitely, and remains `COMPLETED` after activation too.

`reindex_jobs` has no legitimate existing rows at this point in the project's history (this table
was introduced in migration `a8685da857f3`, and no runtime code path outside this branch's own
tests has ever written to it), so `source_collection_name` can be added as `NOT NULL` directly,
with no backfill and no server default.
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = 'fb63f21089ca'
down_revision: str | None = 'a8685da857f3'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE_NAME = "reindex_jobs"
_SOURCE_FK_NAME = "fk_reindex_jobs_source_collection_name_index_collections"


def upgrade() -> None:
    op.add_column(_TABLE_NAME, sa.Column("source_collection_name", sa.String(length=255), nullable=False))
    op.add_column(_TABLE_NAME, sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True))
    op.create_foreign_key(
        _SOURCE_FK_NAME,
        _TABLE_NAME,
        "index_collections",
        ["source_collection_name"],
        ["collection_name"],
    )


def downgrade() -> None:
    op.drop_constraint(_SOURCE_FK_NAME, _TABLE_NAME, type_="foreignkey")
    op.drop_column(_TABLE_NAME, "activated_at")
    op.drop_column(_TABLE_NAME, "source_collection_name")
