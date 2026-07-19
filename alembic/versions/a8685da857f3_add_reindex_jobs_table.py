"""add reindex_jobs table and one-active-per-document partial unique index

Revision ID: a8685da857f3
Revises: 4a4f5c0674f4
Create Date: 2026-07-15 09:00:00.000000

Introduces `reindex_jobs` (Phase 2.8.6, subtask 2 — durable re-index build attempts, see
`app/models/reindex_job.py` and `app/services/indexing/reindex_scheduling_service.py`) plus a
partial unique index enforcing at most one active (`pending`/`processing`) re-index job per
document, mirroring `b7e2f6a1c9d4`'s `ix_ingestion_jobs_one_active_per_document` and
`c8f3a2b6d1e7`'s `ix_document_deletion_jobs_one_active_per_document` pattern exactly.
`ReindexJob.status` is stored as a plain VARCHAR (native_enum=False), so the partial index's WHERE
clause matches the lowercase string values directly.

`target_collection_name` is a mandatory foreign key into `index_collections` — a re-index job
always references an already-persisted `IndexCollection` snapshot (created via
`ensure_active_collection()` at scheduling time) rather than duplicating the target's embedding
provider/model/dimension/version identity onto this table. `target_chunk_size`/
`target_chunk_overlap` are mandatory, since `IndexCollection`/`EmbeddingIndexConfig` carry only the
`chunking_version` label, never the numeric parameters themselves.

Like `c8f3a2b6d1e7`, this migration adds no defensive duplicate-row cleanup before creating the
partial unique index: `reindex_jobs` is a brand-new table created in this same migration, so no
duplicate-active-row data can possibly exist yet in any installation.

This migration is database-only: it creates no rows, schedules no re-index work, and does not
inspect Qdrant or object storage.
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = 'a8685da857f3'
down_revision: str | None = '4a4f5c0674f4'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE_NAME = "reindex_jobs"
_INDEX_NAME = "ix_reindex_jobs_one_active_per_document"


def upgrade() -> None:
    op.create_table(
        _TABLE_NAME,
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('document_id', sa.String(length=36), nullable=False),
        sa.Column('target_collection_name', sa.String(length=255), nullable=False),
        sa.Column('target_chunk_size', sa.Integer(), nullable=False),
        sa.Column('target_chunk_overlap', sa.Integer(), nullable=False),
        sa.Column(
            'status',
            sa.Enum(
                'pending', 'processing', 'completed', 'failed',
                name='reindexjobstatus', native_enum=False, length=20,
            ),
            nullable=False,
        ),
        sa.Column('error_message', sa.String(length=2048), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column(
            'updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False
        ),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id']),
        sa.ForeignKeyConstraint(['target_collection_name'], ['index_collections.collection_name']),
    )
    op.create_index(
        'ix_reindex_jobs_document_id', _TABLE_NAME, ['document_id']
    )
    op.create_index(
        'ix_reindex_jobs_status', _TABLE_NAME, ['status']
    )
    op.create_index(
        _INDEX_NAME,
        _TABLE_NAME,
        ['document_id'],
        unique=True,
        postgresql_where=sa.text("status IN ('pending', 'processing')"),
    )


def downgrade() -> None:
    op.drop_index(_INDEX_NAME, table_name=_TABLE_NAME)
    op.drop_index('ix_reindex_jobs_status', table_name=_TABLE_NAME)
    op.drop_index('ix_reindex_jobs_document_id', table_name=_TABLE_NAME)
    op.drop_table(_TABLE_NAME)
