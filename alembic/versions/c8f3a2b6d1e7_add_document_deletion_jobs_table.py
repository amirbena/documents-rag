"""add document_deletion_jobs table and one-active-per-document partial unique index

Revision ID: c8f3a2b6d1e7
Revises: b7e2f6a1c9d4
Create Date: 2026-07-13 14:00:00.000000

Introduces `document_deletion_jobs` (Phase 2.8.4 full document deletion — see
`app/models/document_deletion_job.py` and `app/services/documents/deletion_service.py`) plus a
partial unique index enforcing at most one active (`pending`/`processing`) deletion job per
document, mirroring `b7e2f6a1c9d4`'s `ix_ingestion_jobs_one_active_per_document` pattern exactly.
`DocumentDeletionJob.status` is stored as a plain VARCHAR (native_enum=False), so the partial
index's WHERE clause matches the lowercase string values directly.

Unlike `b7e2f6a1c9d4`, this migration adds **no defensive duplicate-row cleanup** before creating
the index: `document_deletion_jobs` is a brand-new table created in this same migration, so no
duplicate-active-row data can possibly exist yet in any installation — there is no reachable path
to duplicate data here, not merely an unreachable-in-practice one (contrast this with
`b7e2f6a1c9d4`, which added its index onto a pre-existing, already-populated `ingestion_jobs`
table and therefore needed a defensive cleanup step as a genuine safety net).
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = 'c8f3a2b6d1e7'
down_revision: str | None = 'b7e2f6a1c9d4'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE_NAME = "document_deletion_jobs"
_INDEX_NAME = "ix_document_deletion_jobs_one_active_per_document"


def upgrade() -> None:
    op.create_table(
        _TABLE_NAME,
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('document_id', sa.String(length=36), nullable=False),
        sa.Column(
            'status',
            sa.Enum(
                'pending', 'processing', 'partially_failed', 'completed',
                name='documentdeletionstatus', native_enum=False, length=20,
            ),
            nullable=False,
        ),
        sa.Column('vector_cleanup_completed', sa.Boolean(), nullable=False),
        sa.Column('storage_cleanup_completed', sa.Boolean(), nullable=False),
        sa.Column('error_code', sa.String(length=64), nullable=True),
        sa.Column('error_message', sa.String(length=2048), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column(
            'updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False
        ),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id']),
    )
    op.create_index(
        'ix_document_deletion_jobs_document_id', _TABLE_NAME, ['document_id']
    )
    op.create_index(
        'ix_document_deletion_jobs_status', _TABLE_NAME, ['status']
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
    op.drop_index('ix_document_deletion_jobs_status', table_name=_TABLE_NAME)
    op.drop_index('ix_document_deletion_jobs_document_id', table_name=_TABLE_NAME)
    op.drop_table(_TABLE_NAME)
