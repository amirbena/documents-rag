"""add vector_cleanup_jobs table

Revision ID: 1c2d9f3a7b4e
Revises: 07f849bf2b95
Create Date: 2026-07-11 15:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = '1c2d9f3a7b4e'
down_revision: str | None = '07f849bf2b95'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'vector_cleanup_jobs',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('document_id', sa.String(length=36), nullable=False),
        sa.Column('collection_name', sa.String(length=255), nullable=False),
        sa.Column(
            'status',
            sa.Enum(
                'pending', 'failed', 'completed',
                name='vectorcleanupstatus', native_enum=False, length=20,
            ),
            nullable=False,
        ),
        sa.Column('attempts', sa.Integer(), nullable=False),
        sa.Column('last_error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id']),
    )
    op.create_index(
        'ix_vector_cleanup_jobs_document_id', 'vector_cleanup_jobs', ['document_id']
    )


def downgrade() -> None:
    op.drop_index('ix_vector_cleanup_jobs_document_id', table_name='vector_cleanup_jobs')
    op.drop_table('vector_cleanup_jobs')
