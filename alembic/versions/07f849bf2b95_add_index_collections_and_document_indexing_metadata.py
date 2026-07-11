"""add index_collections and document indexing metadata

Revision ID: 07f849bf2b95
Revises: acf1b01d5a02
Create Date: 2026-07-11 12:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = '07f849bf2b95'
down_revision: str | None = 'acf1b01d5a02'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'index_collections',
        sa.Column('collection_name', sa.String(length=255), nullable=False),
        sa.Column('embedding_provider', sa.String(length=64), nullable=False),
        sa.Column('embedding_model', sa.String(length=255), nullable=False),
        sa.Column('embedding_dimension', sa.Integer(), nullable=False),
        sa.Column('embedding_version', sa.String(length=64), nullable=False),
        sa.Column('chunking_version', sa.String(length=64), nullable=False),
        sa.Column(
            'status',
            sa.Enum(
                'active', 'retired',
                name='indexcollectionstatus', native_enum=False, length=20,
            ),
            nullable=False,
        ),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('collection_name'),
    )
    op.add_column('documents', sa.Column('embedding_provider', sa.String(length=64), nullable=True))
    op.add_column('documents', sa.Column('embedding_model', sa.String(length=255), nullable=True))
    op.add_column('documents', sa.Column('embedding_dimension', sa.Integer(), nullable=True))
    op.add_column('documents', sa.Column('embedding_version', sa.String(length=64), nullable=True))
    op.add_column('documents', sa.Column('chunking_version', sa.String(length=64), nullable=True))
    op.add_column('documents', sa.Column('collection_name', sa.String(length=255), nullable=True))
    op.add_column('documents', sa.Column('indexed_at', sa.DateTime(timezone=True), nullable=True))
    op.create_foreign_key(
        'fk_documents_collection_name_index_collections',
        'documents', 'index_collections',
        ['collection_name'], ['collection_name'],
    )


def downgrade() -> None:
    op.drop_constraint('fk_documents_collection_name_index_collections', 'documents', type_='foreignkey')
    op.drop_column('documents', 'indexed_at')
    op.drop_column('documents', 'collection_name')
    op.drop_column('documents', 'chunking_version')
    op.drop_column('documents', 'embedding_version')
    op.drop_column('documents', 'embedding_dimension')
    op.drop_column('documents', 'embedding_model')
    op.drop_column('documents', 'embedding_provider')
    op.drop_table('index_collections')
