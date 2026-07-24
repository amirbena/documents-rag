"""current schema baseline

Revision ID: a1a302e871c3
Revises:
Create Date: 2026-07-22 00:32:08.668690

Phase 2.10 — replaces the 9 incremental development migrations this project accumulated
(`acf1b01d5a02` through `fb63f21089ca`) with a single baseline representing the complete current
schema. Per the approved policy, no deployed or persistent database requires an in-place upgrade
from those revisions — see `alembic/README.md` for the full rationale, the exact reset commands,
and what to do with an existing local database.

Generated via `alembic revision --autogenerate` against a genuinely empty PostgreSQL database
(with the 9 deleted migrations temporarily removed, so Alembic's own history was empty too), then
hand-verified column-by-column and constraint-by-constraint against every deleted migration file
and every current `app/models/*.py` module before being trusted. Five schema objects exist in the
database but are **not** represented in SQLAlchemy's ORM metadata at all, so autogenerate cannot
see them; they are added below manually, using the exact definitions from the migrations that
originally created them:

- `ix_ingestion_jobs_one_active_per_document` (from deleted revision `b7e2f6a1c9d4`)
- `ix_document_deletion_jobs_one_active_per_document` (from deleted revision `c8f3a2b6d1e7`)
- `ix_reindex_jobs_one_active_per_document` (from deleted revision `a8685da857f3`)
- `uq_documents_content_hash` (from deleted revision `4a4f5c0674f4`)
- `ix_vector_cleanup_jobs_document_id` (from deleted revision `1c2d9f3a7b4e`)

The first three are the partial unique "at most one active job per document" indexes CLAUDE.md's
High-Risk Invariants describe as "enforced by a real Postgres partial unique index, never
application logic alone" — deliberately not declared via SQLAlchemy `Index`/`__table_args__` on
the corresponding models (`IngestionJob`, `DocumentDeletionJob`, `ReindexJob`), which is exactly
why `alembic revision --autogenerate` has never been able to see them and never will unless a
model changes; that is a pre-existing condition, not something this baseline changes or hides.
Confirmed identical before and after this reset: `alembic revision --autogenerate` against a
database built from the old 9-migration chain, and against a database built from this baseline,
each report exactly these same five objects as the only diff versus current ORM metadata —
nothing else differs.

Two foreign keys are given the same explicit names their original migrations used
(`fk_documents_collection_name_index_collections`,
`fk_reindex_jobs_source_collection_name_index_collections`); every other foreign key here was
unnamed in its original migration too, so it stays unnamed (database-assigned name), exactly
matching prior behavior.
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = 'a1a302e871c3'
down_revision: str | None = None
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
    op.create_table(
        'documents',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('original_filename', sa.String(length=1024), nullable=False),
        sa.Column('stored_filename', sa.String(length=255), nullable=False),
        sa.Column('content_type', sa.String(length=255), nullable=False),
        sa.Column('file_size', sa.Integer(), nullable=False),
        sa.Column('stored_path', sa.String(length=2048), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('storage_provider', sa.String(length=32), nullable=True),
        sa.Column('storage_bucket', sa.String(length=255), nullable=True),
        sa.Column('storage_key', sa.String(length=2048), nullable=True),
        sa.Column('storage_etag', sa.String(length=255), nullable=True),
        sa.Column('content_hash', sa.String(length=64), nullable=True),
        sa.Column('embedding_provider', sa.String(length=64), nullable=True),
        sa.Column('embedding_model', sa.String(length=255), nullable=True),
        sa.Column('embedding_dimension', sa.Integer(), nullable=True),
        sa.Column('embedding_version', sa.String(length=64), nullable=True),
        sa.Column('chunking_version', sa.String(length=64), nullable=True),
        sa.Column('collection_name', sa.String(length=255), nullable=True),
        sa.Column('indexed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ['collection_name'], ['index_collections.collection_name'],
            name='fk_documents_collection_name_index_collections',
        ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('stored_filename'),
    )
    op.create_index('uq_documents_content_hash', 'documents', ['content_hash'], unique=True)

    op.create_table(
        'ingestion_jobs',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('document_id', sa.String(length=36), nullable=False),
        sa.Column(
            'status',
            sa.Enum(
                'pending', 'processing', 'completed', 'failed',
                name='ingestionstatus', native_enum=False, length=20,
            ),
            nullable=False,
        ),
        sa.Column('error_message', sa.String(length=2048), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_ingestion_jobs_one_active_per_document',
        'ingestion_jobs',
        ['document_id'],
        unique=True,
        postgresql_where=sa.text("status IN ('pending', 'processing')"),
    )

    op.create_table(
        'document_deletion_jobs',
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
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_document_deletion_jobs_document_id', 'document_deletion_jobs', ['document_id'])
    op.create_index('ix_document_deletion_jobs_status', 'document_deletion_jobs', ['status'])
    op.create_index(
        'ix_document_deletion_jobs_one_active_per_document',
        'document_deletion_jobs',
        ['document_id'],
        unique=True,
        postgresql_where=sa.text("status IN ('pending', 'processing')"),
    )

    op.create_table(
        'reindex_jobs',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('document_id', sa.String(length=36), nullable=False),
        sa.Column('source_collection_name', sa.String(length=255), nullable=False),
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
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('activated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id']),
        sa.ForeignKeyConstraint(
            ['source_collection_name'], ['index_collections.collection_name'],
            name='fk_reindex_jobs_source_collection_name_index_collections',
        ),
        sa.ForeignKeyConstraint(['target_collection_name'], ['index_collections.collection_name']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_reindex_jobs_document_id', 'reindex_jobs', ['document_id'])
    op.create_index('ix_reindex_jobs_status', 'reindex_jobs', ['status'])
    op.create_index(
        'ix_reindex_jobs_one_active_per_document',
        'reindex_jobs',
        ['document_id'],
        unique=True,
        postgresql_where=sa.text("status IN ('pending', 'processing')"),
    )

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
    op.create_index('ix_vector_cleanup_jobs_document_id', 'vector_cleanup_jobs', ['document_id'])


def downgrade() -> None:
    op.drop_table('vector_cleanup_jobs')

    op.drop_index('ix_reindex_jobs_one_active_per_document', table_name='reindex_jobs')
    op.drop_index('ix_reindex_jobs_status', table_name='reindex_jobs')
    op.drop_index('ix_reindex_jobs_document_id', table_name='reindex_jobs')
    op.drop_table('reindex_jobs')

    op.drop_index('ix_document_deletion_jobs_one_active_per_document', table_name='document_deletion_jobs')
    op.drop_index('ix_document_deletion_jobs_status', table_name='document_deletion_jobs')
    op.drop_index('ix_document_deletion_jobs_document_id', table_name='document_deletion_jobs')
    op.drop_table('document_deletion_jobs')

    op.drop_index('ix_ingestion_jobs_one_active_per_document', table_name='ingestion_jobs')
    op.drop_table('ingestion_jobs')

    op.drop_index('uq_documents_content_hash', table_name='documents')
    op.drop_table('documents')

    op.drop_table('index_collections')
