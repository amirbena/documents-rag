"""add document storage identity columns

Revision ID: a3f9c7d2e1b5
Revises: 1c2d9f3a7b4e
Create Date: 2026-07-13 12:00:00.000000

Adds the provider-neutral storage identity columns introduced by Phase 2.6/2.7's FileStorage
abstraction: storage_provider, storage_bucket, storage_key, storage_etag. `stored_path`/
`stored_filename` are kept as-is (not dropped, not renamed) for backward read compatibility —
this is the smaller, safer option versus rewriting existing rows' addressing scheme.

Existing rows are backfilled with storage_provider='local' and storage_key=stored_filename: the
pre-migration LocalFileStorage always wrote files flat under its configured root, keyed by
stored_filename — that value is exactly the object key the new LocalFileStorage needs to locate
the same file, so no file content is read and no data is moved. storage_bucket/storage_etag stay
NULL for these rows (local storage has neither). New uploads populate all four columns going
forward via app/services/document_upload_service.py.
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = 'a3f9c7d2e1b5'
down_revision: str | None = '1c2d9f3a7b4e'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

documents = sa.table(
    "documents",
    sa.column("storage_provider", sa.String),
    sa.column("storage_key", sa.String),
    sa.column("stored_filename", sa.String),
)


def upgrade() -> None:
    op.add_column('documents', sa.Column('storage_provider', sa.String(length=32), nullable=True))
    op.add_column('documents', sa.Column('storage_bucket', sa.String(length=255), nullable=True))
    op.add_column('documents', sa.Column('storage_key', sa.String(length=2048), nullable=True))
    op.add_column('documents', sa.Column('storage_etag', sa.String(length=255), nullable=True))

    op.execute(
        documents.update()
        .where(documents.c.storage_key.is_(None))
        .values(storage_provider='local', storage_key=documents.c.stored_filename)
    )


def downgrade() -> None:
    op.drop_column('documents', 'storage_etag')
    op.drop_column('documents', 'storage_key')
    op.drop_column('documents', 'storage_bucket')
    op.drop_column('documents', 'storage_provider')
