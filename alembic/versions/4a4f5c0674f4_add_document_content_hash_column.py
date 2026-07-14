"""add document content_hash column

Revision ID: 4a4f5c0674f4
Revises: c8f3a2b6d1e7
Create Date: 2026-07-14 09:00:00.000000

Persistence foundation for Phase 2.8.5 content-hash deduplication (subtask 1 of that phase —
schema only, no application behavior). Adds `documents.content_hash` (nullable `VARCHAR(64)`,
intended to eventually hold a lowercase hex SHA-256 digest of the uploaded bytes) and a named
unique index, `uq_documents_content_hash`, enforcing that two documents may never share the same
non-null hash. A normal (non-partial) unique index is sufficient: PostgreSQL never considers two
NULL values equal for uniqueness purposes, so any number of documents may have
`content_hash IS NULL` at once — which is exactly every existing row, since nothing populates
this column yet.

No backfill: existing rows are not read, updated, or migrated in any way — they simply keep
`content_hash = NULL`, which remains valid indefinitely (see app/models/document.py's docstring
on this column, and app/services/documents/upload_service.py, which does not populate it as of
this migration). Computing a hash requires reading a document's stored bytes from Local storage
or MinIO, which a database-only Alembic migration must never do.
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = '4a4f5c0674f4'
down_revision: str | None = 'c8f3a2b6d1e7'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INDEX_NAME = "uq_documents_content_hash"


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("content_hash", sa.String(length=64), nullable=True),
    )

    op.create_index(
        _INDEX_NAME,
        "documents",
        ["content_hash"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(_INDEX_NAME, table_name="documents")
    op.drop_column("documents", "content_hash")
