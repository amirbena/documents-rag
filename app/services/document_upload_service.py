"""Saves an uploaded file's content to storage and persists its Document + IngestionJob rows.

Storage and PostgreSQL are not one atomic transaction — see "Cross-system boundary" in
ARCHITECTURE.md. The sequence is: save the object to `FileStorage`, then persist `Document` +
`IngestionJob` and commit. If the commit fails after the object was already saved, a best-effort
delete of that object is attempted so a DB failure doesn't silently leave an orphaned object
behind; the *original* DB exception is always what propagates, and a cleanup failure is logged
(not raised, not hidden) rather than masking the original error.
"""

import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.models.document import Document
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.storage.contract import FileStorage
from app.storage.keys import generate_object_key

logger = logging.getLogger(__name__)


async def upload_document(
    *,
    content: bytes,
    original_filename: str,
    content_type: str,
    storage: FileStorage,
    session: AsyncSession,
    settings: Settings | None = None,
) -> tuple[Document, IngestionJob]:
    """Save `content` to storage, persist Document + pending IngestionJob rows, and commit.

    On a DB commit failure after the object was already saved, attempts a best-effort delete of
    the just-saved object before re-raising the original DB exception unchanged.
    """
    settings = settings or get_settings()
    document_id = str(uuid.uuid4())
    key = generate_object_key(document_id, original_filename)

    stored = await storage.save(key, content, content_type=content_type)

    bucket = settings.minio_bucket if settings.file_storage_provider == "minio" else None
    document = Document(
        id=document_id,
        original_filename=original_filename,
        stored_filename=key.rsplit("/", 1)[-1],
        content_type=content_type,
        file_size=len(content),
        stored_path=key,
        storage_provider=settings.file_storage_provider,
        storage_bucket=bucket,
        storage_key=stored.key,
        storage_etag=stored.etag,
    )
    session.add(document)

    job = IngestionJob(id=str(uuid.uuid4()), document_id=document.id, status=IngestionStatus.PENDING)
    session.add(job)

    try:
        await session.commit()
    except Exception:
        try:
            await storage.delete(key)
        except Exception:
            logger.warning(
                "Failed to clean up orphaned storage object %r after a DB commit failure.", key
            )
        raise

    return document, job
