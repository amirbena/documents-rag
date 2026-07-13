"""Original-content download: storage-key resolution, byte loading, and download-specific results.

Owns `GET /api/v1/documents/{document_id}/download`'s entire behavior: resolving which storage key
holds a document's original bytes, reading them through the injected `FileStorage`, deriving
content-type/filename response metadata, and mapping every failure mode (missing storage object,
storage backend unavailable, deleted document) to the correct HTTP status. This is the only
document module that ever calls `FileStorage` — `app.services.documents.query_service` never does.

Reuses `query_service.get_document()` (a single `session.get(Document, ...)` call) rather than
duplicating it — a narrow, one-way dependency (`download_service -> query_service`) that
introduces no cycle, since `query_service` never imports from this module.
"""

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_deletion_job import DocumentDeletionStatus
from app.services.documents.deletion_service import get_latest_deletion_job
from app.services.documents.query_service import get_document
from app.storage.contract import FileStorage
from app.storage.errors import StorageObjectNotFoundError, StorageReadError, StorageUnavailableError
from app.storage.keys import resolve_document_storage_key


@dataclass(frozen=True)
class DocumentDownloadResult:
    """Typed outcome of a download attempt: content (on success) plus the HTTP status to apply.

    `detail` carries a safe, fixed message for non-200 outcomes — never a raw storage exception
    message. Storage I/O (`FileStorage.read`) happens here, not in the route, so the route stays
    a thin dependency-injection + status-copy controller.
    """

    status_code: int
    content: bytes | None = None
    content_type: str | None = None
    original_filename: str | None = None
    detail: str | None = None


def resolve_download_key(document: Document) -> str:
    """Resolve the storage key to read a document's original content from — see app.storage.keys."""
    return resolve_document_storage_key(document)


async def download_document(
    session: AsyncSession, document_id: str, storage: FileStorage
) -> DocumentDownloadResult:
    """Resolve a document, read its original bytes from `storage`, and return a typed result.

    404 if the document row doesn't exist; 410 if the document was successfully deleted (Phase
    2.8.4 — the Postgres resource still exists, only its content was intentionally removed, so
    this is Gone rather than Not Found); 409 if the row exists but its storage object is
    otherwise missing (a real document/storage inconsistency); 503 if the storage backend itself
    is unreachable/failing for a reason other than not-found. Never returns a raw storage
    exception message or a local filesystem path.
    """
    document = await get_document(session, document_id)
    if document is None:
        return DocumentDownloadResult(status_code=404, detail="Document not found.")

    latest_deletion_job = await get_latest_deletion_job(session, document_id)
    if latest_deletion_job is not None and latest_deletion_job.status == DocumentDeletionStatus.COMPLETED:
        return DocumentDownloadResult(status_code=410, detail="Document has been deleted.")

    key = resolve_download_key(document)
    try:
        content = await storage.read(key)
    except StorageObjectNotFoundError:
        return DocumentDownloadResult(
            status_code=409, detail="Document exists but its content is unavailable in storage."
        )
    except (StorageUnavailableError, StorageReadError):
        return DocumentDownloadResult(status_code=503, detail="Storage backend is unavailable.")

    return DocumentDownloadResult(
        status_code=200,
        content=content,
        content_type=document.content_type,
        original_filename=document.original_filename,
    )


__all__ = [
    "DocumentDownloadResult",
    "download_document",
    "resolve_download_key",
]
