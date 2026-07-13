"""Document upload endpoint: stores the file and queues a pending ingestion job.

Does not parse, chunk, embed, or index the document — that happens in a later milestone,
outside the request. This endpoint only saves the file and creates DB rows, via
app.services.document_upload_service.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.schemas.documents import DocumentUploadResponse
from app.services.document_upload_service import upload_document
from app.storage.contract import FileStorage
from app.storage.factory import create_file_storage

router = APIRouter()


def get_file_storage() -> FileStorage:
    """Build the configured FileStorage implementation via the storage factory."""
    return create_file_storage()


@router.post(
    "/documents",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_document_route(
    file: UploadFile,
    db: AsyncSession = Depends(get_db_session),
    storage: FileStorage = Depends(get_file_storage),
) -> DocumentUploadResponse:
    """Save the uploaded file, create Document + pending IngestionJob rows, return 202."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    original_filename = file.filename or "unnamed"
    content_type = file.content_type or "application/octet-stream"

    document, job = await upload_document(
        content=content,
        original_filename=original_filename,
        content_type=content_type,
        storage=storage,
        session=db,
    )

    return DocumentUploadResponse(document_id=document.id, job_id=job.id, status=job.status)
