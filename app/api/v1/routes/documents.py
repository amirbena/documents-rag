"""Document upload endpoint: stores the file and queues a pending ingestion job.

Does not parse, chunk, embed, or index the document — that happens in a later milestone,
outside the request. This endpoint only saves the file and creates DB rows.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.models.document import Document
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.schemas.documents import DocumentUploadResponse
from app.services.local_file_storage import LocalFileStorage

router = APIRouter()


def get_local_file_storage() -> LocalFileStorage:
    """Build a LocalFileStorage instance."""
    return LocalFileStorage()


@router.post(
    "/documents",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_document(
    file: UploadFile,
    db: AsyncSession = Depends(get_db_session),
    storage: LocalFileStorage = Depends(get_local_file_storage),
) -> DocumentUploadResponse:
    """Save the uploaded file, create Document + pending IngestionJob rows, return 202."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    original_filename = file.filename or "unnamed"
    content_type = file.content_type or "application/octet-stream"

    stored_filename, stored_path = await storage.save(content, original_filename)

    document = Document(
        id=str(uuid.uuid4()),
        original_filename=original_filename,
        stored_filename=stored_filename,
        content_type=content_type,
        file_size=len(content),
        stored_path=stored_path,
    )
    db.add(document)

    job = IngestionJob(
        id=str(uuid.uuid4()),
        document_id=document.id,
        status=IngestionStatus.PENDING,
    )
    db.add(job)

    await db.commit()

    return DocumentUploadResponse(document_id=document.id, job_id=job.id, status=job.status)
