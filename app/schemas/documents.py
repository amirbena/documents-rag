"""Response schemas for document upload/ingestion endpoints."""

from pydantic import BaseModel

from app.models.ingestion_job import IngestionStatus


class DocumentUploadResponse(BaseModel):
    """Shape returned by POST /api/v1/documents."""

    document_id: str
    job_id: str
    status: IngestionStatus
