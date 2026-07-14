"""Content-hash deduplication decision model for document upload (Phase 2.8.5).

Internal only — nothing here is imported by `app/api/v1/routes/documents.py` yet, and no Pydantic
schema exposes `UploadOutcome` or any of this module's exception types. `upload_service.py`'s
actual upload flow is unchanged except for populating the newly computed hash on a new `Document`
row; wiring this decision model into that flow (skipping storage/job creation on reuse, recovering
from the content_hash unique-index race) is deliberately deferred to a later subtask.

## Deletion precedence

`decide_upload()` evaluates a matching document's latest `DocumentDeletionJob` *before* its latest
`IngestionJob` — a document with any blocking deletion state must never be represented as a normal
upload-reuse outcome (`UploadOutcome` has no deletion-related member at all). Deletion conflicts
are signaled via typed exceptions instead, mirroring this repository's existing typed-outcome
service conventions (`RetryOutcome`, `DeletionRequestOutcome`) while keeping `UploadOutcome` itself
limited to the four reuse/creation states the upload route decision is actually about:

- `PENDING`/`PROCESSING` -> `DeletionActiveError` (an active deletion attempt is in flight).
- `PARTIALLY_FAILED` -> `DeletionIncompleteError` (a previous attempt left external resources
  behind; the old lifecycle still owns them and must be resolved first).
- `COMPLETED` -> `DeletionInvariantViolationError`. A completed deletion is expected to have
  released its `content_hash` (set it back to `NULL` — a later subtask's responsibility); finding
  a matching document at all means its `content_hash` is non-null, so observing a `COMPLETED`
  deletion job here means that release never happened. This is a data-invariant break, not a
  normal, expected conflict — it is not a new persisted deletion status.
"""

import hashlib
from dataclasses import dataclass
from enum import StrEnum

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_deletion_job import DocumentDeletionStatus
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.services.documents.deletion_service import get_latest_deletion_job
from app.services.documents.query_service import get_latest_ingestion_job


def compute_content_hash(content: bytes) -> str:
    """Return the lowercase hex SHA-256 digest of `content` — the exact uploaded bytes.

    Pure and deterministic: identical bytes always produce the same 64-character result: no
    normalization, no configurable algorithm, no storage or database access. `content` is already
    fully buffered in memory by the time it reaches this function (the upload route already reads
    the whole file before any processing begins), so no streaming/incremental hashing is needed.
    """
    return hashlib.sha256(content).hexdigest()


class UploadOutcome(StrEnum):
    """The upload-time decision for a matching (or absent) document — reuse states only.

    Deletion-blocking states are never represented here — see the module docstring's "Deletion
    precedence" section for why they are raised as exceptions instead.
    """

    CREATED = "created"
    REUSED_ACTIVE = "reused_active"
    REUSED_INDEXED = "reused_indexed"
    REUSED_FAILED = "reused_failed"


class DeletionActiveError(Exception):
    """A `PENDING`/`PROCESSING` deletion attempt exists for the matching document.

    Uploading identical content while its previous instance is being deleted is a conflict, never
    a reuse outcome — the caller must not treat this as `REUSED_*`.
    """

    def __init__(self, document_id: str) -> None:
        self.document_id = document_id
        super().__init__(f"Document {document_id} has an active deletion in progress.")


class DeletionIncompleteError(Exception):
    """A `PARTIALLY_FAILED` deletion attempt exists for the matching document.

    The old lifecycle still owns external resources (vectors and/or the stored object) that were
    never fully cleaned up — it must be resolved (retried to completion) before this content can
    be treated as available again.
    """

    def __init__(self, document_id: str) -> None:
        self.document_id = document_id
        super().__init__(f"Document {document_id} has an incomplete (partially failed) deletion.")


class DeletionInvariantViolationError(Exception):
    """A `COMPLETED` deletion exists for the matching document, but its content_hash is non-null.

    A completed deletion is expected to release its content_hash back to NULL so the hash becomes
    available for a genuinely new upload — this exception means that release never happened,
    which is a data-invariant break, not an expected, normal conflict.
    """

    def __init__(self, document_id: str) -> None:
        self.document_id = document_id
        super().__init__(
            f"Document {document_id} has a COMPLETED deletion but a non-null content_hash — "
            "the hash should have been released on deletion completion."
        )


@dataclass(frozen=True)
class UploadDecision:
    """The outcome of `decide_upload()`: what an upload with this content_hash should do.

    `document`/`ingestion_job` are populated for every `REUSED_*` outcome (the existing document
    and its latest ingestion attempt); both are `None` for `CREATED`, since no matching document
    exists yet.
    """

    outcome: UploadOutcome
    document: Document | None
    ingestion_job: IngestionJob | None


async def find_document_by_content_hash(session: AsyncSession, content_hash: str) -> Document | None:
    """Return the Document whose content_hash matches `content_hash`, or None.

    `content_hash` is enforced unique (non-null) at the database level (see
    `uq_documents_content_hash`), so at most one row can ever match.
    """
    stmt = select(Document).where(Document.content_hash == content_hash).limit(1)
    result = await session.execute(stmt)
    return result.scalars().first()


async def decide_upload(session: AsyncSession, content_hash: str) -> UploadDecision:
    """Decide what an upload with `content_hash` should do, given any existing matching document.

    Issues at most three queries total for the single document a hash can match (the hash lookup,
    plus one latest-deletion-job and one latest-ingestion-job lookup) — a fixed, non-N+1 cost
    regardless of how many other documents exist, since content_hash uniqueness means there is
    never more than one candidate row to investigate.

    Raises `DeletionActiveError`/`DeletionIncompleteError`/`DeletionInvariantViolationError` if the
    matching document's deletion state blocks treating it as reusable — see the module docstring.
    """
    document = await find_document_by_content_hash(session, content_hash)
    if document is None:
        return UploadDecision(outcome=UploadOutcome.CREATED, document=None, ingestion_job=None)

    latest_deletion_job = await get_latest_deletion_job(session, document.id)
    if latest_deletion_job is not None:
        if latest_deletion_job.status in (
            DocumentDeletionStatus.PENDING,
            DocumentDeletionStatus.PROCESSING,
        ):
            raise DeletionActiveError(document.id)
        if latest_deletion_job.status == DocumentDeletionStatus.PARTIALLY_FAILED:
            raise DeletionIncompleteError(document.id)
        if latest_deletion_job.status == DocumentDeletionStatus.COMPLETED:
            raise DeletionInvariantViolationError(document.id)

    latest_job = await get_latest_ingestion_job(session, document.id)
    if latest_job is None:
        # Structurally unreachable via the normal upload flow: upload_document() always creates
        # exactly one Document and one IngestionJob in the same commit, so a document reachable
        # by content_hash always has at least one job — mirrors query_service's own defensive
        # (never actually hit) "no job at all" case.
        raise RuntimeError(f"Document {document.id} matched by content_hash has no ingestion job.")

    if latest_job.status in (IngestionStatus.PENDING, IngestionStatus.PROCESSING):
        outcome = UploadOutcome.REUSED_ACTIVE
    elif latest_job.status == IngestionStatus.COMPLETED:
        outcome = UploadOutcome.REUSED_INDEXED
    else:
        outcome = UploadOutcome.REUSED_FAILED

    return UploadDecision(outcome=outcome, document=document, ingestion_job=latest_job)


__all__ = [
    "DeletionActiveError",
    "DeletionIncompleteError",
    "DeletionInvariantViolationError",
    "UploadDecision",
    "UploadOutcome",
    "compute_content_hash",
    "decide_upload",
    "find_document_by_content_hash",
]
