"""Saves an uploaded file's content to storage and persists its Document + IngestionJob rows —
now content-hash-deduplicated (Phase 2.8.5, subtask 3).

## Fast path, then the existing creation sequence, unchanged

Before ever writing to storage, `upload_document()` computes the content hash and calls
`dedup_service.decide_upload()`. A matching reusable document (or a blocking deletion state,
raised as a typed exception) is handled with zero storage writes and zero new rows — see that
module for the full lifecycle-precedence rule. Only a `CREATED` decision continues through the
exact pre-existing sequence: `storage.save()`, then persist `Document` (now with `content_hash`)
+ `IngestionJob`, then commit. Storage and PostgreSQL are still not one atomic transaction — see
"Cross-system boundary" in ARCHITECTURE.md.

## The database unique index is the concurrency guarantee, not the fast-path lookup

Two requests can both observe "no match" before either commits — the fast-path `SELECT` is an
optimization, never the safety mechanism. When the real race lands, both attempts write their own
storage object, both attempt to insert a `Document` with the same `content_hash`, and
`uq_documents_content_hash` lets exactly one commit succeed. The loser:

1. rolls back its transaction;
2. confirms the failure is `uq_documents_content_hash` specifically
   (`dedup_service.is_content_hash_violation()`) — never inferred from message text, and never
   assumed for *any* `IntegrityError` (a `stored_filename` collision, a foreign-key violation, a
   NOT NULL violation, or any other constraint is re-raised unchanged after the same best-effort
   storage cleanup, exactly like the pre-dedup behavior);
3. best-effort deletes only *its own* just-written object (keyed by its own generated
   `document_id`, never derived from the content hash — see `app.storage.keys`);
4. reloads the winner by content_hash and re-runs the full lifecycle decision against it (its
   lifecycle may have changed between the winner's commit and this reload) — if that reload
   somehow finds nothing, `dedup_service.MissingWinnerAfterRaceError` is raised rather than
   silently creating a second document.

No advisory lock, no transaction held open across the storage write, no content-addressed keys,
no shared physical objects between documents — each attempted creation still gets its own unique
object key from `app.storage.keys.generate_object_key()`.
"""

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.models.document import Document
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.services.documents.dedup_service import (
    MissingWinnerAfterRaceError,
    UploadDecision,
    UploadOutcome,
    compute_content_hash,
    decide_upload,
    is_content_hash_violation,
)
from app.storage.contract import FileStorage
from app.storage.keys import generate_object_key

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UploadResult:
    """The outcome of `upload_document()`: the document/job to report, and how they were reached.

    Populated identically whether the document was newly created or reused — the route never
    needs to infer the outcome from `ingestion_job.status`, and never receives a freshly generated
    `document.id` for a document that already existed under this content hash.
    """

    document: Document
    ingestion_job: IngestionJob
    outcome: UploadOutcome


async def _delete_best_effort(storage: FileStorage, key: str, *, reason: str) -> None:
    """Best-effort delete of `key`; a failure is logged, never raised, never hides the caller's
    real outcome."""
    try:
        await storage.delete(key)
    except Exception:
        logger.warning("Failed to clean up storage object %r after %s.", key, reason)


def _decision_to_result(decision: UploadDecision) -> UploadResult:
    """Convert a non-CREATED `UploadDecision` (which always carries a document/job) to a result."""
    assert decision.document is not None
    assert decision.ingestion_job is not None
    return UploadResult(
        document=decision.document, ingestion_job=decision.ingestion_job, outcome=decision.outcome
    )


async def upload_document(
    *,
    content: bytes,
    original_filename: str,
    content_type: str,
    storage: FileStorage,
    session: AsyncSession,
    settings: Settings | None = None,
) -> UploadResult:
    """Reuse an existing document with the same content, or save+persist a genuinely new one.

    Raises `dedup_service.DeletionActiveError`/`DeletionIncompleteError`/
    `DeletionInvariantViolationError` if a matching document's deletion state blocks reuse (no
    storage write happens in that case), and `dedup_service.MissingWinnerAfterRaceError` in the
    (should-be-unreachable) case where a content-hash race is reported but no winner can be
    reloaded. See the module docstring for the full fast-path/race-recovery sequence.
    """
    settings = settings or get_settings()
    content_hash = compute_content_hash(content)

    decision = await decide_upload(session, content_hash)
    if decision.outcome != UploadOutcome.CREATED:
        return _decision_to_result(decision)

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
        content_hash=content_hash,
    )
    session.add(document)

    job = IngestionJob(id=str(uuid.uuid4()), document_id=document.id, status=IngestionStatus.PENDING)
    session.add(job)

    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()

        if not is_content_hash_violation(exc):
            await _delete_best_effort(storage, key, reason="an unrelated DB integrity failure")
            raise

        await _delete_best_effort(storage, key, reason="losing a content-hash race")

        winner = await decide_upload(session, content_hash)
        if winner.outcome == UploadOutcome.CREATED:
            raise MissingWinnerAfterRaceError(content_hash) from exc
        return _decision_to_result(winner)
    except Exception:
        await _delete_best_effort(storage, key, reason="a DB commit failure")
        raise

    return UploadResult(document=document, ingestion_job=job, outcome=UploadOutcome.CREATED)
