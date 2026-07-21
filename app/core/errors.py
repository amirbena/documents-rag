"""Backend error-category base classes (Phase 2.10).

This is additive, not a rewrite: the ~7 existing exception hierarchies in this codebase (storage
errors, provider errors, RAG/embedding/prompt errors, documents/dedup errors, indexing/reindex
errors, reconciliation/audit errors) are NOT reparented under `AppError` — each already has its
own, correct, route-level mapping via per-route outcome tables (see e.g.
`app/api/v1/routes/documents.py`'s `_RETRY_OUTCOME_ERRORS`). Reparenting them would risk changing
`isinstance` semantics those routes rely on, for no real benefit.

`AppError` and its subclasses exist for NEW code (the lifespan/config/retry work in this phase)
that needs a category to raise, and as a documented, permanent *fallback net* — see
`app/core/exception_handlers.py` — for any exception that reaches a route without having already
been translated to an `HTTPException` by that route's own (more specific, and always-checked-first)
outcome mapping. Existing exception modules get a one-line comment noting which category they
correspond to, purely for documentation — they are not modified otherwise.
"""

from fastapi import status


class AppError(Exception):
    """Base for every category below. Carries the HTTP status the fallback handler maps it to."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ConfigurationError(AppError):
    """Invalid or incomplete application configuration (e.g. Settings validation)."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "configuration_error"


class ValidationError(AppError):
    """A request or input value failed validation."""

    status_code = status.HTTP_400_BAD_REQUEST
    code = "validation_error"


class NotFoundError(AppError):
    """A referenced resource does not exist."""

    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"


class ConflictError(AppError):
    """The requested operation conflicts with the resource's current lifecycle state."""

    status_code = status.HTTP_409_CONFLICT
    code = "conflict"


class ProviderError(AppError):
    """An external provider (embedding, LLM, vector store, object storage) call failed."""

    status_code = status.HTTP_502_BAD_GATEWAY
    code = "provider_error"


class OperationTimeoutError(AppError):
    """An operation exceeded its configured timeout."""

    status_code = status.HTTP_504_GATEWAY_TIMEOUT
    code = "timeout"


class LifecycleError(AppError):
    """An application lifecycle operation (startup, shutdown) failed."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "lifecycle_error"


class InternalError(AppError):
    """An unexpected internal failure with no more specific category."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "internal_error"
