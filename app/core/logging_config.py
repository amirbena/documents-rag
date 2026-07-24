"""Structured logging configuration (Phase 2.10) — stdlib `logging` only, no new dependency.

Wires the previously-dead `LOG_LEVEL` setting to something real: `configure_logging()` sets the
root logger's level and installs a JSON formatter on a single stream handler. Every log record
automatically carries the current request's correlation ID (via `app.core.correlation`), so no
call site needs to pass it explicitly. Idempotent — safe to call multiple times (e.g. once from
`app/main.py`'s lifespan, and again from a standalone script) without installing duplicate handlers.
"""

import json
import logging
from datetime import UTC, datetime
from typing import Any

from app.core.config import Settings
from app.core.correlation import get_correlation_id

_RESERVED_LOG_RECORD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__)


class _JsonFormatter(logging.Formatter):
    """Renders one log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": get_correlation_id(),
        }
        # Any caller-supplied `extra={...}` fields ride along verbatim (event, operation, module,
        # provider, document_id, job_id, collection_name, duration, retry attempt, outcome, error
        # category, etc.) — this formatter never invents or requires a fixed field set.
        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOG_RECORD_ATTRS and key not in payload:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(settings: Settings) -> None:
    """Configure the root logger with settings.log_level and a JSON stream handler.

    Idempotent: clears any handlers this function previously installed before adding a fresh one,
    so repeated calls (lifespan re-entry in tests, a script importing this) never duplicate log
    lines. Never touches handlers this function did not itself install.
    """
    root_logger = logging.getLogger()
    for existing in list(root_logger.handlers):
        if getattr(existing, "_documents_rag_managed", False):
            root_logger.removeHandler(existing)

    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    handler._documents_rag_managed = True  # type: ignore[attr-defined]
    root_logger.addHandler(handler)
    root_logger.setLevel(settings.log_level.upper())
