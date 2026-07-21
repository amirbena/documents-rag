"""Tests for structured logging setup (Phase 2.10, app/core/logging_config.py)."""

import json
import logging

from app.core.config import Settings
from app.core.correlation import set_correlation_id
from app.core.logging_config import configure_logging


def test_configure_logging_honors_log_level() -> None:
    configure_logging(Settings(LOG_LEVEL="WARNING"))
    assert logging.getLogger().level == logging.WARNING

    configure_logging(Settings(LOG_LEVEL="DEBUG"))
    assert logging.getLogger().level == logging.DEBUG


def test_configure_logging_is_idempotent_no_duplicate_handlers() -> None:
    configure_logging(Settings())
    managed_before = [
        h for h in logging.getLogger().handlers if getattr(h, "_documents_rag_managed", False)
    ]

    configure_logging(Settings())
    managed_after = [
        h for h in logging.getLogger().handlers if getattr(h, "_documents_rag_managed", False)
    ]

    assert len(managed_before) == 1
    assert len(managed_after) == 1


def test_log_record_is_emitted_as_json_with_correlation_id(capsys) -> None:
    configure_logging(Settings(LOG_LEVEL="INFO"))
    set_correlation_id("test-log-correlation-id")

    logging.getLogger("test.logger").info("something happened", extra={"event": "unit_test"})

    captured = capsys.readouterr()
    line = captured.err.strip().splitlines()[-1]
    payload = json.loads(line)

    assert payload["message"] == "something happened"
    assert payload["correlation_id"] == "test-log-correlation-id"
    assert payload["event"] == "unit_test"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test.logger"
