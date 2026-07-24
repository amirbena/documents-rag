"""Tests for the correlation-ID contextvar primitive (Phase 2.10, app/core/correlation.py)."""

from app.core.correlation import (
    CORRELATION_ID_HEADER,
    correlation_headers,
    generate_correlation_id,
    get_correlation_id,
    set_correlation_id,
)


def test_generate_correlation_id_produces_distinct_values() -> None:
    first = generate_correlation_id()
    second = generate_correlation_id()
    assert first != second
    assert isinstance(first, str) and first


def test_set_and_get_correlation_id_round_trip() -> None:
    set_correlation_id("test-correlation-id-1")
    assert get_correlation_id() == "test-correlation-id-1"


def test_correlation_headers_carries_the_current_id() -> None:
    set_correlation_id("test-correlation-id-2")
    assert correlation_headers() == {CORRELATION_ID_HEADER: "test-correlation-id-2"}


def test_get_correlation_id_has_a_defined_default_outside_any_request() -> None:
    """Reading before any set_correlation_id() call in this context must never raise."""
    import contextvars

    def _read_in_fresh_context() -> str:
        return get_correlation_id()

    ctx = contextvars.Context()
    result = ctx.run(_read_in_fresh_context)
    assert isinstance(result, str) and result
