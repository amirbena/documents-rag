"""Tests for CorrelationIdMiddleware (Phase 2.10) via the real, fully-wired app.

Uses GET /health (no dependency I/O, no DB session needed) purely as a vehicle to exercise the
middleware through a real request/response round trip.
"""

import re

from fastapi.testclient import TestClient

from app.core.correlation import CORRELATION_ID_HEADER
from app.main import app

client = TestClient(app)

_UUID4_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", re.IGNORECASE
)


def test_absent_correlation_header_is_generated_and_echoed() -> None:
    response = client.get("/health")

    assert CORRELATION_ID_HEADER in response.headers
    assert _UUID4_PATTERN.match(response.headers[CORRELATION_ID_HEADER])


def test_present_correlation_header_is_passed_through_unchanged() -> None:
    incoming = "operator-supplied-correlation-id-123"

    response = client.get("/health", headers={CORRELATION_ID_HEADER: incoming})

    assert response.headers[CORRELATION_ID_HEADER] == incoming


def test_two_requests_receive_distinct_generated_correlation_ids() -> None:
    first = client.get("/health").headers[CORRELATION_ID_HEADER]
    second = client.get("/health").headers[CORRELATION_ID_HEADER]

    assert first != second
