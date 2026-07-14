"""Focused OpenAPI schema assertions for POST /api/v1/documents (Phase 2.8.5 subtask 4).

Asserts the generated schema/documented responses, not a full-document snapshot — see
CLAUDE.md's OpenAPI-test guidance.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _upload_schema() -> dict:
    openapi = client.get("/openapi.json").json()
    return openapi["components"]["schemas"]["DocumentUploadResponse"]


def _upload_operation() -> dict:
    openapi = client.get("/openapi.json").json()
    return openapi["paths"]["/api/v1/documents"]["post"]


def test_outcome_field_documents_all_four_successful_values() -> None:
    schema = _upload_schema()
    openapi = client.get("/openapi.json").json()
    outcome_enum = openapi["components"]["schemas"]["DocumentUploadOutcome"]["enum"]

    assert "outcome" in schema["properties"]
    assert set(outcome_enum) == {"CREATED", "REUSED_ACTIVE", "REUSED_INDEXED", "REUSED_FAILED"}


def test_original_filename_is_required() -> None:
    schema = _upload_schema()

    assert "original_filename" in schema["properties"]
    assert "original_filename" in schema["required"]


def test_success_responses_document_both_200_and_202() -> None:
    operation = _upload_operation()

    assert "200" in operation["responses"]
    assert "202" in operation["responses"]


def test_409_conflict_is_documented() -> None:
    operation = _upload_operation()

    assert "409" in operation["responses"]


def test_response_schema_never_exposes_content_hash_or_storage_metadata() -> None:
    schema = _upload_schema()

    for field in ("content_hash", "storage_key", "storage_bucket", "storage_etag"):
        assert field not in schema["properties"]
