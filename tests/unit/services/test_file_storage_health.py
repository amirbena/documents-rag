"""Unit tests for app.services.platform_health.check_file_storage — no real MinIO, no Docker."""

from pathlib import Path

from app.core.config import Settings
from app.services.platform_health import check_file_storage


async def test_local_storage_ready(tmp_path: Path) -> None:
    """A writable local storage root should report status 'ok'."""
    settings = Settings(FILE_STORAGE_PROVIDER="local", LOCAL_STORAGE_ROOT=str(tmp_path / "docs"))

    result = await check_file_storage(settings)

    assert result.name == "file_storage"
    assert result.status == "ok"
    assert result.required is True


async def test_minio_misconfiguration_reports_error_not_raise() -> None:
    """An unusable MinIO configuration must surface as an 'error' check result, never raise."""
    settings = Settings(FILE_STORAGE_PROVIDER="minio")  # missing MINIO_ENDPOINT/MINIO_BUCKET

    result = await check_file_storage(settings)

    assert result.status == "error"
    assert result.required is True


async def test_detail_never_exposes_internal_information(tmp_path: Path) -> None:
    """A failing check's detail must be a fixed, generic message — no path/credential/stack trace."""
    settings = Settings(FILE_STORAGE_PROVIDER="minio")

    result = await check_file_storage(settings)

    assert result.detail == "File storage is unavailable."
