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


def _unreachable_minio_settings() -> Settings:
    """A structurally complete but genuinely unreachable MinIO configuration.

    As of Phase 2.10, an *incomplete* MinIO configuration (missing MINIO_ENDPOINT/MINIO_BUCKET/
    etc.) is rejected at Settings construction itself — see test_config.py's coverage of
    Settings._validate_minio_configuration_complete(). This helper instead exercises the
    connectivity-failure path check_file_storage() must still handle gracefully: complete,
    well-formed settings pointing at a host nothing is listening on.
    """
    return Settings(
        FILE_STORAGE_PROVIDER="minio",
        MINIO_ENDPOINT="localhost:1",
        MINIO_ACCESS_KEY="unreachable-key",
        MINIO_SECRET_KEY="unreachable-secret",
        MINIO_BUCKET="unreachable-bucket",
    )


async def test_minio_misconfiguration_reports_error_not_raise() -> None:
    """An unreachable MinIO endpoint must surface as an 'error' check result, never raise."""
    settings = _unreachable_minio_settings()

    result = await check_file_storage(settings)

    assert result.status == "error"
    assert result.required is True


async def test_detail_never_exposes_internal_information(tmp_path: Path) -> None:
    """A failing check's detail must be a fixed, generic message — no path/credential/stack trace."""
    settings = _unreachable_minio_settings()

    result = await check_file_storage(settings)

    assert result.detail == "File storage is unavailable."
