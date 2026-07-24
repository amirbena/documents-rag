"""Tests for app.storage.factory.create_file_storage: provider selection and configuration errors."""

import pytest

from app.core.config import Settings
from app.storage.factory import create_file_storage
from app.storage.local_storage import LocalFileStorage
from app.storage.minio_storage import MinioFileStorage


def test_local_provider_selection() -> None:
    """FILE_STORAGE_PROVIDER=local should resolve to LocalFileStorage."""
    settings = Settings(FILE_STORAGE_PROVIDER="local", LOCAL_STORAGE_ROOT="storage/documents")

    storage = create_file_storage(settings)

    assert isinstance(storage, LocalFileStorage)


def test_minio_provider_selection() -> None:
    """FILE_STORAGE_PROVIDER=minio, with required MinIO settings, should resolve to MinioFileStorage."""
    settings = Settings(
        FILE_STORAGE_PROVIDER="minio",
        MINIO_ENDPOINT="localhost:9000",
        MINIO_ACCESS_KEY="key",
        MINIO_SECRET_KEY="secret",
        MINIO_BUCKET="documents",
    )

    storage = create_file_storage(settings)

    assert isinstance(storage, MinioFileStorage)


def test_minio_provider_requires_endpoint_and_bucket() -> None:
    """Selecting minio without MINIO_ENDPOINT/MINIO_BUCKET must fail clearly, not construct a client.

    As of Phase 2.10, this is caught at Settings construction (fail-fast config validation) rather
    than only when create_file_storage() later tries to build a MinioFileStorage — see
    Settings._validate_minio_configuration_complete(). The runtime StorageConfigurationError in
    MinioFileStorage.__init__ remains as defense in depth for any caller that bypasses Settings
    validation, but is no longer reachable via normal Settings construction.
    """
    with pytest.raises(ValueError, match="FILE_STORAGE_PROVIDER=minio requires"):
        Settings(FILE_STORAGE_PROVIDER="minio")


def test_local_settings_not_required_in_minio_mode() -> None:
    """Selecting minio must not require any LOCAL_STORAGE_ROOT customization — the default is fine."""
    settings = Settings(
        FILE_STORAGE_PROVIDER="minio",
        MINIO_ENDPOINT="localhost:9000",
        MINIO_ACCESS_KEY="key",
        MINIO_SECRET_KEY="secret",
        MINIO_BUCKET="documents",
    )

    # Must not raise despite LOCAL_STORAGE_ROOT being left at its default.
    storage = create_file_storage(settings)
    assert isinstance(storage, MinioFileStorage)


def test_minio_settings_not_required_in_local_mode() -> None:
    """Selecting local must not require any MINIO_* setting to be populated."""
    settings = Settings(FILE_STORAGE_PROVIDER="local")

    storage = create_file_storage(settings)
    assert isinstance(storage, LocalFileStorage)


def test_invalid_provider_is_rejected_by_settings_validation() -> None:
    """An unrecognized FILE_STORAGE_PROVIDER value must fail Settings construction clearly."""
    with pytest.raises(ValueError, match="FILE_STORAGE_PROVIDER"):
        Settings(FILE_STORAGE_PROVIDER="s3")
