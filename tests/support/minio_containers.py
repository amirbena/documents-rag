"""Shared Testcontainers-based ephemeral MinIO container startup, reused by both the integration
and E2E backend suites.

Both `tests/integration/conftest.py` and `tests/e2e/backend/conftest.py` need to start the exact
same ephemeral MinIO container (dynamic port, test-only credentials, no persistent volume) but
belong to separate pytest packages with independently-scoped fixtures — this module holds the one
piece of container-startup logic both `conftest.py` files call, instead of duplicating the
`DockerContainer` setup verbatim in each.
"""

from collections.abc import Iterator

from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import HttpWaitStrategy

MINIO_TEST_ROOT_USER = "minioadmin-test"
MINIO_TEST_ROOT_PASSWORD = "minioadmin-test-secret"


def start_minio_container() -> DockerContainer:
    """Start and return one ephemeral MinIO container with test-only credentials, dynamic port."""
    container = (
        DockerContainer("minio/minio:latest")
        .with_exposed_ports(9000)
        .with_env("MINIO_ROOT_USER", MINIO_TEST_ROOT_USER)
        .with_env("MINIO_ROOT_PASSWORD", MINIO_TEST_ROOT_PASSWORD)
        .with_command("server /data")
    )
    container.waiting_for(HttpWaitStrategy(9000, "/minio/health/live"))
    container.start()
    return container


def minio_container_session() -> Iterator[DockerContainer]:
    """Yield one ephemeral MinIO container, stopping it on teardown — for a session-scoped fixture."""
    container = start_minio_container()
    try:
        yield container
    finally:
        container.stop()


def minio_endpoint(container: DockerContainer) -> str:
    """Return the dynamically-assigned host:port endpoint for a running MinIO container."""
    host = container.get_container_host_ip()
    port = container.get_exposed_port(9000)
    return f"{host}:{port}"
