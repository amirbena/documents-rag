"""Regression coverage for MinIO retry semantics against the real configured HTTP transport
(`app/storage/minio_storage.py`'s `urllib3.PoolManager`) — not a monkeypatched `Minio` client
method.

A tiny local HTTP server scripts a sequence of S3-style responses so a genuine urllib3
request/response round trip happens for every attempt; the retry loop under test is
`retry_async` (`app/core/retry.py`) via `MinioFileStorage._retry_call`, with the real `Minio` SDK
and its real `urllib3.PoolManager` (`retries=Retry(total=None, connect=0, read=0, redirect=1,
status=0, other=0)`) in the loop — this is what proves `retry_async` is the *only* layer retrying
a transient MinIO failure, not a monkeypatch of `storage._client.remove_object` that would never
exercise the PoolManager's own retry configuration at all. It also proves urllib3's own automatic
redirect-following — deliberately left enabled for exactly one hop — still works, since the
`minio` SDK has no redirect-following logic of its own (see `app/storage/minio_storage.py`'s
constructor comment).

`PROVIDER_RETRY_MAX_ATTEMPTS` is the TOTAL number of attempts, including the first (non-retry) one
— see `app/core/retry.py`'s `retry_async` docstring and
`docs/providers/README.md#timeout-and-retry-policy-phase-210`. Every assertion below counts real
requests against that total-attempts meaning, never against urllib3's own `Retry(...)` counters.
"""

import asyncio
import http.server
import threading
from collections.abc import Callable, Iterator

import pytest

from app.core.config import Settings
from app.storage.errors import StorageDeleteError, StorageUnavailableError
from app.storage.minio_storage import MinioFileStorage

_S3_ERROR_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    "<Error><Code>{code}</Code><Message>scripted error</Message>"
    "<Resource>/documents/some-key</Resource><RequestId>req-1</RequestId>"
    "<HostId>host-1</HostId></Error>"
)


class _ScriptedResponse:
    """One scripted response: an S3-error XML body (`code` set), a redirect (`location` set), or
    a plain empty-body response (neither set — e.g. a 204 success)."""

    def __init__(
        self, status: int, code: str | None = None, location: str | None = None
    ) -> None:
        self.status = status
        self.code = code
        self.location = location


class _ScriptedS3Server(http.server.HTTPServer):
    """Returns a pre-scripted sequence of S3-style responses to every request (the last scripted
    response repeats once the script is exhausted) and counts real requests received."""

    def __init__(self, script: list[_ScriptedResponse]) -> None:
        super().__init__(("127.0.0.1", 0), _ScriptedS3RequestHandler)
        self.script = script
        self.request_count = 0
        self.lock = threading.Lock()


class _ScriptedS3RequestHandler(http.server.BaseHTTPRequestHandler):
    server: _ScriptedS3Server

    def log_message(self, *args: object) -> None:  # silence default stderr access logging
        return

    def _respond(self) -> None:
        with self.server.lock:
            index = min(self.server.request_count, len(self.server.script) - 1)
            self.server.request_count += 1
            scripted = self.server.script[index]

        if scripted.location is not None:
            self.send_response(scripted.status)
            self.send_header("Location", scripted.location)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if scripted.code is None:
            self.send_response(scripted.status)
            self.end_headers()
            return

        body = _S3_ERROR_XML.format(code=scripted.code).encode()
        self.send_response(scripted.status)
        self.send_header("Content-Type", "application/xml")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_DELETE(self) -> None:
        self._respond()

    def do_GET(self) -> None:
        self._respond()

    def do_PUT(self) -> None:
        self._respond()

    def do_HEAD(self) -> None:
        self._respond()


ScriptedServerFactory = Callable[[list[_ScriptedResponse]], _ScriptedS3Server]


@pytest.fixture
def scripted_server() -> Iterator[ScriptedServerFactory]:
    servers: list[_ScriptedS3Server] = []

    def factory(script: list[_ScriptedResponse]) -> _ScriptedS3Server:
        server = _ScriptedS3Server(script)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append(server)
        return server

    yield factory

    for server in servers:
        server.shutdown()
        server.server_close()


def _settings(server: _ScriptedS3Server, **overrides: object) -> Settings:
    fields: dict[str, object] = {
        "FILE_STORAGE_PROVIDER": "minio",
        "MINIO_ENDPOINT": f"127.0.0.1:{server.server_port}",
        "MINIO_ACCESS_KEY": "key",
        "MINIO_SECRET_KEY": "secret",
        "MINIO_BUCKET": "documents",
        # A fixed region avoids the SDK's own separate GetBucketLocation request (Minio._get_region)
        # on the first call against a bucket — without it, that extra, unretried request would
        # consume a slot from this test's scripted response sequence before the operation under
        # test (remove_object) ever runs, decoupling "real request" from "retry_async attempt."
        "MINIO_REGION": "us-east-1",
        "PROVIDER_RETRY_MAX_ATTEMPTS": 3,
        "PROVIDER_RETRY_BASE_DELAY_SECONDS": 0.01,
        "PROVIDER_RETRY_MAX_DELAY_SECONDS": 0.02,
    }
    fields.update(overrides)
    return Settings(**fields)  # type: ignore[arg-type]


async def test_persistent_transient_failure_makes_no_more_real_requests_than_the_attempt_ceiling(
    scripted_server: ScriptedServerFactory,
) -> None:
    """A sustained transient 500/InternalError must exhaust at exactly PROVIDER_RETRY_MAX_ATTEMPTS
    (3) real requests — never urllib3's own (disabled) retry amplifying that further."""
    server = scripted_server([_ScriptedResponse(500, code="InternalError")])  # every request
    storage = MinioFileStorage(settings=_settings(server))

    with pytest.raises(StorageDeleteError):
        await storage.delete("some-key")

    assert server.request_count == 3


async def test_transient_failure_then_success_succeeds_at_the_second_of_three_attempts(
    scripted_server: ScriptedServerFactory,
) -> None:
    """One transient failure followed by success must succeed on the 2nd of 3 configured total
    attempts — exactly 2 real requests, not 1 (no retry at all) and not more than 2."""
    server = scripted_server([_ScriptedResponse(500, code="InternalError"), _ScriptedResponse(204)])
    storage = MinioFileStorage(settings=_settings(server))

    await storage.delete("some-key")  # must not raise

    assert server.request_count == 2


async def test_permanent_failure_is_not_retried(scripted_server: ScriptedServerFactory) -> None:
    """A permanent S3 error code (AccessDenied) must fail on the very first request — no retry."""
    server = scripted_server([_ScriptedResponse(403, code="AccessDenied")])
    storage = MinioFileStorage(settings=_settings(server))

    with pytest.raises(StorageDeleteError):
        await storage.delete("some-key")

    assert server.request_count == 1


async def test_one_redirect_is_followed_transparently_and_succeeds(
    scripted_server: ScriptedServerFactory,
) -> None:
    """A single legitimate S3-style redirect must be followed by urllib3 itself — transparently,
    with no application-level retry involved at all — and the operation succeeds. This is the
    behavior a bare `Retry(total=0)` would have broken (see app/storage/minio_storage.py's
    constructor comment): the SDK has no redirect-following logic of its own."""
    server = scripted_server(
        [_ScriptedResponse(307, location="/documents/some-key?redirected=1"), _ScriptedResponse(204)]
    )
    storage = MinioFileStorage(settings=_settings(server))

    await storage.delete("some-key")  # must not raise

    # Both the original and the redirected request land on this single server, so 2 real requests
    # are observed — but only 1 retry_async attempt (the redirect is resolved entirely inside
    # urllib3, before retry_async's classifier ever sees a result).
    assert server.request_count == 2


async def test_redirect_loop_fails_after_the_explicit_redirect_limit(
    scripted_server: ScriptedServerFactory,
) -> None:
    """A redirect that never resolves (a loop) must still fail — the explicit `redirect=1` allows
    exactly one hop, not unlimited hops — and, once it does fail, PROVIDER_RETRY_MAX_ATTEMPTS still
    bounds how many times retry_async retries the resulting MaxRetryError."""
    server = scripted_server([_ScriptedResponse(307, location="/documents/some-key")])  # loops forever
    storage = MinioFileStorage(settings=_settings(server))

    # A redirect loop is exhausted as a urllib3 MaxRetryError, which delete() maps to
    # StorageUnavailableError (its connection-level failure mapping) — never StorageDeleteError,
    # which is reserved for a body-parsed S3Error.
    with pytest.raises(StorageUnavailableError):
        await storage.delete("some-key")

    # Each of the 3 configured retry_async attempts follows exactly 1 redirect hop before its own
    # MaxRetryError (2 real requests per attempt) — 3 attempts x 2 requests = 6, never unbounded.
    assert server.request_count == 6


async def test_cancellation_is_propagated_and_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """asyncio.CancelledError raised mid-call must propagate immediately, unmodified, and must
    never trigger a retry attempt — retry_async's classification loop only ever catches
    `Exception`, never `BaseException`."""
    settings = Settings(
        FILE_STORAGE_PROVIDER="minio",
        MINIO_ENDPOINT="127.0.0.1:1",
        MINIO_ACCESS_KEY="key",
        MINIO_SECRET_KEY="secret",
        MINIO_BUCKET="documents",
        PROVIDER_RETRY_MAX_ATTEMPTS=3,
    )
    storage = MinioFileStorage(settings=settings)
    calls: list[int] = []

    def _raise_cancelled(*args: object, **kwargs: object) -> None:
        calls.append(1)
        raise asyncio.CancelledError()

    monkeypatch.setattr(storage._client, "remove_object", _raise_cancelled)

    with pytest.raises(asyncio.CancelledError):
        await storage.delete("some-key")

    assert len(calls) == 1
