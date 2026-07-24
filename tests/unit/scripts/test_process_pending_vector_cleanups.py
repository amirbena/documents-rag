"""Tests for scripts/process_pending_vector_cleanups.py's bounded, one-job-per-invocation contract.

Mirrors tests/unit/scripts/test_process_pending_reindex_jobs.py's style: monkeypatch the script
module's own imported names to fakes and call `main()` directly — no real Postgres/Qdrant is used.
Proves: the existing `process_next_vector_cleanup_job()` service function is invoked exactly once
per invocation (never looped), `NO_JOB`/`COMPLETED`/`FAILED` outcomes all exit 0 (the service
itself already recorded the failure — including the active-collection safety guard refusing to
delete — this is never hidden or reported as success by the script), an unexpected exception exits
1 without leaking a raw stack trace, and the session context manager is always exited.
"""

from contextlib import contextmanager

import pytest

from app.services.indexing.cleanup_job_service import VectorCleanupWorkerOutcome, VectorCleanupWorkerResult
from scripts import process_pending_vector_cleanups as script


class _FakeStop:
    """Controllable stand-in for scripts._shutdown.StopRequested."""

    def __init__(self, *, already_stopped: bool) -> None:
        self._requested = already_stopped
        self.signal_name = "SIGTERM" if already_stopped else None

    def __bool__(self) -> bool:
        return self._requested


def _install_fake_stop(monkeypatch: pytest.MonkeyPatch, *, already_stopped: bool) -> None:
    stop = _FakeStop(already_stopped=already_stopped)

    @contextmanager
    def _fake_install_stop_signal_handlers():
        yield stop

    monkeypatch.setattr(script, "install_stop_signal_handlers", _fake_install_stop_signal_handlers)


class _FakeSession:
    pass


class _FakeSessionContextManager:
    def __init__(self, tracker: dict) -> None:
        self._tracker = tracker

    async def __aenter__(self) -> _FakeSession:
        self._tracker["entered"] += 1
        return _FakeSession()

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        self._tracker["exited"] += 1
        return False


def _install_fake_session_factory(monkeypatch: pytest.MonkeyPatch) -> dict:
    tracker = {"entered": 0, "exited": 0}
    monkeypatch.setattr(script, "async_session_factory", lambda: _FakeSessionContextManager(tracker))
    return tracker


def _install_fake_vector_store(monkeypatch: pytest.MonkeyPatch) -> object:
    sentinel = object()
    monkeypatch.setattr(script, "get_vector_store", lambda settings: sentinel)
    return sentinel


def _install_fake_processor(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: VectorCleanupWorkerResult | None = None,
    exc: Exception | None = None,
) -> dict:
    calls: dict = {"count": 0, "args": None}

    async def _fake_process_next_vector_cleanup_job(session: object, vector_store: object):
        calls["count"] += 1
        calls["args"] = (session, vector_store)
        if exc is not None:
            raise exc
        assert result is not None
        return result

    monkeypatch.setattr(script, "process_next_vector_cleanup_job", _fake_process_next_vector_cleanup_job)
    return calls


async def test_no_eligible_job_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _install_fake_vector_store(monkeypatch)
    _install_fake_session_factory(monkeypatch)
    calls = _install_fake_processor(
        monkeypatch,
        result=VectorCleanupWorkerResult(
            outcome=VectorCleanupWorkerOutcome.NO_JOB, job_id=None, document_id=None, collection_name=None
        ),
    )

    exit_code = await script.main()

    assert exit_code == 0
    assert calls["count"] == 1
    assert "No pending or retry-eligible" in capsys.readouterr().out


async def test_completed_cleanup_exits_zero_and_reports_identity(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _install_fake_vector_store(monkeypatch)
    _install_fake_session_factory(monkeypatch)
    calls = _install_fake_processor(
        monkeypatch,
        result=VectorCleanupWorkerResult(
            outcome=VectorCleanupWorkerOutcome.COMPLETED,
            job_id="job-1",
            document_id="doc-1",
            collection_name="documents_v1",
        ),
    )

    exit_code = await script.main()

    assert exit_code == 0
    assert calls["count"] == 1
    output = capsys.readouterr().out
    assert "job-1" in output
    assert "doc-1" in output
    assert "documents_v1" in output
    assert "completed" in output


async def test_failed_cleanup_is_represented_not_hidden_and_still_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A FAILED outcome (e.g. the active-collection safety guard, or a real Qdrant failure) is the
    service correctly recording and persisting the failure — not a script-invocation failure — so
    the script exits 0, and the output must say FAILED, never claim success."""
    _install_fake_vector_store(monkeypatch)
    _install_fake_session_factory(monkeypatch)
    _install_fake_processor(
        monkeypatch,
        result=VectorCleanupWorkerResult(
            outcome=VectorCleanupWorkerOutcome.FAILED,
            job_id="job-1",
            document_id="doc-1",
            collection_name="documents_v1",
        ),
    )

    exit_code = await script.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "failed" in output
    assert "completed" not in output


async def test_unexpected_exception_exits_nonzero_without_leaking_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _install_fake_vector_store(monkeypatch)
    _install_fake_session_factory(monkeypatch)
    calls = _install_fake_processor(monkeypatch, exc=RuntimeError("qdrant unreachable at internal-host:6333"))

    exit_code = await script.main()

    assert exit_code == 1
    assert calls["count"] == 1
    output = capsys.readouterr().out
    assert "internal-host" not in output
    assert "6333" not in output
    assert "Traceback" not in output
    assert "RuntimeError" not in output


async def test_process_next_vector_cleanup_job_is_called_at_most_once_no_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_vector_store(monkeypatch)
    _install_fake_session_factory(monkeypatch)
    calls = _install_fake_processor(
        monkeypatch,
        result=VectorCleanupWorkerResult(
            outcome=VectorCleanupWorkerOutcome.COMPLETED,
            job_id="job-1",
            document_id="doc-1",
            collection_name="documents_v1",
        ),
    )

    await script.main()

    assert calls["count"] == 1


async def test_session_context_manager_is_always_exited(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_vector_store(monkeypatch)
    tracker = _install_fake_session_factory(monkeypatch)
    _install_fake_processor(
        monkeypatch,
        result=VectorCleanupWorkerResult(
            outcome=VectorCleanupWorkerOutcome.NO_JOB, job_id=None, document_id=None, collection_name=None
        ),
    )

    await script.main()

    assert tracker["entered"] == 1
    assert tracker["exited"] == 1


async def test_service_is_called_with_the_configured_vector_store(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = _install_fake_vector_store(monkeypatch)
    _install_fake_session_factory(monkeypatch)
    calls = _install_fake_processor(
        monkeypatch,
        result=VectorCleanupWorkerResult(
            outcome=VectorCleanupWorkerOutcome.NO_JOB, job_id=None, document_id=None, collection_name=None
        ),
    )

    await script.main()

    assert calls["args"][1] is sentinel


async def test_stop_requested_before_the_claim_prevents_any_processing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Phase 2.10: a SIGINT/SIGTERM received before this script's one claim must skip it entirely."""
    _install_fake_vector_store(monkeypatch)
    _install_fake_session_factory(monkeypatch)
    _install_fake_stop(monkeypatch, already_stopped=True)
    calls = _install_fake_processor(
        monkeypatch,
        result=VectorCleanupWorkerResult(
            outcome=VectorCleanupWorkerOutcome.COMPLETED,
            job_id="job-1",
            document_id="doc-1",
            collection_name="documents_v1",
        ),
    )

    exit_code = await script.main()

    assert exit_code == 0
    assert calls["count"] == 0
    assert "Stop requested" in capsys.readouterr().out
