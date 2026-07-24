"""Tests for scripts/process_pending_document_deletions.py's bounded batch loop and its
cooperative SIGINT/SIGTERM stop behavior (Phase 2.10).

Monkeypatches the script module's own imported names to fakes — no real Postgres/Qdrant/object
storage is used, and `scripts._shutdown.install_stop_signal_handlers` is replaced with a fake
context manager yielding a controllable stop double, so these tests never depend on real OS
signal delivery or timing.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from scripts import process_pending_document_deletions as script


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


def _install_fake_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(script, "get_vector_store", lambda settings: object())
    monkeypatch.setattr(script, "create_file_storage", lambda: object())


def _install_fake_worker(monkeypatch: pytest.MonkeyPatch, process_next_job) -> None:
    """process_next_job: an async callable(session) -> job|None, called on each loop iteration."""

    class _FakeWorker:
        def __init__(self, vector_store: object, file_storage: object) -> None:
            pass

        async def process_next_job(self, session: object):
            return await process_next_job(session)

    monkeypatch.setattr(script, "DocumentDeletionWorker", _FakeWorker)


def _job_sequence(jobs: list) -> tuple:
    """Returns (process_next_job, calls) — pops one job off `jobs` per call, None once exhausted."""
    calls = {"count": 0}
    remaining = list(jobs)

    async def _process_next_job(session: object):
        calls["count"] += 1
        return remaining.pop(0) if remaining else None

    return _process_next_job, calls


class _FakeStop:
    """Controllable stand-in for scripts._shutdown.StopRequested."""

    def __init__(self, *, already_stopped: bool = False) -> None:
        self._requested = already_stopped
        self.signal_name = "SIGTERM" if already_stopped else None

    def __bool__(self) -> bool:
        return self._requested

    def set(self, signal_name: str) -> None:
        self._requested = True
        self.signal_name = signal_name


def _install_fake_stop(monkeypatch: pytest.MonkeyPatch, stop: _FakeStop) -> None:
    @contextmanager
    def _fake_install_stop_signal_handlers():
        yield stop

    monkeypatch.setattr(script, "install_stop_signal_handlers", _fake_install_stop_signal_handlers)


async def test_stop_before_the_loop_prevents_any_claim(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _install_fake_dependencies(monkeypatch)
    _install_fake_session_factory(monkeypatch)
    _install_fake_stop(monkeypatch, _FakeStop(already_stopped=True))
    process_next_job, calls = _job_sequence(
        [SimpleNamespace(id="job-1", document_id="doc-1", status="completed")]
    )
    _install_fake_worker(monkeypatch, process_next_job)

    exit_code = await script.main()

    assert exit_code == 0
    assert calls["count"] == 0
    assert "Processed 0 document deletion job(s)" in capsys.readouterr().out


async def test_stop_during_processing_allows_the_current_unit_then_stops_before_the_next_claim(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A signal arriving while job 1 is in flight must not cut job 1 short — it completes, and
    only the *next* claim is prevented."""
    _install_fake_dependencies(monkeypatch)
    _install_fake_session_factory(monkeypatch)
    stop = _FakeStop(already_stopped=False)
    _install_fake_stop(monkeypatch, stop)

    calls = {"count": 0}

    async def _process_next_job(session: object):
        calls["count"] += 1
        if calls["count"] == 1:
            stop.set("SIGTERM")  # simulates a signal arriving mid-processing of job 1
            return SimpleNamespace(id="job-1", document_id="doc-1", status="completed")
        raise AssertionError("must not claim a second job once stop was requested")

    _install_fake_worker(monkeypatch, _process_next_job)

    exit_code = await script.main()

    assert exit_code == 0
    assert calls["count"] == 1
    assert "Processed 1 document deletion job(s)" in capsys.readouterr().out


async def test_normal_bounded_processing_is_unchanged(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """With no signal ever received, the loop still drains until no job remains — existing
    behavior is preserved exactly."""
    _install_fake_dependencies(monkeypatch)
    _install_fake_session_factory(monkeypatch)
    _install_fake_stop(monkeypatch, _FakeStop(already_stopped=False))
    process_next_job, calls = _job_sequence(
        [
            SimpleNamespace(id="job-1", document_id="doc-1", status="completed"),
            SimpleNamespace(id="job-2", document_id="doc-2", status="completed"),
            SimpleNamespace(id="job-3", document_id="doc-3", status="partially_failed"),
        ]
    )
    _install_fake_worker(monkeypatch, process_next_job)

    exit_code = await script.main()

    assert exit_code == 0
    assert calls["count"] == 4  # 3 jobs processed, 4th call returns None and ends the loop
    assert "Processed 3 document deletion job(s)" in capsys.readouterr().out


async def test_genuine_worker_exception_propagates_uncaught(monkeypatch: pytest.MonkeyPatch) -> None:
    """This script has never wrapped process_next_job in a try/except — an unexpected failure
    must still propagate out of main() exactly as it did before this change."""
    _install_fake_dependencies(monkeypatch)
    _install_fake_session_factory(monkeypatch)
    _install_fake_stop(monkeypatch, _FakeStop(already_stopped=False))

    async def _process_next_job(session: object):
        raise RuntimeError("qdrant unreachable")

    _install_fake_worker(monkeypatch, _process_next_job)

    with pytest.raises(RuntimeError, match="qdrant unreachable"):
        await script.main()


async def test_session_context_manager_is_always_exited(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_dependencies(monkeypatch)
    tracker = _install_fake_session_factory(monkeypatch)
    _install_fake_stop(monkeypatch, _FakeStop(already_stopped=False))
    process_next_job, _ = _job_sequence([])
    _install_fake_worker(monkeypatch, process_next_job)

    await script.main()

    assert tracker["entered"] == 1
    assert tracker["exited"] == 1
