"""Tests for scripts/process_pending_reindex_jobs.py's bounded, one-job-per-invocation contract.

Mirrors tests/unit/scripts/test_smoke_multilingual_real.py's style: monkeypatch the script
module's own imported names to fakes and call `main()` directly — no real Postgres/Qdrant/object
storage is used. Proves: the existing `ReindexWorker` is constructed and invoked exactly once per
invocation (never looped), a `NO_JOB`/`COMPLETED`/`FAILED`/`SKIPPED_DELETED` outcome all exit 0
(all are outcomes the worker itself already recorded, not script failures), an unexpected
exception exits 1 without leaking a raw stack trace, and the session context manager is always
exited (dependencies are not leaked).
"""

import pytest

from app.services.indexing.reindex_worker import ReindexWorkerOutcome, ReindexWorkerResult
from scripts import process_pending_reindex_jobs as script


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


def _install_fake_file_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(script, "create_file_storage", lambda settings: object())


def _install_fake_worker(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: ReindexWorkerResult | None = None,
    exc: Exception | None = None,
) -> dict:
    calls: dict = {"count": 0, "file_storage": None}

    class _FakeWorker:
        def __init__(self, file_storage: object) -> None:
            calls["file_storage"] = file_storage

        async def process_next_job(self, session: object, settings: object) -> ReindexWorkerResult:
            calls["count"] += 1
            if exc is not None:
                raise exc
            assert result is not None
            return result

    monkeypatch.setattr(script, "ReindexWorker", _FakeWorker)
    return calls


async def test_no_pending_job_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _install_fake_file_storage(monkeypatch)
    _install_fake_session_factory(monkeypatch)
    calls = _install_fake_worker(
        monkeypatch,
        result=ReindexWorkerResult(outcome=ReindexWorkerOutcome.NO_JOB, job_id=None, document_id=None),
    )

    exit_code = await script.main()

    assert exit_code == 0
    assert calls["count"] == 1
    assert "No pending re-index job" in capsys.readouterr().out


async def test_completed_job_exits_zero_and_reports_identity(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _install_fake_file_storage(monkeypatch)
    _install_fake_session_factory(monkeypatch)
    calls = _install_fake_worker(
        monkeypatch,
        result=ReindexWorkerResult(
            outcome=ReindexWorkerOutcome.COMPLETED, job_id="job-1", document_id="doc-1"
        ),
    )

    exit_code = await script.main()

    assert exit_code == 0
    assert calls["count"] == 1
    output = capsys.readouterr().out
    assert "job-1" in output
    assert "doc-1" in output
    assert "completed" in output


@pytest.mark.parametrize("outcome", [ReindexWorkerOutcome.FAILED, ReindexWorkerOutcome.SKIPPED_DELETED])
async def test_recorded_failure_outcomes_still_exit_zero(
    monkeypatch: pytest.MonkeyPatch, outcome: ReindexWorkerOutcome
) -> None:
    """A FAILED/SKIPPED_DELETED outcome is the worker correctly recording and persisting a real
    build failure — not a script-invocation failure — so the script itself still exits 0."""
    _install_fake_file_storage(monkeypatch)
    _install_fake_session_factory(monkeypatch)
    _install_fake_worker(
        monkeypatch, result=ReindexWorkerResult(outcome=outcome, job_id="job-1", document_id="doc-1")
    )

    exit_code = await script.main()

    assert exit_code == 0


async def test_unexpected_worker_exception_exits_nonzero_without_leaking_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _install_fake_file_storage(monkeypatch)
    _install_fake_session_factory(monkeypatch)
    calls = _install_fake_worker(monkeypatch, exc=RuntimeError("qdrant unreachable at internal-host:6333"))

    exit_code = await script.main()

    assert exit_code == 1
    assert calls["count"] == 1
    output = capsys.readouterr().out
    assert "internal-host" not in output
    assert "6333" not in output
    assert "Traceback" not in output
    assert "RuntimeError" not in output


async def test_process_next_job_is_called_at_most_once_no_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_file_storage(monkeypatch)
    _install_fake_session_factory(monkeypatch)
    calls = _install_fake_worker(
        monkeypatch,
        result=ReindexWorkerResult(
            outcome=ReindexWorkerOutcome.COMPLETED, job_id="job-1", document_id="doc-1"
        ),
    )

    await script.main()

    assert calls["count"] == 1


async def test_session_context_manager_is_always_exited(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_file_storage(monkeypatch)
    tracker = _install_fake_session_factory(monkeypatch)
    _install_fake_worker(
        monkeypatch,
        result=ReindexWorkerResult(outcome=ReindexWorkerOutcome.NO_JOB, job_id=None, document_id=None),
    )

    await script.main()

    assert tracker["entered"] == 1
    assert tracker["exited"] == 1


async def test_worker_is_constructed_with_the_created_file_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    monkeypatch.setattr(script, "create_file_storage", lambda settings: sentinel)
    _install_fake_session_factory(monkeypatch)
    calls = _install_fake_worker(
        monkeypatch,
        result=ReindexWorkerResult(outcome=ReindexWorkerOutcome.NO_JOB, job_id=None, document_id=None),
    )

    await script.main()

    assert calls["file_storage"] is sentinel
