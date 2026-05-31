"""Tests for Ray job submission state in ``facefusion_job``."""

import asyncio
from collections.abc import Iterator

import pytest

import facefusion_job


@pytest.fixture(autouse=True)
def _clear_task_state() -> Iterator[None]:
    facefusion_job.tasks.clear()
    facefusion_job.task_logs.clear()
    yield
    facefusion_job.tasks.clear()
    facefusion_job.task_logs.clear()


def test_unknown_task_reports_not_found() -> None:
    status, logs = facefusion_job.get_task_status("missing")
    assert status == "not_found"
    assert logs == []


def test_known_task_returns_status_and_logs() -> None:
    facefusion_job.tasks["task-1"] = "processing"
    facefusion_job.task_logs["task-1"] = ["Starting task task-1"]
    status, logs = facefusion_job.get_task_status("task-1")
    assert status == "processing"
    assert logs == ["Starting task task-1"]


def test_task_status_returns_log_snapshot() -> None:
    facefusion_job.tasks["task-1"] = "processing"
    facefusion_job.task_logs["task-1"] = ["original"]

    _, logs = facefusion_job.get_task_status("task-1")
    logs.append("mutated")

    assert facefusion_job.task_logs["task-1"] == ["original"]


def test_run_facefusion_with_ray_job_initializes_state_and_schedules_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, str, str, str]] = []

    async def fake_run_subprocess(
        task_id: str,
        source_path: str,
        target_path: str,
        output_path: str,
        execution_provider: str,
    ) -> None:
        calls.append(
            (task_id, source_path, target_path, output_path, execution_provider)
        )

    monkeypatch.setattr(facefusion_job, "_run_subprocess", fake_run_subprocess)

    async def scenario() -> None:
        await facefusion_job.run_facefusion_with_ray_job(
            "task-1",
            "/tmp/source.jpg",
            "/tmp/target.jpg",
            "/tmp/output.jpg",
            execution_provider="cpu",
        )
        await asyncio.sleep(0)

    asyncio.run(scenario())

    assert facefusion_job.tasks["task-1"] == "processing"
    assert calls == [
        ("task-1", "/tmp/source.jpg", "/tmp/target.jpg", "/tmp/output.jpg", "cpu")
    ]
    assert facefusion_job.task_logs["task-1"] == [
        "Starting task task-1",
        "Source: /tmp/source.jpg",
        "Target: /tmp/target.jpg",
        "Output: /tmp/output.jpg",
    ]


class _FakeStdout:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    async def readline(self) -> bytes:
        if self._lines:
            return self._lines.pop(0)
        return b""


class _FakeStderr:
    def __init__(self, data: bytes) -> None:
        self._data = data

    async def read(self) -> bytes:
        return self._data


class _FakeProcess:
    def __init__(self, returncode: int, stdout: list[bytes], stderr: bytes) -> None:
        self.returncode = returncode
        self.stdout = _FakeStdout(stdout)
        self.stderr = _FakeStderr(stderr)

    async def wait(self) -> int:
        return self.returncode


def test_run_subprocess_marks_task_completed(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _FakeProcess(returncode=0, stdout=[b"queued\n", b"running\n"], stderr=b"")
    captured_args: tuple[str, ...] = ()

    async def fake_create_subprocess_exec(
        *args: str, **_kwargs: object
    ) -> _FakeProcess:
        nonlocal captured_args
        captured_args = args
        return process

    monkeypatch.setattr(
        facefusion_job.asyncio, "create_subprocess_exec", fake_create_subprocess_exec
    )
    facefusion_job.task_logs["task-1"] = []

    asyncio.run(
        facefusion_job._run_subprocess(
            "task-1", "/tmp/source.jpg", "/tmp/target.jpg", "/tmp/output.jpg", "cuda"
        )
    )

    assert captured_args[:6] == (
        "ray",
        "job",
        "submit",
        "--address=auto",
        "--",
        "python",
    )
    assert "queued" in facefusion_job.task_logs["task-1"]
    assert "running" in facefusion_job.task_logs["task-1"]
    assert facefusion_job.tasks["task-1"] == "completed"
    assert facefusion_job.task_logs["task-1"][-1] == "Task completed successfully"


def test_run_subprocess_records_stderr_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(returncode=7, stdout=[], stderr=b"bad input\nray failed\n")

    async def fake_create_subprocess_exec(
        *_args: str, **_kwargs: object
    ) -> _FakeProcess:
        return process

    monkeypatch.setattr(
        facefusion_job.asyncio, "create_subprocess_exec", fake_create_subprocess_exec
    )
    facefusion_job.task_logs["task-1"] = []

    asyncio.run(
        facefusion_job._run_subprocess(
            "task-1", "/tmp/source.jpg", "/tmp/target.jpg", "/tmp/output.jpg", "cuda"
        )
    )

    assert facefusion_job.tasks["task-1"] == "failed"
    assert "ERROR: bad input" in facefusion_job.task_logs["task-1"]
    assert "ERROR: ray failed" in facefusion_job.task_logs["task-1"]
    assert facefusion_job.task_logs["task-1"][-1] == "Task failed with return code 7"
