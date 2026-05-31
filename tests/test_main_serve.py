"""Tests for the FastAPI route handlers and file helpers."""

import asyncio
import os
import uuid
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

import main_serve


@pytest.fixture
def client() -> TestClient:
    return TestClient(main_serve.app)


def test_safe_filename_uses_basename() -> None:
    assert main_serve.FileManager.safe_filename("../portrait.jpg") == "portrait.jpg"


def test_safe_filename_rejects_missing_name() -> None:
    with pytest.raises(HTTPException) as exc_info:
        main_serve.FileManager.safe_filename(None)

    assert exc_info.value.status_code == 400


def test_save_upload_file_writes_sanitized_filename(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(main_serve, "UPLOAD_DIR", tmp_path)
    upload = UploadFile(filename="../portrait.jpg", file=BytesIO(b"image-bytes"))

    saved_path = asyncio.run(
        main_serve.FileManager.save_upload_file(upload, "task-source")
    )

    assert saved_path == tmp_path / "task-source_portrait.jpg"
    assert saved_path.read_bytes() == b"image-bytes"


def test_cleanup_old_files_deletes_only_expired_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(main_serve, "UPLOAD_DIR", tmp_path)
    old_file = tmp_path / "old.jpg"
    fresh_file = tmp_path / "fresh.jpg"
    nested_dir = tmp_path / "nested"
    old_file.write_text("old")
    fresh_file.write_text("fresh")
    nested_dir.mkdir()

    old_timestamp = (datetime.now() - timedelta(days=10)).timestamp()
    os.utime(old_file, (old_timestamp, old_timestamp))

    main_serve.FileManager.cleanup_old_files(cutoff_days=5)

    assert not old_file.exists()
    assert fresh_file.exists()
    assert nested_dir.exists()


def test_health_route(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_stats_route_counts_upload_file_bytes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(main_serve, "UPLOAD_DIR", tmp_path)
    (tmp_path / "one.bin").write_bytes(b"123")
    (tmp_path / "two.bin").write_bytes(b"45")
    (tmp_path / "nested").mkdir()

    response = client.get("/stats")

    assert response.status_code == 200
    assert response.json() == {"upload_dir_size": 5}


def test_status_route_returns_completed_output(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_path = tmp_path / "task-1_output.jpg"
    output_path.write_bytes(b"result")
    monkeypatch.setattr(main_serve, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        main_serve, "get_task_status", lambda task_id: ("completed", ["done"])
    )

    response = client.get("/status/task-1")

    assert response.status_code == 200
    assert response.json() == {
        "task_id": "task-1",
        "status": "completed",
        "result": str(output_path),
        "logs": ["done"],
    }


def test_swap_route_saves_uploads_and_schedules_job(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    uploads_dir = tmp_path / "uploads"
    outputs_dir = tmp_path / "outputs"
    monkeypatch.setattr(main_serve, "UPLOAD_DIR", uploads_dir)
    monkeypatch.setattr(main_serve, "OUTPUT_DIR", outputs_dir)
    monkeypatch.setattr(
        main_serve.uuid,
        "uuid4",
        lambda: uuid.UUID("12345678-1234-5678-1234-567812345678"),
    )
    calls: list[tuple[str, str, str, str]] = []

    async def fake_run_facefusion_with_ray_job(
        task_id: str, source_path: str, target_path: str, output_path: str
    ) -> None:
        calls.append((task_id, source_path, target_path, output_path))

    monkeypatch.setattr(
        main_serve,
        "run_facefusion_with_ray_job",
        fake_run_facefusion_with_ray_job,
    )

    response = client.post(
        "/swap",
        files={
            "source_image": ("../source.jpg", b"source", "image/jpeg"),
            "target_image": ("target.png", b"target", "image/png"),
        },
    )

    task_id = "12345678-1234-5678-1234-567812345678"
    source_path = uploads_dir / f"{task_id}_source_source.jpg"
    target_path = uploads_dir / f"{task_id}_target_target.png"
    output_path = outputs_dir / f"{task_id}_output.png"

    assert response.status_code == 200
    assert response.json() == {
        "task_id": task_id,
        "status": "processing",
        "output_path": str(output_path),
        "error": None,
    }
    assert source_path.read_bytes() == b"source"
    assert target_path.read_bytes() == b"target"
    assert calls == [(task_id, str(source_path), str(target_path), str(output_path))]
