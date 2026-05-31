"""Tests for the Pydantic API models."""

from models import FaceFusionResponse, TaskStatus


def test_response_optional_fields_default_to_none() -> None:
    response = FaceFusionResponse(task_id="abc", status="processing")
    assert response.output_path is None
    assert response.error is None


def test_response_serialization_roundtrip() -> None:
    response = FaceFusionResponse(
        task_id="abc", status="completed", output_path="/outputs/result.jpg"
    )
    dumped = response.model_dump()
    assert dumped["task_id"] == "abc"
    assert dumped["output_path"] == "/outputs/result.jpg"


def test_task_status_logs_default_to_empty_list() -> None:
    status = TaskStatus(task_id="abc", status="processing")
    assert status.result is None
    assert status.logs == []


def test_task_status_logs_do_not_share_default_list() -> None:
    first = TaskStatus(task_id="first", status="processing")
    second = TaskStatus(task_id="second", status="processing")

    first.logs.append("only first")

    assert second.logs == []
