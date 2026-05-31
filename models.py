"""Pydantic response models for the FaceFusion API.

The models intentionally mirror the public HTTP response payloads and avoid
runtime-only implementation details.

Typical usage example:
    response = FaceFusionResponse(
        task_id="123",
        status="processing",
        output_path="/path/to/output.jpg"
    )
"""

from pydantic import BaseModel, Field


class FaceFusionResponse(BaseModel):
    """Response model for face fusion operations.

    Attributes:
        task_id: Unique identifier for the operation.
        status: Current processing status.
        output_path: Path to result file when completed.
        error: Error message if failed.

    Example:
        >>> response = FaceFusionResponse(
        ...     task_id="123",
        ...     status="completed",
        ...     output_path="/outputs/result.jpg"
        ... )
    """

    task_id: str
    status: str
    output_path: str | None = None
    error: str | None = None


class TaskStatus(BaseModel):
    """Model representing task status information.

    Attributes:
        task_id: Unique identifier for the task.
        status: Current processing status.
        result: Path to output file if completed.
        logs: Processing log messages.

    Example:
        >>> status = TaskStatus(
        ...     task_id="123",
        ...     status="processing",
        ...     logs=["Starting processing", "Step 1 complete"]
        ... )
    """

    task_id: str
    status: str
    result: str | None = None
    logs: list[str] = Field(default_factory=list)
