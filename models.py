"""Data models for the FaceFusion API.

This module defines Pydantic models used for request/response handling
in the FaceFusion service API endpoints.

Typical usage example:
    response = FaceFusionResponse(
        task_id="123",
        status="processing",
        output_path="/path/to/output.jpg"
    )
"""

from typing import List, Optional

from pydantic import BaseModel

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
    output_path: Optional[str] = None
    error: Optional[str] = None

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
    result: Optional[str] = None
    logs: List[str] = []

