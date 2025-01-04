"""Data models for the FaceFusion API.

This module defines the Pydantic models used for request/response handling
in the FaceFusion service.
"""

from pydantic import BaseModel
from typing import Optional

class FaceFusionResponse(BaseModel):
    """Response model for face fusion operations.

    Attributes:
        task_id: Unique identifier for the face fusion task.
        status: Current status of the task (processing/completed/failed).
        output_path: Path to the output file when task is completed.
        error: Error message if the task failed.
    """
    task_id: str
    status: str
    output_path: Optional[str] = None
    error: Optional[str] = None

class TaskStatus(BaseModel):
    """Model representing the current status of a face fusion task.
    
    Attributes:
        task_id: Unique identifier for the task.
        status: Current status of the task.
        result: Path to the result file if task is completed.
    """
    task_id: str
    status: str
    result: Optional[str] = None

