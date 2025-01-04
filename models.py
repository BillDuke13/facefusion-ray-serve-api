from pydantic import BaseModel
from typing import Optional

class FaceFusionResponse(BaseModel):
    task_id: str
    status: str
    output_path: Optional[str] = None
    error: Optional[str] = None

class TaskStatus(BaseModel):
    task_id: str
    status: str
    result: Optional[str] = None

