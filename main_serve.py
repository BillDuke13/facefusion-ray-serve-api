import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import ray
from fastapi import FastAPI, File, HTTPException, UploadFile
from ray import serve

from config import OUTPUT_DIR, UPLOAD_DIR, RAY_ADDRESS
from facefusion_actor import FaceFusionActor
from models import FaceFusionResponse, TaskStatus


class FileManager:
    """Handles file operations for the FaceFusion service."""

    @staticmethod
    async def save_upload_file(upload_file: UploadFile, uid: str) -> Path:
        """Saves an uploaded file to the upload directory.

        Args:
            upload_file: The uploaded file object.
            uid: Unique identifier for the file.

        Returns:
            Path object pointing to the saved file.
        """
        file_path = UPLOAD_DIR / f"{uid}_{upload_file.filename}"
        with open(file_path, "wb") as buffer:
            content = await upload_file.read()
            buffer.write(content)
        return file_path

    @staticmethod
    def cleanup_old_files(cutoff_days: int = 5) -> None:
        """Removes files older than the specified cutoff period.

        Args:
            cutoff_days: Number of days after which files should be removed.
        """
        cutoff = datetime.now() - timedelta(days=cutoff_days)
        for file_path in UPLOAD_DIR.iterdir():
            if (file_path.is_file() and 
                datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff):
                # TODO(developer): Implement S3 upload before deletion
                file_path.unlink()


app = FastAPI(title="FaceFusion Service")
face_fusion_actor = FaceFusionActor.remote()


@app.post("/swap", response_model=FaceFusionResponse)
async def face_swap(
    source_image: UploadFile = File(...),
    target_image: UploadFile = File(...)
) -> FaceFusionResponse:
    """Processes a face swap request.

    Args:
        source_image: Source face image.
        target_image: Target image for face swapping.

    Returns:
        FaceFusionResponse containing task ID and status.

    Raises:
        HTTPException: If file processing fails.
    """
    task_id = str(uuid.uuid4())
    
    try:
        source_path = await FileManager.save_upload_file(
            source_image, f"{task_id}_source")
        target_path = await FileManager.save_upload_file(
            target_image, f"{task_id}_target")
        
        extension = Path(target_image.filename).suffix
        output_path = OUTPUT_DIR / f"{task_id}_output{extension}"

        await face_fusion_actor.process_face_fusion.remote(
            task_id,
            str(source_path),
            str(target_path),
            str(output_path)
        )

        return FaceFusionResponse(
            task_id=task_id,
            status="processing",
            output_path=str(output_path)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{task_id}", response_model=TaskStatus)
async def get_status(task_id: str) -> TaskStatus:
    """Retrieves the status of a face fusion task.

    Args:
        task_id: The unique identifier of the task.

    Returns:
        TaskStatus containing the current status and result if completed.
    """
    status = await ray.get(face_fusion_actor.get_task_status.remote(task_id))
    return TaskStatus(
        task_id=task_id,
        status=status,
        result=str(OUTPUT_DIR / f"{task_id}_output.jpg") if status == "completed" else None
    )


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Returns the health status of the service."""
    return {"status": "ok"}


@app.get("/stats")
async def service_stats() -> Dict[str, int]:
    """Returns service statistics including upload directory size."""
    size_bytes = sum(f.stat().st_size for f in UPLOAD_DIR.glob("*") if f.is_file())
    return {"upload_dir_size": size_bytes}


@serve.deployment(
    num_replicas=1,
    # ray_actor_options={
    #     "num_cpus": 8,
    #     "num_gpus": 0.95,
    #     "memory": 16 * 1024 * 1024 * 1024
    # },
    max_ongoing_requests=4,
    health_check_period_s=30,
    health_check_timeout_s=60,
    graceful_shutdown_wait_loop_s=120,
    graceful_shutdown_timeout_s=60
)
@serve.ingress(app)
class FaceFusionService:
    """Main service class for face fusion processing."""

    def __init__(self):
        """Initializes the service and starts the cleanup thread."""
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, 
            daemon=True
        )
        self._cleanup_thread.start()

    def _cleanup_loop(self) -> None:
        """Continuous loop for periodic cleanup of old files."""
        while True:
            FileManager.cleanup_old_files()
            time.sleep(86400)  # Cleanup once a day


facefusion_service = FaceFusionService.bind()

if __name__ == "__main__":
    ray.init(address=RAY_ADDRESS)
    serve.start(
        http_options={
            "host": "0.0.0.0",
            "port": 9999,
            "location": "EveryNode"
        }
    )
    serve.run(
        facefusion_service,
        route_prefix="/v1/model/facefusion",
        name="facefusion_service"
    )