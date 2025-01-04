"""FastAPI and Ray Serve implementation of the FaceFusion service.

This module provides the main service implementation combining FastAPI for HTTP
handling with Ray Serve for distributed processing. It includes file management,
task processing, and status tracking functionality.
"""

import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import logging
from logging.handlers import RotatingFileHandler
import sys

import ray
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from ray import serve
from ray.serve.deployment import Deployment

from config import OUTPUT_DIR, UPLOAD_DIR, RAY_ADDRESS
from facefusion_actor import FaceFusionActor
from models import FaceFusionResponse, TaskStatus

def setup_logging() -> logging.Logger:
    """Configures application-wide logging with file and console handlers.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    file_handler = RotatingFileHandler(
        log_dir / "facefusion_service.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging()

class FileManager:
    """Manages file operations for uploaded content and results.
    
    Provides static methods for saving uploaded files and cleaning up old files
    to prevent disk space issues.
    """

    @staticmethod
    async def save_upload_file(upload_file: UploadFile, uid: str) -> Path:
        """Saves an uploaded file with a unique identifier.

        Args:
            upload_file: FastAPI UploadFile object.
            uid: Unique identifier to prepend to filename.

        Returns:
            Path: Path where the file was saved.

        Raises:
            IOError: If file cannot be written to disk.
        """
        try:
            file_path = UPLOAD_DIR / f"{uid}_{upload_file.filename}"
            logger.info(f"Saving uploaded file to: {file_path}")

            with open(file_path, "wb") as buffer:
                content = await upload_file.read()
                buffer.write(content)

            logger.debug(f"File saved successfully: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Error saving uploaded file: {str(e)}", exc_info=True)
            raise

    @staticmethod
    def cleanup_old_files(cutoff_days: int = 5) -> None:
        """Removes files older than 'cutoff_days' in UPLOAD_DIR."""
        try:
            logger.info(f"Starting cleanup of files older than {cutoff_days} days")
            cutoff = datetime.now() - timedelta(days=cutoff_days)

            for file_path in UPLOAD_DIR.iterdir():
                try:
                    if file_path.is_file() and datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff:
                        logger.debug(f"Removing old file: {file_path}")
                        file_path.unlink()
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")

            logger.info("File cleanup completed")

        except Exception as e:
            logger.error(f"Error during file cleanup: {str(e)}", exc_info=True)

app = FastAPI(title="FaceFusion Service")

@serve.deployment(
    num_replicas=1,
    health_check_period_s=30,
    health_check_timeout_s=60,
    graceful_shutdown_wait_loop_s=120,
    graceful_shutdown_timeout_s=60
)
@serve.ingress(app)
class FaceFusionService:
    """Ray Serve deployment for the FaceFusion service.

    Combines FastAPI routing with Ray actor management to provide a scalable
    face fusion service. Includes health checking and file cleanup functionality.

    Attributes:
        face_fusion_actor: Reference to the Ray actor handling face fusion tasks.
    """

    def __init__(self):
        """Initializes the service with a FaceFusionActor and cleanup thread."""
        logger.info("Initializing FaceFusionService")

        try:
            self.face_fusion_actor = FaceFusionActor.remote()
            logger.info("FaceFusionActor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FaceFusionActor: {str(e)}", exc_info=True)
            raise

        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="CleanupThread"
        )
        self._cleanup_thread.start()
        logger.info("Cleanup thread started")

    def _cleanup_loop(self) -> None:
        """Background task for periodic cleanup."""
        while True:
            try:
                logger.info("Starting scheduled cleanup")
                FileManager.cleanup_old_files()
                time.sleep(86400)  # cleanup once a day
            except Exception as e:
                logger.error(f"Error in cleanup loop: {str(e)}", exc_info=True)
                time.sleep(3600)  # Wait an hour before retrying after error

    @app.post("/swap", response_model=FaceFusionResponse)
    async def face_swap(
        self,
        source_image: UploadFile = File(...),
        target_image: UploadFile = File(...)
    ) -> FaceFusionResponse:
        """Processes a face swap request with the provided images.

        Args:
            source_image: Source face image file.
            target_image: Target image/video file.

        Returns:
            FaceFusionResponse: Contains task ID and initial status.

        Raises:
            HTTPException: If file processing fails.
        """
        task_id = str(uuid.uuid4())
        logger.info(f"Starting face swap task: {task_id}")

        try:
            source_path = await FileManager.save_upload_file(
                source_image, f"{task_id}_source")
            target_path = await FileManager.save_upload_file(
                target_image, f"{task_id}_target")

            logger.debug(f"Files saved - Source: {source_path}, Target: {target_path}")

            extension = Path(target_image.filename).suffix
            output_path = OUTPUT_DIR / f"{task_id}_output{extension}"

            logger.debug(f"Submitting task to FaceFusionActor")
            actor = FaceFusionActor.remote()  # create a new actor each request
            actor.process_face_fusion.remote(
                task_id,
                str(source_path),
                str(target_path),
                str(output_path)
            )

            logger.info(f"Task {task_id} submitted successfully")
            return FaceFusionResponse(
                task_id=task_id,
                status="processing",
                output_path=str(output_path)
            )

        except Exception as e:
            logger.error(f"Error processing face swap task {task_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/status/{task_id}", response_model=TaskStatus)
    async def get_status(self, task_id: str) -> TaskStatus:
        """Retrieve the status of a face fusion task."""
        try:
            logger.debug(f"Checking status for task: {task_id}")
            status = await ray.get(self.face_fusion_actor.get_task_status.remote(task_id))

            response = TaskStatus(
                task_id=task_id,
                status=status,
                result=str(OUTPUT_DIR / f"{task_id}_output.jpg") if status == "completed" else None
            )
            logger.debug(f"Task {task_id} status: {response}")
            return response

        except Exception as e:
            logger.error(f"Error getting status for task {task_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check(self) -> Dict[str, str]:
        """Check service health."""
        try:
            return {"status": "ok"}
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Service unhealthy")

    @app.get("/stats")
    async def service_stats(self) -> Dict[str, int]:
        """Get service statistics."""
        try:
            size_bytes = sum(f.stat().st_size for f in UPLOAD_DIR.glob("*") if f.is_file())
            return {"upload_dir_size": size_bytes}
        except Exception as e:
            logger.error(f"Error getting service stats: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

# Bind deployment
facefusion_service = FaceFusionService.bind()

if __name__ == "__main__":
    try:
        logger.info("Starting FaceFusion Service")

        Path("logs").mkdir(exist_ok=True)
        OUTPUT_DIR.mkdir(exist_ok=True)
        UPLOAD_DIR.mkdir(exist_ok=True)

        if RAY_ADDRESS:
            logger.info(f"Connecting to Ray cluster at {RAY_ADDRESS}")
            ray.init(address=RAY_ADDRESS)
        else:
            logger.info("Initializing Ray locally")
            ray.init()

        cluster_info = ray.cluster_resources()
        logger.info(f"Ray cluster resources: {cluster_info}")

        logger.info("Starting Ray Serve")
        serve.start(
            http_options={
                "host": "0.0.0.0",
                "port": 9999,
                "location": "EveryNode"
            }
        )

        logger.info("Deploying FaceFusion Service")
        serve.run(
            facefusion_service,
            route_prefix="/v1/model/facefusion",
            name="facefusion_service"
        )

        logger.info("Service deployment completed successfully")
        logger.info("Service is running at http://0.0.0.0:9999/v1/model/facefusion")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")

    except Exception as e:
        logger.critical(f"Fatal error during service startup: {str(e)}", exc_info=True)
        sys.exit(1)