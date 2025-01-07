"""FastAPI and Ray Serve implementation of the FaceFusion service.

This module provides the main service implementation combining FastAPI for HTTP
handling with Ray Serve for distributed processing.

Typical usage example:
    if __name__ == "__main__":
        service = FaceFusionService()
        serve.run(service, route_prefix="/v1/model/facefusion")
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Any

import ray
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ray import serve

from config import (
    OUTPUT_DIR,
    UPLOAD_DIR,
    RAY_ADDRESS,
    SERVICE_HOST,
    SERVICE_PORT
)
from facefusion_job import run_facefusion_with_ray_job, get_task_status
from models import FaceFusionResponse, TaskStatus

# Constants
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "facefusion_service.log"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5
CLEANUP_INTERVAL = 86400  # 1 day in seconds
RETRY_INTERVAL = 3600    # 1 hour in seconds
DEFAULT_CUTOFF_DAYS = 5

# Setup logging
logger = logging.getLogger(__name__)

def setup_logging() -> None:
    """Configure application-wide logging with file and console handlers."""
    LOG_DIR.mkdir(exist_ok=True)
    
    formatter = logging.Formatter(LOG_FORMAT)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # File handler
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=MAX_LOG_SIZE,
        backupCount=LOG_BACKUP_COUNT
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

class FileManager:
    """File operations manager for the FaceFusion service.
    
    Handles file uploads and cleanup operations for the service.
    """

    @staticmethod
    async def save_upload_file(upload_file: UploadFile, uid: str) -> Path:
        """Save uploaded file with unique identifier.

        Args:
            upload_file: The uploaded file object.
            uid: Unique identifier for the file.

        Returns:
            Path to the saved file.

        Raises:
            HTTPException: If file saving fails.
        """
        try:
            file_path = UPLOAD_DIR / f"{uid}_{upload_file.filename}"
            content = await upload_file.read()
            
            file_path.write_bytes(content)
            logger.info(f"Saved file: {file_path}")
            
            return file_path
            
        except Exception as e:
            logger.error(f"File save error: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save file: {str(e)}"
            )

    @staticmethod
    def cleanup_old_files(cutoff_days: int = DEFAULT_CUTOFF_DAYS) -> None:
        """Remove files older than the cutoff period.

        Args:
            cutoff_days: Number of days after which files should be removed.
        """
        try:
            cutoff = datetime.now() - timedelta(days=cutoff_days)
            
            for file_path in UPLOAD_DIR.iterdir():
                if not file_path.is_file():
                    continue
                    
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if mtime < cutoff:
                    try:
                        file_path.unlink()
                        logger.debug(f"Removed old file: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to remove {file_path}: {e}")
                        
        except Exception as e:
            logger.error(f"Cleanup error: {e}", exc_info=True)

@serve.deployment(
    num_replicas=1,
    health_check_period_s=30,
    health_check_timeout_s=60,
    graceful_shutdown_wait_loop_s=120,
    graceful_shutdown_timeout_s=60
)
@serve.ingress(app := FastAPI(title="FaceFusion Service"))
class FaceFusionService:
    """Ray Serve deployment for the FaceFusion service."""

    def __init__(self) -> None:
        """Initialize service and start cleanup thread."""
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="CleanupThread"
        )
        self._cleanup_thread.start()
        logger.info("Service initialized")

    def _cleanup_loop(self) -> None:
        """Periodic cleanup task runner."""
        while True:
            try:
                FileManager.cleanup_old_files()
                time.sleep(CLEANUP_INTERVAL)
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                time.sleep(RETRY_INTERVAL)

    @app.post("/swap", response_model=FaceFusionResponse)
    async def face_swap(
        self,
        source_image: UploadFile = File(...),
        target_image: UploadFile = File(...)
    ) -> FaceFusionResponse:
        """Process face swap operation.

        Args:
            source_image: Source face image.
            target_image: Target image/video.

        Returns:
            FaceFusionResponse with task details.

        Raises:
            HTTPException: If processing fails.
        """
        task_id = str(uuid.uuid4())
        logger.info(f"Face swap task: {task_id}")

        try:
            source_path = await FileManager.save_upload_file(
                source_image, f"{task_id}_source"
            )
            target_path = await FileManager.save_upload_file(
                target_image, f"{task_id}_target"
            )
            
            extension = Path(target_image.filename).suffix
            output_path = OUTPUT_DIR / f"{task_id}_output{extension}"

            await run_facefusion_with_ray_job(
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
            logger.error(f"Face swap error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/status/{task_id}", response_model=TaskStatus)
    async def get_status(self, task_id: str) -> TaskStatus:
        """Get task status.

        Args:
            task_id: Task identifier.

        Returns:
            TaskStatus with current status and output path if completed.

        Raises:
            HTTPException: If status check fails.
        """
        try:
            status, logs = get_task_status(task_id)
            output_files = list(OUTPUT_DIR.glob(f"{task_id}_output.*"))
            output_path = output_files[0] if output_files else None

            return TaskStatus(
                task_id=task_id,
                status=status,
                result=str(output_path) if status == "completed" and output_path else None,
                logs=logs
            )

        except Exception as e:
            logger.error(f"Status check error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check(self) -> Dict[str, str]:
        """Check service health status."""
        return {"status": "ok"}

    @app.get("/stats")
    async def service_stats(self) -> Dict[str, int]:
        """Get service statistics."""
        try:
            size_bytes = sum(
                f.stat().st_size 
                for f in UPLOAD_DIR.glob("*") 
                if f.is_file()
            )
            return {"upload_dir_size": size_bytes}
        except Exception as e:
            logger.error(f"Stats error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

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
            FaceFusionService.bind(),
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