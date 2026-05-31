"""FastAPI routes and Ray Serve deployment for FaceFusion jobs.

The route functions are module-level so they can be unit-tested without a Ray
Serve replica context. Ray Serve deploys the same FastAPI app through
``@serve.ingress(app)``.

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

import ray
from fastapi import FastAPI, File, HTTPException, UploadFile
from ray import serve

from config import (
    LOG_LEVEL,
    OUTPUT_DIR,
    RAY_ADDRESS,
    SERVICE_HOST,
    SERVICE_PORT,
    UPLOAD_DIR,
)
from facefusion_job import get_task_status, run_facefusion_with_ray_job
from models import FaceFusionResponse, TaskStatus

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "facefusion_service.log"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5
CLEANUP_INTERVAL = 86400  # 1 day in seconds
RETRY_INTERVAL = 3600  # 1 hour in seconds
DEFAULT_CUTOFF_DAYS = 5

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure application-wide logging with file and console handlers."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT)
    root_logger = logging.getLogger()
    configured_log_level = getattr(logging, LOG_LEVEL, logging.DEBUG)
    root_logger.setLevel(configured_log_level)

    if not any(
        isinstance(handler, RotatingFileHandler)
        and Path(handler.baseFilename) == LOG_FILE.resolve()
        for handler in root_logger.handlers
    ):
        file_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=MAX_LOG_SIZE, backupCount=LOG_BACKUP_COUNT
        )
        file_handler.setLevel(configured_log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    if not any(
        getattr(handler, "_facefusion_console_handler", False)
        for handler in root_logger.handlers
    ):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(configured_log_level)
        console_handler.setFormatter(formatter)
        setattr(console_handler, "_facefusion_console_handler", True)
        root_logger.addHandler(console_handler)


class FileManager:
    """File operations for uploads and periodic cleanup."""

    @staticmethod
    def safe_filename(filename: str | None) -> str:
        """Return a basename-only upload filename.

        Args:
            filename: Client-provided upload filename.

        Returns:
            A safe basename suitable for storage under ``UPLOAD_DIR``.

        Raises:
            HTTPException: If the upload does not include a usable filename.
        """
        if not filename:
            raise HTTPException(
                status_code=400, detail="Uploaded file needs a filename"
            )

        safe_name = Path(filename).name
        if safe_name in {"", ".", ".."}:
            raise HTTPException(
                status_code=400, detail="Uploaded file needs a filename"
            )

        return safe_name

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
            safe_name = FileManager.safe_filename(upload_file.filename)
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            file_path = UPLOAD_DIR / f"{uid}_{safe_name}"
            content = await upload_file.read()

            file_path.write_bytes(content)
            logger.info(f"Saved file: {file_path}")

            return file_path

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"File save error: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Failed to save file: {str(e)}"
            )

    @staticmethod
    def cleanup_old_files(cutoff_days: int = DEFAULT_CUTOFF_DAYS) -> None:
        """Remove files older than the cutoff period.

        Args:
            cutoff_days: Number of days after which files should be removed.
        """
        try:
            if not UPLOAD_DIR.exists():
                return

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


app = FastAPI(title="FaceFusion Service")


@app.post("/swap", response_model=FaceFusionResponse)
async def face_swap(
    source_image: UploadFile = File(...), target_image: UploadFile = File(...)
) -> FaceFusionResponse:
    """Accept uploads and schedule a FaceFusion Ray job."""
    task_id = str(uuid.uuid4())
    logger.info(f"Face swap task: {task_id}")

    try:
        target_filename = FileManager.safe_filename(target_image.filename)
        source_path = await FileManager.save_upload_file(
            source_image, f"{task_id}_source"
        )
        target_path = await FileManager.save_upload_file(
            target_image, f"{task_id}_target"
        )

        extension = Path(target_filename).suffix
        output_path = OUTPUT_DIR / f"{task_id}_output{extension}"

        await run_facefusion_with_ray_job(
            task_id, str(source_path), str(target_path), str(output_path)
        )

        return FaceFusionResponse(
            task_id=task_id, status="processing", output_path=str(output_path)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face swap error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{task_id}", response_model=TaskStatus)
async def get_status(task_id: str) -> TaskStatus:
    """Return task status, logs, and output path when available."""
    try:
        status, logs = get_task_status(task_id)
        output_files = list(OUTPUT_DIR.glob(f"{task_id}_output.*"))
        output_path = output_files[0] if output_files else None

        return TaskStatus(
            task_id=task_id,
            status=status,
            result=str(output_path) if status == "completed" and output_path else None,
            logs=logs,
        )

    except Exception as e:
        logger.error(f"Status check error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Return a minimal readiness response."""
    return {"status": "ok"}


@app.get("/stats")
async def service_stats() -> dict[str, int]:
    """Return upload storage metrics."""
    try:
        if not UPLOAD_DIR.exists():
            return {"upload_dir_size": 0}

        size_bytes = sum(f.stat().st_size for f in UPLOAD_DIR.glob("*") if f.is_file())
        return {"upload_dir_size": size_bytes}
    except Exception as e:
        logger.error(f"Stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@serve.deployment(
    num_replicas=1,
    health_check_period_s=30,
    health_check_timeout_s=60,
    graceful_shutdown_wait_loop_s=120,
    graceful_shutdown_timeout_s=60,
)
@serve.ingress(app)
class FaceFusionService:
    """Ray Serve deployment wrapper for the FastAPI app."""

    def __init__(self) -> None:
        """Initialize service and start cleanup thread."""
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name="CleanupThread"
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


if __name__ == "__main__":
    try:
        setup_logging()
        logger.info("Starting FaceFusion Service")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        if RAY_ADDRESS:
            logger.info(f"Connecting to Ray cluster at {RAY_ADDRESS}")
            ray.init(address=RAY_ADDRESS)
        else:
            logger.info("Initializing Ray locally")
            ray.init()

        # cluster_resources() is part of Ray's untyped public API.
        cluster_info = ray.cluster_resources()  # type: ignore[no-untyped-call]
        logger.info(f"Ray cluster resources: {cluster_info}")

        logger.info("Starting Ray Serve")
        serve.start(
            http_options={
                "host": SERVICE_HOST,
                "port": SERVICE_PORT,
                "location": "EveryNode",
            }
        )

        logger.info("Deploying FaceFusion Service")
        serve.run(
            # bind() is injected by the @serve.deployment decorator.
            FaceFusionService.bind(),  # type: ignore[attr-defined]
            route_prefix="/v1/model/facefusion",
            name="facefusion_service",
        )

        logger.info("Service deployment completed successfully")
        logger.info(
            f"Service is running at "
            f"http://{SERVICE_HOST}:{SERVICE_PORT}/v1/model/facefusion"
        )

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")

    except Exception as e:
        logger.critical(f"Fatal error during service startup: {str(e)}", exc_info=True)
        sys.exit(1)
