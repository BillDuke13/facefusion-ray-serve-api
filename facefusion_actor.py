"""Ray actor implementation for FaceFusion processing.

This module provides a Ray actor that handles face fusion processing tasks
using the facefusion command line tool. It manages task execution and status
tracking.
"""

import ray
import subprocess
import logging
import os
from pathlib import Path
from typing import Dict
from config import BASE_DIR

logger = logging.getLogger(__name__)

@ray.remote
class FaceFusionActor:
    """Handles face fusion processing using the facefusion command line tool.

    This actor manages the execution of face fusion tasks by running the facefusion.py
    script with the appropriate parameters. It maintains a dictionary of task statuses
    and handles the execution environment.

    Attributes:
        tasks: Dictionary mapping task IDs to their current status.
        base_dir: Base directory path for the facefusion installation.
    """

    def __init__(self) -> None:
        """Initializes the FaceFusionActor with an empty task dictionary."""
        logger.info("Initializing FaceFusionActor")
        self.tasks: Dict[str, str] = {}
        self.base_dir = Path(BASE_DIR)
        logger.info("FaceFusionActor initialized successfully")

    async def process_face_fusion(
        self,
        task_id: str,
        source_path: str,
        target_path: str,
        output_path: str
    ) -> Dict[str, str]:
        """Executes a face fusion processing task.

        Args:
            task_id: Unique identifier for the task.
            source_path: Path to the source face image.
            target_path: Path to the target video/image.
            output_path: Path where the processed result should be saved.

        Returns:
            Dict containing task status and either output_path or error message.

        Raises:
            FileNotFoundError: If source or target files don't exist.
            subprocess.CalledProcessError: If the facefusion command fails.
        """
        logger.info(f"Starting face fusion task: {task_id}")
        try:
            # Update task status
            self.tasks[task_id] = "processing"

            # Verify input files exist
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Source file not found: {source_path}")
            if not os.path.exists(target_path):
                raise FileNotFoundError(f"Target file not found: {target_path}")

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Construct command matching the working pattern
            command = [
                "python",
                str(self.base_dir / "facefusion.py"),
                "headless-run",
                "-s", source_path,
                "-t", target_path,
                "-o", output_path
            ]

            logger.info(f"Executing command: {' '.join(command)}")

            # Execute the command
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                cwd=str(self.base_dir)  # Set working directory to base directory
            )

            # Process the results
            if process.returncode == 0:
                logger.info(f"Task {task_id} completed successfully")
                self.tasks[task_id] = "completed"
                return {
                    "status": "completed",
                    "output_path": output_path
                }
            else:
                raise subprocess.CalledProcessError(
                    process.returncode, command, process.stdout, process.stderr
                )

        except Exception as e:
            error_message = f"Error processing task {task_id}: {str(e)}"
            logger.error(error_message, exc_info=True)
            self.tasks[task_id] = "failed"
            return {
                "status": "failed",
                "error": error_message
            }

    async def get_task_status(self, task_id: str) -> str:
        """Retrieves the current status of a face fusion task.

        Args:
            task_id: The unique identifier of the task.

        Returns:
            str: Current status of the task ("processing", "completed", "failed",
                or "not_found" if task doesn't exist).
        """
        status = self.tasks.get(task_id, "not_found")
        logger.debug(f"Status for task {task_id}: {status}")
        return status