"""Ray job management module for FaceFusion tasks.

This module handles the creation and management of Ray jobs for face fusion
processing. It provides functionality for running face fusion tasks and
tracking their status and logs.

Typical usage example:
    task_id = "123"
    await run_facefusion_with_ray_job(
        task_id=task_id,
        source_path="source.jpg",
        target_path="target.jpg",
        output_path="output.jpg"
    )
    status, logs = get_task_status(task_id)
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import dotenv

from config import FACEFUSION_SCRIPT, EXECUTION_PROVIDER

# Module setup
logger = logging.getLogger(__name__)
dotenv.load_dotenv()

# Task tracking state
TaskStatus = str  # Type alias for task status strings
tasks: Dict[str, TaskStatus] = {}
task_logs: Dict[str, List[str]] = {}

async def run_facefusion_with_ray_job(
    task_id: str,
    source_path: str,
    target_path: str,
    output_path: str,
    execution_provider: str = EXECUTION_PROVIDER
) -> None:
    """Initiates a face fusion task using Ray job submission.

    Args:
        task_id: Unique identifier for the task.
        source_path: Path to source face image.
        target_path: Path to target image/video.
        output_path: Path where result should be saved.
        execution_provider: Backend for processing (cuda/cpu).

    Raises:
        RuntimeError: If Ray job submission fails.
    """
    tasks[task_id] = "processing"
    task_logs[task_id] = []
    task_logs[task_id].append(f"Starting task {task_id}")
    task_logs[task_id].append(f"Source: {source_path}")
    task_logs[task_id].append(f"Target: {target_path}")
    task_logs[task_id].append(f"Output: {output_path}")
    
    asyncio.create_task(_run_subprocess(task_id, source_path, target_path, output_path, execution_provider))

async def _run_subprocess(
    task_id: str,
    source_path: str,
    target_path: str,
    output_path: str,
    execution_provider: str
) -> None:
    """Executes the face fusion subprocess using Ray job submit.

    Args:
        task_id: Unique identifier for the task.
        source_path: Path to source face image.
        target_path: Path to target image/video.
        output_path: Path where result should be saved.
        execution_provider: Backend for processing (cuda/cpu).
    """
    command = [
        "ray", "job", "submit",
        "--address=auto",
        "--",
        "python",
        str(FACEFUSION_SCRIPT),
        "headless-run",
        "-s", source_path,
        "-t", target_path,
        "-o", output_path,
        "--execution-provider", execution_provider
    ]
    
    task_logs[task_id].append(f"Executing command: {' '.join(command)}")
    
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        

        while True:
            line = await process.stdout.readline()
            if not line:
                break
            log_line = line.decode().strip()
            task_logs[task_id].append(log_line)
            logger.info(f"Task {task_id}: {log_line}")


        stderr = await process.stderr.read()
        if stderr:
            error_lines = stderr.decode().strip().split('\n')
            for error_line in error_lines:
                task_logs[task_id].append(f"ERROR: {error_line}")
                logger.error(f"Task {task_id} error: {error_line}")
            
        await process.wait()
        if process.returncode == 0:
            tasks[task_id] = "completed"
            task_logs[task_id].append("Task completed successfully")
        else:
            tasks[task_id] = "failed"
            task_logs[task_id].append(f"Task failed with return code {process.returncode}")
            
    except Exception as e:
        tasks[task_id] = "failed"
        task_logs[task_id].append(f"Error: {str(e)}")
        logger.error(f"Error running subprocess: {e}", exc_info=True)

def get_task_status(task_id: str) -> Tuple[str, List[str]]:
    """Retrieves the current status and logs for a task.

    Args:
        task_id: Unique identifier for the task.

    Returns:
        A tuple containing:
        - Current task status (str): One of "processing", "completed", "failed", "not_found"
        - List of log messages (List[str])
    """
    status = tasks.get(task_id, "not_found")
    logs = task_logs.get(task_id, [])
    return status, logs