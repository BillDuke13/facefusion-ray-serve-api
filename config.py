"""Configuration module for the FaceFusion service.

This module handles all configuration settings for the FaceFusion service,
including path resolution, environment variable loading, and directory setup.

Typical usage example:
    from config import UPLOAD_DIR, OUTPUT_DIR
    uploaded_file = UPLOAD_DIR / "example.jpg"
"""

from typing import Union
import os
from pathlib import Path

import dotenv

# Load environment variables
dotenv.load_dotenv()

# Type aliases
PathLike = Union[str, Path]

def _get_env_path(env_key: str, default: str) -> Path:
    """Gets path from environment variable with default fallback.
    
    Args:
        env_key: Environment variable key.
        default: Default value if env var not set.
    
    Returns:
        Path object for the directory/file.
    """
    return BASE_DIR / os.getenv(env_key, default)

# Base directory configuration
BASE_DIR: Path = Path(__file__).resolve().parent

# Storage directories configuration
UPLOAD_DIR: Path = _get_env_path("UPLOAD_DIR", "uploads")
OUTPUT_DIR: Path = _get_env_path("OUTPUT_DIR", "outputs")
FACEFUSION_SCRIPT: Path = _get_env_path("FACEFUSION_PATH", "facefusion.py")

# Ensure required directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Service configuration
SERVICE_HOST: str = os.getenv("SERVICE_HOST", "localhost")
SERVICE_PORT: int = int(os.getenv("SERVICE_PORT", "8000"))
RAY_ADDRESS: str = os.getenv("RAY_ADDRESS", "auto")
EXECUTION_PROVIDER: str = os.getenv("EXECUTION_PROVIDER", "cuda")