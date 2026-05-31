"""Runtime configuration for the FaceFusion Ray Serve API.

Environment values are loaded once at import time. Relative paths are resolved
from the repository root, while absolute paths are preserved by ``pathlib``.

Typical usage example:
    from config import UPLOAD_DIR, OUTPUT_DIR
    uploaded_file = UPLOAD_DIR / "example.jpg"
"""

import os
from pathlib import Path

import dotenv

dotenv.load_dotenv()

PathLike = str | Path


def _get_env_path(env_key: str, default: str) -> Path:
    """Return an environment-provided path or a repository-relative default.

    Args:
        env_key: Environment variable key.
        default: Relative default value used when the variable is unset.

    Returns:
        Resolved path value.
    """
    return BASE_DIR / os.getenv(env_key, default)


BASE_DIR: Path = Path(__file__).resolve().parent

UPLOAD_DIR: Path = _get_env_path("UPLOAD_DIR", "uploads")
OUTPUT_DIR: Path = _get_env_path("OUTPUT_DIR", "outputs")
FACEFUSION_SCRIPT: Path = _get_env_path("FACEFUSION_PATH", "facefusion.py")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SERVICE_HOST: str = os.getenv("SERVICE_HOST", "0.0.0.0")
SERVICE_PORT: int = int(os.getenv("SERVICE_PORT", "9999"))
RAY_ADDRESS: str = os.getenv("RAY_ADDRESS", "auto")
EXECUTION_PROVIDER: str = os.getenv("EXECUTION_PROVIDER", "cuda")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG").upper()
