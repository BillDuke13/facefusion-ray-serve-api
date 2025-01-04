import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
FACEFUSION_PATH = BASE_DIR / "facefusion.py"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Service configuration
SERVICE_HOST = "localhost"
SERVICE_PORT = 8000
RAY_ADDRESS = "auto"