"""Configuration settings for the FaceFusion service.

This module contains all configuration variables and directory setup for the
FaceFusion service. It handles path resolution and directory creation.
"""

import os
from pathlib import Path

# Base directory is the parent directory of this file
BASE_DIR = Path(__file__).resolve().parent

# Directory for storing uploaded files
UPLOAD_DIR = BASE_DIR / "uploads"

# Directory for storing processed outputs
OUTPUT_DIR = BASE_DIR / "outputs"

# Path to the facefusion script
FACEFUSION_PATH = BASE_DIR / "facefusion.py"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Service configuration
SERVICE_HOST = "localhost"
SERVICE_PORT = 8000
RAY_ADDRESS = "auto"