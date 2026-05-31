"""Pytest session setup.

Redirects the service's upload/output directories to a temporary location
before ``config`` is imported, so importing project modules during tests does
not create ``uploads/`` or ``outputs/`` directories inside the repository.
"""

import os
import tempfile

_TMP_ROOT = tempfile.mkdtemp(prefix="facefusion_tests_")
os.environ.setdefault("UPLOAD_DIR", os.path.join(_TMP_ROOT, "uploads"))
os.environ.setdefault("OUTPUT_DIR", os.path.join(_TMP_ROOT, "outputs"))
