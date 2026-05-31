"""Tests for import-time configuration behavior."""

from pathlib import Path

import config


def test_config_creates_test_upload_and_output_dirs() -> None:
    assert config.UPLOAD_DIR.is_dir()
    assert config.OUTPUT_DIR.is_dir()
    assert config.LOG_LEVEL == "DEBUG"


def test_get_env_path_resolves_relative_paths(monkeypatch) -> None:
    monkeypatch.setenv("FACEFUSION_TEST_PATH", "relative/path")

    assert config._get_env_path("FACEFUSION_TEST_PATH", "fallback") == (
        config.BASE_DIR / "relative/path"
    )


def test_get_env_path_preserves_absolute_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FACEFUSION_TEST_PATH", str(tmp_path))

    assert config._get_env_path("FACEFUSION_TEST_PATH", "fallback") == tmp_path
