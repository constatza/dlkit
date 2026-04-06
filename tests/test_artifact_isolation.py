"""Test artifact isolation via DLKIT_ROOT_DIR environment variable."""

from __future__ import annotations

import os
from pathlib import Path

from dlkit.infrastructure.io.locations import mlruns_dir, output


def test_artifacts_go_to_dlkit_root_dir() -> None:
    """Test that artifacts are directed to DLKIT_ROOT_DIR during tests."""
    root_dir = os.environ.get("DLKIT_ROOT_DIR")
    assert root_dir is not None, "DLKIT_ROOT_DIR must be set in test environment"

    expected_base = Path(root_dir)

    mlruns_path = mlruns_dir()
    output_path = output()

    # All paths should be under DLKIT_ROOT_DIR
    # Use as_posix() on both sides to avoid backslash/forward-slash mismatch on Windows
    assert expected_base.as_posix() in mlruns_path.as_posix(), (
        f"MLruns path should be under DLKIT_ROOT_DIR {expected_base}: {mlruns_path}"
    )
    assert expected_base.as_posix() in output_path.as_posix(), (
        f"Output path should be under DLKIT_ROOT_DIR {expected_base}: {output_path}"
    )


def test_artifacts_directory_is_created(test_artifacts_dir: Path) -> None:
    """Test that the test artifacts directory fixture works correctly."""
    assert test_artifacts_dir.exists(), "Test artifacts directory should exist"
    assert test_artifacts_dir.is_dir(), "Test artifacts directory should be a directory"


def test_artifact_creation_and_isolation() -> None:
    """Test that creating artifacts during tests stays within the test temp tree."""
    root_dir = os.environ.get("DLKIT_ROOT_DIR")
    assert root_dir is not None, "DLKIT_ROOT_DIR must be set in test environment"

    mlruns_path = mlruns_dir()
    mlruns_path.parent.mkdir(parents=True, exist_ok=True)
    if mlruns_path.exists() and not mlruns_path.is_dir():
        mlruns_path.unlink()
    mlruns_path.mkdir(parents=True, exist_ok=True)

    test_artifact = mlruns_path / "test_artifact.txt"
    test_artifact.write_text("test content")

    assert test_artifact.exists(), "Test artifact should be created"
    # Artifact should be under the temp tree (contains the tmp path prefix)
    assert Path(root_dir).as_posix() in test_artifact.as_posix(), (
        f"Artifact should be under DLKIT_ROOT_DIR: {test_artifact}"
    )
