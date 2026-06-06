"""Test artifact isolation via test environment."""

from __future__ import annotations

from pathlib import Path


def test_artifacts_directory_is_created(test_artifacts_dir: Path) -> None:
    """Test that the test artifacts directory fixture works correctly."""
    assert test_artifacts_dir.exists(), "Test artifacts directory should exist"
    assert test_artifacts_dir.is_dir(), "Test artifacts directory should be a directory"
