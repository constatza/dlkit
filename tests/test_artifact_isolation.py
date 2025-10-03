"""Test artifact isolation and cleanup functionality."""

from __future__ import annotations

from pathlib import Path


from dlkit.tools.io.locations import mlruns_dir, mlartifacts_dir, output, optuna_storage_uri


def test_artifacts_go_to_test_directory() -> None:
    """Test that all artifacts are directed to tests/artifacts directory during tests."""
    # Get artifact paths
    mlruns_path = mlruns_dir()
    mlartifacts_path = mlartifacts_dir()
    output_path = output()
    optuna_uri = optuna_storage_uri()

    # All paths should contain 'tests/artifacts'
    assert "tests/artifacts" in mlruns_path.as_posix(), (
        f"MLruns path should go to tests/artifacts: {mlruns_path}"
    )
    assert "tests/artifacts" in mlartifacts_path.as_posix(), (
        f"MLartifacts path should go to tests/artifacts: {mlartifacts_path}"
    )
    assert "tests/artifacts" in output_path.as_posix(), (
        f"Output path should go to tests/artifacts: {output_path}"
    )
    assert "tests/artifacts" in optuna_uri.replace("\\", "/"), (
        f"Optuna URI should reference tests/artifacts: {optuna_uri}"
    )


def test_artifacts_directory_is_created(test_artifacts_dir: Path) -> None:
    """Test that the test artifacts directory fixture works correctly."""
    assert test_artifacts_dir.exists(), "Test artifacts directory should exist"
    assert test_artifacts_dir.is_dir(), "Test artifacts directory should be a directory"
    assert "tests/artifacts" in test_artifacts_dir.as_posix(), "Should be under tests/artifacts"


def test_artifact_creation_and_isolation() -> None:
    """Test that creating artifacts during tests works and they're isolated."""
    # Create some mock artifacts
    mlruns_path = mlruns_dir()
    mlruns_path.mkdir(parents=True, exist_ok=True)

    test_artifact = mlruns_path / "test_artifact.txt"
    test_artifact.write_text("test content")

    # Verify the artifact was created in test directory
    assert test_artifact.exists(), "Test artifact should be created"
    assert "tests/artifacts" in test_artifact.as_posix(), "Artifact should be in test directory"

    # The cleanup will be verified by the session fixture
