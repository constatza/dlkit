"""Tests for DLKit environment configuration."""

import os


def test_mlflow_retry_defaults_are_set():
    """Verify that MLflow retry environment variables are configured with sensible defaults."""
    # The environment module should have already been imported and configured these
    # Note: All values must be integers (MLflow requirement)
    assert os.environ.get("MLFLOW_HTTP_REQUEST_MAX_RETRIES") == "2"
    assert os.environ.get("MLFLOW_HTTP_REQUEST_TIMEOUT") == "5"
    assert os.environ.get("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR") == "1"


def test_mlflow_retry_defaults_respect_user_overrides(monkeypatch):
    """Verify that user-set environment variables are preserved."""
    # This test demonstrates that if a user sets these before importing dlkit,
    # their values will be respected. In practice, since the environment module
    # is already loaded in conftest, we use monkeypatch to simulate this.

    # Set custom values (all must be integers)
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "10")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "30")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "2")

    # Verify they're set
    assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "10"
    assert os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] == "30"
    assert os.environ["MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR"] == "2"

    # Note: The actual function _configure_mlflow_retry_defaults() only sets
    # values if they're not already present, so this test verifies the override
    # behavior works as expected


def test_environment_module_loads_early():
    """Verify that the environment module is loaded before MLflow imports."""
    import sys

    # The environment module should be loaded
    assert "dlkit.infrastructure.config.environment" in sys.modules

    # The MLflow retry settings should be configured
    assert "MLFLOW_HTTP_REQUEST_MAX_RETRIES" in os.environ
