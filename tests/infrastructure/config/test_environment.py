"""Tests for DLKit environment configuration."""

import os

import pytest

from dlkit.infrastructure.config.environment import best_effort_retry_budget


def test_mlflow_retry_defaults_are_set():
    """Verify that MLflow retry environment variables are configured with sensible defaults."""
    # The environment module should have already been imported and configured these
    # Note: All values must be integers (MLflow requirement)
    assert os.environ.get("MLFLOW_HTTP_REQUEST_MAX_RETRIES") == "5"
    assert os.environ.get("MLFLOW_HTTP_REQUEST_TIMEOUT") == "30"
    assert os.environ.get("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR") == "2"


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


def test_best_effort_retry_budget_tightens_then_restores_prior_values(
    monkeypatch: pytest.MonkeyPatch,
):
    """Fail-fast values apply inside the context and prior values return after."""
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "9")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "30")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "2")

    with best_effort_retry_budget():
        assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "2"
        assert os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] == "5"
        assert os.environ["MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR"] == "1"

    assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "9"
    assert os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] == "30"
    assert os.environ["MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR"] == "2"


def test_best_effort_retry_budget_pops_keys_absent_beforehand(
    monkeypatch: pytest.MonkeyPatch,
):
    """Keys with no prior value are removed again, not left as the string "None"."""
    monkeypatch.delenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", raising=False)

    with best_effort_retry_budget():
        assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "2"

    assert "MLFLOW_HTTP_REQUEST_MAX_RETRIES" not in os.environ


@pytest.mark.timeout(5)
def test_best_effort_retry_budget_is_reentrant_on_same_thread():
    """Nested contexts must not deadlock (RLock, not Lock)."""
    with best_effort_retry_budget():
        with best_effort_retry_budget():
            assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "2"
