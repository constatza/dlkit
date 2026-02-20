"""Tests for MLflowResourceManager edge cases."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import pytest
from unittest.mock import Mock, MagicMock

import dlkit.runtime.workflows.strategies.tracking.mlflow_resource_manager as resource_module
from dlkit.runtime.workflows.strategies.tracking.mlflow_client_factory import (
    MLflowClientFactory,
)
from dlkit.runtime.workflows.strategies.tracking.mlflow_resource_manager import (
    MLflowResourceManager,
)
from dlkit.tools.config.mlflow_settings import MLflowSettings, MLflowClientSettings


def test_manager_raises_when_server_cannot_start(
    monkeypatch: pytest.MonkeyPatch, mlflow_server_test_settings
) -> None:
    """Ensure we fail fast instead of hanging when the server refuses to start."""

    class _FailingContext:
        def __init__(self, *args, **kwargs):
            pass

        def start_server(self):
            raise RuntimeError("boom")

        def stop_server(self):  # pragma: no cover - should never be called
            raise AssertionError("stop_server should not be invoked when start fails")

    # Any attempt to build a client after the failure should be caught in the test
    def _unexpected_client(*args, **kwargs):
        raise AssertionError("client creation should not run when server start fails")

    monkeypatch.setattr(resource_module, "MLflowServerContext", _FailingContext, raising=True)
    monkeypatch.setattr(MLflowClientFactory, "create_client", _unexpected_client, raising=True)
    monkeypatch.setattr(
        MLflowClientFactory,
        "create_client_from_server_info",
        _unexpected_client,
        raising=True,
    )

    manager = MLflowResourceManager(mlflow_server_test_settings)

    with pytest.raises(RuntimeError, match="Failed to initialize MLflow server"):
        with manager:
            pass

    assert isinstance(manager._state.server_start_error, RuntimeError)


def test_create_run_uses_fluent_start_run_with_nested_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ensure run lifecycle is owned by mlflow.start_run context manager."""
    settings = MLflowSettings(
        enabled=True,
        client=MLflowClientSettings(
            tracking_uri=(tmp_path / "mlruns").as_uri(),
            experiment_name="test_exp",
            run_name="test_run",
        ),
    )
    manager = MLflowResourceManager(settings)
    mock_client = Mock()
    mock_start_run = MagicMock()
    mock_start_run.return_value.__enter__.return_value = SimpleNamespace(
        info=SimpleNamespace(run_id="run-123")
    )
    mock_start_run.return_value.__exit__.return_value = None

    monkeypatch.setattr(MLflowClientFactory, "create_client", lambda *_args, **_kwargs: mock_client)
    monkeypatch.setattr(
        MLflowClientFactory,
        "get_or_create_experiment",
        lambda *_args, **_kwargs: "exp-123",
    )
    monkeypatch.setattr(resource_module.mlflow, "start_run", mock_start_run, raising=True)

    with manager:
        with manager.create_run(experiment_name="exp", run_name="my-run", nested=True):
            assert manager._state.active_run_stack == ["run-123"]

    mock_start_run.assert_called_once_with(
        experiment_id="exp-123",
        run_name="my-run",
        nested=True,
    )
    assert manager._state.active_run_stack == []
