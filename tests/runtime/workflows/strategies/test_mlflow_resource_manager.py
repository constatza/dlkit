"""Tests for MLflowResourceManager edge cases."""

from __future__ import annotations

import pytest

import dlkit.runtime.workflows.strategies.tracking.mlflow_resource_manager as resource_module
from dlkit.runtime.workflows.strategies.tracking.mlflow_client_factory import (
    MLflowClientFactory,
)
from dlkit.runtime.workflows.strategies.tracking.mlflow_resource_manager import (
    MLflowResourceManager,
)


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
