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
from dlkit.tools.config.mlflow_settings import (
    MLflowSettings,
    MLflowClientSettings,
    MLflowServerSettings,
)
from dlkit.tools.io import url_resolver


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


def test_create_run_normalizes_plain_local_artifact_destination_for_experiment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ensure plain local artifact paths are normalized before experiment creation."""
    artifacts_dir = tmp_path / "mlartifacts"
    settings = MLflowSettings(
        enabled=True,
        client=MLflowClientSettings(
            tracking_uri=(tmp_path / "mlruns").as_uri(),
            experiment_name="test_exp",
            run_name="test_run",
        ),
        server=MLflowServerSettings(artifacts_destination=str(artifacts_dir)),
    )

    manager = MLflowResourceManager(settings)
    mock_client = Mock()
    mock_start_run = MagicMock()
    mock_start_run.return_value.__enter__.return_value = SimpleNamespace(
        info=SimpleNamespace(run_id="run-123")
    )
    mock_start_run.return_value.__exit__.return_value = None

    captured: dict[str, str | None] = {}

    def _capture_experiment(
        _client: Mock,
        _experiment_name: str,
        artifact_location: str | None = None,
    ) -> str:
        captured["artifact_location"] = artifact_location
        return "exp-123"

    monkeypatch.setattr(MLflowClientFactory, "create_client", lambda *_args, **_kwargs: mock_client)
    monkeypatch.setattr(
        MLflowClientFactory,
        "get_or_create_experiment",
        _capture_experiment,
    )
    monkeypatch.setattr(resource_module.mlflow, "start_run", mock_start_run, raising=True)

    with manager:
        with manager.create_run(experiment_name="exp", run_name="my-run"):
            pass

    expected = url_resolver.normalize_uri(str(artifacts_dir), Path.cwd())
    assert captured["artifact_location"] == expected
    assert captured["artifact_location"] is not None
    assert captured["artifact_location"].startswith("file://")


def test_create_run_preserves_cloud_artifact_destination_for_experiment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ensure cloud artifact URIs pass through unchanged."""
    cloud_uri = "s3://bucket/path/to/artifacts"
    settings = MLflowSettings(
        enabled=True,
        client=MLflowClientSettings(
            tracking_uri=(tmp_path / "mlruns").as_uri(),
            experiment_name="test_exp",
            run_name="test_run",
        ),
        server=MLflowServerSettings(artifacts_destination=cloud_uri),
    )

    manager = MLflowResourceManager(settings)
    mock_client = Mock()
    mock_start_run = MagicMock()
    mock_start_run.return_value.__enter__.return_value = SimpleNamespace(
        info=SimpleNamespace(run_id="run-123")
    )
    mock_start_run.return_value.__exit__.return_value = None

    captured: dict[str, str | None] = {}

    def _capture_experiment(
        _client: Mock,
        _experiment_name: str,
        artifact_location: str | None = None,
    ) -> str:
        captured["artifact_location"] = artifact_location
        return "exp-123"

    monkeypatch.setattr(MLflowClientFactory, "create_client", lambda *_args, **_kwargs: mock_client)
    monkeypatch.setattr(
        MLflowClientFactory,
        "get_or_create_experiment",
        _capture_experiment,
    )
    monkeypatch.setattr(resource_module.mlflow, "start_run", mock_start_run, raising=True)

    with manager:
        with manager.create_run(experiment_name="exp", run_name="my-run"):
            pass

    assert captured["artifact_location"] == cloud_uri


def test_create_run_normalizes_windows_drive_artifact_destination_for_experiment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ensure Windows drive paths are normalized to file:// URIs for MLflow."""
    windows_path = r"C:\Users\runneradmin\AppData\Local\Temp\pytest-1\mlartifacts"
    settings = MLflowSettings(
        enabled=True,
        client=MLflowClientSettings(
            tracking_uri=(tmp_path / "mlruns").as_uri(),
            experiment_name="test_exp",
            run_name="test_run",
        ),
        server=MLflowServerSettings(artifacts_destination=windows_path),
    )

    manager = MLflowResourceManager(settings)
    mock_client = Mock()
    mock_start_run = MagicMock()
    mock_start_run.return_value.__enter__.return_value = SimpleNamespace(
        info=SimpleNamespace(run_id="run-123")
    )
    mock_start_run.return_value.__exit__.return_value = None

    captured: dict[str, str | None] = {}

    def _capture_experiment(
        _client: Mock,
        _experiment_name: str,
        artifact_location: str | None = None,
    ) -> str:
        captured["artifact_location"] = artifact_location
        return "exp-123"

    monkeypatch.setattr(MLflowClientFactory, "create_client", lambda *_args, **_kwargs: mock_client)
    monkeypatch.setattr(
        MLflowClientFactory,
        "get_or_create_experiment",
        _capture_experiment,
    )
    monkeypatch.setattr(resource_module.mlflow, "start_run", mock_start_run, raising=True)

    with manager:
        with manager.create_run(experiment_name="exp", run_name="my-run"):
            pass

    assert captured["artifact_location"] is not None
    expected = url_resolver.normalize_uri(windows_path, Path.cwd())
    assert captured["artifact_location"] == expected
