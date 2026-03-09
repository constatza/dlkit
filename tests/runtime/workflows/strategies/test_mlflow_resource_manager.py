"""Tests for MLflow client-only URI resolution and nested run behavior."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import mlflow
import pytest

import dlkit.runtime.workflows.strategies.tracking.uri_resolver as uri_resolver
from dlkit.runtime.workflows.strategies.tracking.mlflow_client_factory import (
    MLflowClientFactory,
)
from dlkit.runtime.workflows.strategies.tracking.mlflow_resource_manager import (
    MLflowResourceManager,
)
from dlkit.tools.config.mlflow_settings import MLflowSettings


def test_resolve_tracking_uri_prefers_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.company.local")
    monkeypatch.setattr(uri_resolver, "local_host_alive", lambda: True, raising=True)

    resolved = uri_resolver.resolve_tracking_uri()
    assert resolved == "https://mlflow.company.local"


def test_resolve_tracking_uri_uses_localhost_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(uri_resolver, "local_host_alive", lambda: True, raising=True)

    resolved = uri_resolver.resolve_tracking_uri()
    assert resolved == "http://127.0.0.1:5000"


def test_resolve_tracking_uri_falls_back_to_sqlite(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(uri_resolver, "local_host_alive", lambda: False, raising=True)
    monkeypatch.setattr(
        "dlkit.runtime.workflows.strategies.tracking.uri_resolver.locations.mlruns_backend_uri",
        lambda: "sqlite:///tests/artifacts/mlruns/mlflow.db",
    )

    resolved = uri_resolver.resolve_tracking_uri()
    assert resolved.startswith("sqlite:///")
    assert resolved.endswith("mlflow.db")


def test_parse_mlflow_scheme_rejects_invalid_scheme() -> None:
    with pytest.raises(ValueError, match="Unsupported MLflow tracking URI scheme"):
        uri_resolver.parse_mlflow_scheme("ftp://example.com/mlflow")


def test_resolve_artifact_uri_derives_from_sqlite(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_ARTIFACT_URI", raising=False)
    tracking_uri = "sqlite:////tmp/mlruns/mlflow.db"

    artifact_uri = uri_resolver.resolve_artifact_uri(tracking_uri)
    assert artifact_uri == "file:///tmp/mlruns/artifacts"


def test_create_run_preserves_nested_parent_child_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = MLflowSettings(enabled=True, experiment_name="exp")
    manager = MLflowResourceManager(settings)

    mock_client = Mock()
    monkeypatch.setattr(MLflowClientFactory, "create_client", lambda **_kwargs: mock_client)
    monkeypatch.setattr(MLflowClientFactory, "validate_client_connectivity", lambda _c: True)
    monkeypatch.setattr(
        MLflowClientFactory,
        "get_or_create_experiment",
        lambda *_args, **_kwargs: "exp-123",
    )
    monkeypatch.setattr(
        "dlkit.runtime.workflows.strategies.tracking.mlflow_resource_manager.resolve_mlflow_uris",
        lambda: uri_resolver.ResolvedMlflowUris(
            tracking_uri="sqlite:////tmp/mlruns/mlflow.db",
            artifact_uri="file:///tmp/mlruns/artifacts",
            scheme="sqlite",
        ),
    )

    start_run = MagicMock()
    start_run.side_effect = [
        MagicMock(
            __enter__=MagicMock(
                return_value=SimpleNamespace(info=SimpleNamespace(run_id="parent-run"))
            ),
            __exit__=MagicMock(return_value=None),
        ),
        MagicMock(
            __enter__=MagicMock(
                return_value=SimpleNamespace(info=SimpleNamespace(run_id="child-run"))
            ),
            __exit__=MagicMock(return_value=None),
        ),
    ]
    monkeypatch.setattr(
        "dlkit.runtime.workflows.strategies.tracking.mlflow_resource_manager.mlflow.start_run",
        start_run,
    )

    with manager:
        with manager.create_run(experiment_name="exp", run_name="study", nested=False):
            with manager.create_run(experiment_name="exp", run_name="trial-1", nested=True):
                pass

    parent_call = start_run.call_args_list[0].kwargs
    child_call = start_run.call_args_list[1].kwargs

    assert parent_call["nested"] is False
    assert child_call["nested"] is True
    assert child_call["parent_run_id"] == "parent-run"


def test_reset_global_state_preserves_tracking_uri_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracking_uri = "sqlite:////tmp/mlruns/env-preserve.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)

    MLflowResourceManager.reset_global_state()

    assert os.environ.get("MLFLOW_TRACKING_URI") == tracking_uri
    assert mlflow.get_tracking_uri() == tracking_uri
