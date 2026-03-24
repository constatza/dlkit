"""Tests for MLflow client-only URI resolution and nested run behavior."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import mlflow
import pytest

import dlkit.runtime.workflows.strategies.tracking.uri_resolver as uri_resolver
from dlkit.runtime.workflows.strategies.tracking.backend import (
    LocalServerBackend,
    LocalSqliteBackend,
)
from dlkit.runtime.workflows.strategies.tracking.mlflow_client_factory import (
    MLflowClientFactory,
)
from dlkit.runtime.workflows.strategies.tracking.mlflow_resource_manager import (
    MLflowResourceManager,
)
from dlkit.tools.config.mlflow_settings import MLflowSettings
from dlkit.tools.io import url_resolver


def test_resolve_tracking_uri_prefers_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.company.local")
    monkeypatch.setattr(uri_resolver, "local_host_alive", lambda: True, raising=True)

    resolved = uri_resolver.resolve_tracking_uri()
    assert resolved == "https://mlflow.company.local"


def test_resolve_tracking_uri_uses_mocked_localhost_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_resolve_tracking_uri_honours_sqlite_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """select_backend() now respects sqlite:/// MLFLOW_TRACKING_URI env vars."""
    sqlite_uri = f"sqlite:///{(tmp_path / 'explicit.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", sqlite_uri)
    monkeypatch.setattr(uri_resolver, "local_host_alive", lambda: False, raising=True)

    resolved = uri_resolver.resolve_tracking_uri()
    assert resolved == sqlite_uri


def test_parse_mlflow_scheme_rejects_invalid_scheme() -> None:
    with pytest.raises(ValueError, match="Unsupported MLflow tracking URI scheme"):
        uri_resolver.parse_mlflow_scheme("ftp://example.com/mlflow")


def test_resolve_artifact_uri_derives_from_sqlite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("MLFLOW_ARTIFACT_URI", raising=False)
    db_path = tmp_path / "mlruns" / "mlflow.db"
    tracking_uri = url_resolver.build_uri(db_path, scheme="sqlite")

    artifact_uri = uri_resolver.resolve_artifact_uri(tracking_uri)
    expected_artifact_uri = url_resolver.build_uri(db_path.parent / "artifacts", scheme="file")
    assert artifact_uri == expected_artifact_uri


def test_create_run_preserves_nested_parent_child_structure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = MLflowSettings(experiment_name="exp")
    db_path = tmp_path / "mlruns" / "mlflow.db"
    backend = LocalSqliteBackend(db_path=db_path)
    manager = MLflowResourceManager(settings, backend)

    mock_client = Mock()
    monkeypatch.setattr(MLflowClientFactory, "create_client", lambda **_kwargs: mock_client)
    monkeypatch.setattr(MLflowClientFactory, "validate_client_connectivity", lambda _c: True)
    monkeypatch.setattr(
        MLflowClientFactory,
        "get_or_create_experiment",
        lambda *_args, **_kwargs: "exp-123",
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


def test_reset_global_state_preserves_sqlite_tracking_uri(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SQLite URIs in env are now preserved by reset_global_state to avoid CWD leak."""
    tracking_uri = url_resolver.build_uri(tmp_path / "mlruns" / "env-preserve.db", scheme="sqlite")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)

    MLflowResourceManager.reset_global_state()

    # SQLite URI must be preserved so the isolation URI set by fixtures is never lost.
    assert os.environ.get("MLFLOW_TRACKING_URI") == tracking_uri


def test_reset_global_state_preserves_http_tracking_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP URIs should be preserved by reset_global_state (user-configured)."""
    http_uri = "http://mlflow.example.com:5000"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", http_uri)
    mlflow.set_tracking_uri(http_uri)

    MLflowResourceManager.reset_global_state()

    assert os.environ.get("MLFLOW_TRACKING_URI") == http_uri


def test_initialize_resources_suppresses_bootstrap_logs_for_sqlite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = MLflowSettings(experiment_name="exp")
    db_path = tmp_path / "mlruns" / "mlflow.db"
    backend = LocalSqliteBackend(db_path=db_path)
    manager = MLflowResourceManager(settings, backend)

    mock_client = Mock()
    suppression_calls: list[str] = []

    class _SuppressionContext:
        def __enter__(self):
            suppression_calls.append("enter")
            return None

        def __exit__(self, exc_type, exc_val, exc_tb):
            suppression_calls.append("exit")
            return False

    monkeypatch.setattr(MLflowClientFactory, "create_client", lambda **_kwargs: mock_client)
    monkeypatch.setattr(MLflowClientFactory, "validate_client_connectivity", lambda _c: True)
    monkeypatch.setattr(
        "dlkit.runtime.workflows.strategies.tracking.mlflow_resource_manager.suppress_mlflow_sqlite_bootstrap_logs",
        lambda: _SuppressionContext(),
    )

    with manager:
        pass

    assert suppression_calls == ["enter", "exit"]


def test_initialize_resources_skips_bootstrap_suppression_for_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = MLflowSettings(experiment_name="exp")
    backend = LocalServerBackend()
    manager = MLflowResourceManager(settings, backend)

    mock_client = Mock()
    suppression_calls: list[str] = []

    class _SuppressionContext:
        def __enter__(self):
            suppression_calls.append("enter")
            return None

        def __exit__(self, exc_type, exc_val, exc_tb):
            suppression_calls.append("exit")
            return False

    monkeypatch.setattr(MLflowClientFactory, "create_client", lambda **_kwargs: mock_client)
    monkeypatch.setattr(MLflowClientFactory, "validate_client_connectivity", lambda _c: True)
    monkeypatch.setattr(
        "dlkit.runtime.workflows.strategies.tracking.mlflow_resource_manager.suppress_mlflow_sqlite_bootstrap_logs",
        lambda: _SuppressionContext(),
    )

    with manager:
        pass

    assert suppression_calls == []
