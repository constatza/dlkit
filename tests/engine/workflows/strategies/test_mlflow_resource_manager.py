"""Tests for MLflow client-only URI resolution and nested run behavior."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import mlflow
import pytest

import dlkit.engine.tracking.uri_resolver as uri_resolver
from dlkit.engine.tracking.backend import (
    LocalServerBackend,
    LocalSqliteBackend,
)
from dlkit.engine.tracking.mlflow_client_factory import (
    MLflowClientFactory,
)
from dlkit.engine.tracking.mlflow_resource_manager import (
    MLflowResourceManager,
)
from dlkit.infrastructure.config.tracking_settings import TrackingSettings
from dlkit.infrastructure.io import url_resolver


def test_select_backend_uses_explicit_http_uri(tmp_path: Path) -> None:
    from dlkit.engine.tracking.backend import RemoteServerBackend, select_backend

    backend = select_backend(uri="https://mlflow.company.local")
    assert isinstance(backend, RemoteServerBackend)
    assert backend.tracking_uri() == "https://mlflow.company.local"


def test_select_backend_uses_explicit_sqlite_uri(tmp_path: Path) -> None:
    from dlkit.engine.tracking.backend import LocalSqliteBackend, select_backend

    sqlite_uri = f"sqlite:///{(tmp_path / 'explicit.db').as_posix()}"
    backend = select_backend(uri=sqlite_uri)
    assert isinstance(backend, LocalSqliteBackend)
    assert backend.tracking_uri() == sqlite_uri


def test_select_backend_uses_probe_when_no_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    from dlkit.engine.tracking.backend import LocalServerBackend, select_backend

    backend = select_backend(uri=None, probe=lambda: True)
    assert isinstance(backend, LocalServerBackend)
    assert backend.tracking_uri() == "http://127.0.0.1:5000"


def test_select_backend_falls_back_to_sqlite_when_no_uri_and_no_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlkit.engine.tracking.backend import LocalSqliteBackend, select_backend

    monkeypatch.setattr(
        "dlkit.engine.tracking.discovery.locations.mlruns_backend_uri",
        lambda: "sqlite:///tests/artifacts/mlruns/mlflow.db",
    )
    backend = select_backend(uri=None, probe=lambda: False)
    assert isinstance(backend, LocalSqliteBackend)
    assert backend.tracking_uri().startswith("sqlite:///")
    assert backend.tracking_uri().endswith("mlflow.db")


def test_select_backend_raises_on_unsupported_scheme() -> None:
    from dlkit.engine.tracking.backend import select_backend

    with pytest.raises(ValueError, match="Unsupported MLflow tracking URI scheme"):
        select_backend(uri="ftp://example.com/mlflow")


def test_parse_mlflow_scheme_rejects_invalid_scheme() -> None:
    with pytest.raises(ValueError, match="Unsupported MLflow tracking URI scheme"):
        uri_resolver.parse_mlflow_scheme("ftp://example.com/mlflow")


def test_local_sqlite_backend_derives_artifact_uri_from_db_path(tmp_path: Path) -> None:
    from dlkit.engine.tracking.backend import LocalSqliteBackend

    db_path = tmp_path / "mlruns" / "mlflow.db"
    backend = LocalSqliteBackend(db_path=db_path)
    expected = url_resolver.build_uri(db_path.parent / "artifacts", scheme="file")
    assert backend.artifact_uri() == expected


def test_remote_backend_artifact_uri_is_none() -> None:
    from dlkit.engine.tracking.backend import RemoteServerBackend

    backend = RemoteServerBackend(uri="https://mlflow.example.com")
    assert backend.artifact_uri() is None


def test_create_run_preserves_nested_parent_child_structure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = TrackingSettings(backend="mlflow")
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
        "dlkit.engine.tracking.mlflow_resource_manager.mlflow.start_run",
        start_run,
    )

    with manager, manager.create_run(experiment_name="exp", run_name="study", nested=False):
        with manager.create_run(experiment_name="exp", run_name="trial-1", nested=True):
            pass

    parent_call = start_run.call_args_list[0].kwargs
    child_call = start_run.call_args_list[1].kwargs

    assert parent_call["nested"] is False
    assert child_call["nested"] is True
    assert child_call["parent_run_id"] == "parent-run"


def test_reset_global_state_applies_explicit_tracking_uri(tmp_path: Path) -> None:
    """reset_global_state re-applies the explicit URI to prevent CWD-leak in MLflow 3.x."""
    tracking_uri = url_resolver.build_uri(tmp_path / "mlruns" / "isolated.db", scheme="sqlite")
    mlflow.set_tracking_uri("sqlite:///some/other.db")

    MLflowResourceManager.reset_global_state(tracking_uri=tracking_uri)

    assert mlflow.get_tracking_uri() == tracking_uri


def test_reset_global_state_without_uri_leaves_mlflow_state_unchanged(tmp_path: Path) -> None:
    """When no URI is passed reset_global_state does not touch mlflow.set_tracking_uri."""
    tracking_uri = url_resolver.build_uri(tmp_path / "mlruns" / "keep.db", scheme="sqlite")
    mlflow.set_tracking_uri(tracking_uri)

    MLflowResourceManager.reset_global_state()  # no tracking_uri arg

    # State is not changed — reset_global_state doesn't re-apply anything
    # (caller is responsible for passing the correct URI when needed)
    assert mlflow.get_tracking_uri() == tracking_uri


def test_initialize_resources_suppresses_bootstrap_logs_for_sqlite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = TrackingSettings(backend="mlflow")
    db_path = tmp_path / "mlruns" / "mlflow.db"
    backend = LocalSqliteBackend(db_path=db_path)
    manager = MLflowResourceManager(settings, backend)

    mock_client = Mock()
    suppression_calls: list[str] = []

    class _SuppressionContext:
        def __enter__(self):
            suppression_calls.append("enter")

        def __exit__(self, exc_type, exc_val, exc_tb):
            suppression_calls.append("exit")
            return False

    monkeypatch.setattr(MLflowClientFactory, "create_client", lambda **_kwargs: mock_client)
    monkeypatch.setattr(MLflowClientFactory, "validate_client_connectivity", lambda _c: True)
    monkeypatch.setattr(
        "dlkit.engine.tracking.mlflow_resource_manager.suppress_mlflow_sqlite_bootstrap_logs",
        lambda: _SuppressionContext(),
    )

    with manager:
        pass

    assert suppression_calls == ["enter", "exit"]


def test_initialize_resources_skips_bootstrap_suppression_for_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = TrackingSettings(backend="mlflow")
    backend = LocalServerBackend()
    manager = MLflowResourceManager(settings, backend)

    mock_client = Mock()
    suppression_calls: list[str] = []

    class _SuppressionContext:
        def __enter__(self):
            suppression_calls.append("enter")

        def __exit__(self, exc_type, exc_val, exc_tb):
            suppression_calls.append("exit")
            return False

    monkeypatch.setattr(MLflowClientFactory, "create_client", lambda **_kwargs: mock_client)
    monkeypatch.setattr(MLflowClientFactory, "validate_client_connectivity", lambda _c: True)
    monkeypatch.setattr(
        "dlkit.engine.tracking.mlflow_resource_manager.suppress_mlflow_sqlite_bootstrap_logs",
        lambda: _SuppressionContext(),
    )

    with manager:
        pass

    assert suppression_calls == []
