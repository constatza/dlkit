"""Tests for `MLflowResourceManager`'s connectivity-failure handling.

Connectivity is driven deterministically via `MLflowClientFactory
.validate_client_connectivity` (monkeypatched to a fixed True/False) rather
than real network I/O, and the "fallback_local" path's own `select_backend`
call is monkeypatched to a `tmp_path`-scoped `LocalSqliteBackend` rather than
letting it resolve the developer's real default local mlruns location.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlkit.common.errors import TrackingError
from dlkit.engine.tracking.backend import LocalSqliteBackend, RemoteServerBackend
from dlkit.engine.tracking.mlflow_client_factory import MLflowClientFactory
from dlkit.engine.tracking.mlflow_resource_manager import MLflowResourceManager
from dlkit.engine.tracking.mlflow_tracker import MLflowTracker
from dlkit.infrastructure.config.tracking_settings import TrackingSettings

UNREACHABLE_URI = "http://unreachable.invalid:5000"


@pytest.fixture
def connectivity_always_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every connectivity probe report failure, with no real network I/O."""
    monkeypatch.setattr(MLflowClientFactory, "validate_client_connectivity", lambda client: False)


@pytest.fixture
def fallback_backend(tmp_path: Path) -> LocalSqliteBackend:
    """The `tmp_path`-scoped backend `fallback_local` should land on."""
    return LocalSqliteBackend(db_path=tmp_path / "fallback" / "mlflow.db")


@pytest.fixture
def force_fallback_to(
    monkeypatch: pytest.MonkeyPatch, fallback_backend: LocalSqliteBackend
) -> LocalSqliteBackend:
    """Make the resource manager's fallback `select_backend()` call return
    `fallback_backend` instead of resolving a real, unscoped default location.
    """
    monkeypatch.setattr(
        "dlkit.engine.tracking.mlflow_resource_manager.select_backend",
        lambda **kwargs: fallback_backend,
    )
    return fallback_backend


def test_raises_when_local_backend_itself_is_unreachable(
    tmp_path: Path, connectivity_always_fails: None
) -> None:
    """Already on local SQLite with no connectivity: there is no fallback target,
    so this raises regardless of `on_connectivity_failure`.
    """
    backend = LocalSqliteBackend(db_path=tmp_path / "mlruns" / "mlflow.db")
    config = TrackingSettings(backend="mlflow", on_connectivity_failure="fallback_local")
    manager = MLflowResourceManager(config, backend)

    with pytest.raises(TrackingError), manager:
        pass


def test_raises_by_default_when_remote_backend_unreachable(
    connectivity_always_fails: None,
) -> None:
    backend = RemoteServerBackend(uri=UNREACHABLE_URI)
    config = TrackingSettings(backend="mlflow", uri=UNREACHABLE_URI)  # default: "raise"
    manager = MLflowResourceManager(config, backend)

    with pytest.raises(TrackingError), manager:
        pass


def test_raises_when_config_is_none_and_backend_unreachable(
    connectivity_always_fails: None,
) -> None:
    """No config at all still fails safe -- `None` defaults to `"raise"`, not a
    silent continue.
    """
    backend = RemoteServerBackend(uri=UNREACHABLE_URI)
    manager = MLflowResourceManager(None, backend)

    with pytest.raises(TrackingError), manager:
        pass


def test_falls_back_to_local_backend_when_configured(
    connectivity_always_fails: None, force_fallback_to: LocalSqliteBackend
) -> None:
    backend = RemoteServerBackend(uri=UNREACHABLE_URI)
    config = TrackingSettings(
        backend="mlflow", uri=UNREACHABLE_URI, on_connectivity_failure="fallback_local"
    )
    manager = MLflowResourceManager(config, backend)

    with manager:
        assert isinstance(manager.backend, LocalSqliteBackend)
        assert manager.get_tracking_uri() == force_fallback_to.tracking_uri()


def test_run_created_after_fallback_is_tagged_degraded(
    connectivity_always_fails: None, force_fallback_to: LocalSqliteBackend
) -> None:
    backend = RemoteServerBackend(uri=UNREACHABLE_URI)
    config = TrackingSettings(
        backend="mlflow", uri=UNREACHABLE_URI, on_connectivity_failure="fallback_local"
    )
    manager = MLflowResourceManager(config, backend)

    with manager:
        with manager.create_run(experiment_name="degraded-run-test") as run:
            run_id = run.run_id

    client = MLflowClientFactory.create_client(tracking_uri=force_fallback_to.tracking_uri())
    tags = client.get_run(run_id).data.tags
    assert tags["tracking.degraded_fallback"] == "true"
    assert UNREACHABLE_URI in tags["tracking.degraded_reason"]


def test_run_created_without_fallback_is_not_tagged_degraded(tmp_path: Path) -> None:
    """A healthy local backend never needed a fallback -- no degraded tags at all."""
    backend = LocalSqliteBackend(db_path=tmp_path / "mlruns" / "mlflow.db")
    config = TrackingSettings(backend="mlflow")
    manager = MLflowResourceManager(config, backend)

    with manager:
        with manager.create_run(experiment_name="healthy-run-test") as run:
            run_id = run.run_id

    client = MLflowClientFactory.create_client(tracking_uri=backend.tracking_uri())
    tags = client.get_run(run_id).data.tags
    assert "tracking.degraded_fallback" not in tags


def test_tracker_reports_effective_backend_after_fallback(
    connectivity_always_fails: None, force_fallback_to: LocalSqliteBackend
) -> None:
    """`MLflowTracker.get_tracking_uri()`/`is_local()` must reflect the resource
    manager's post-fallback backend, not the pre-entry one selected in
    `MLflowTracker.__enter__` -- otherwise a caller would be told runs are
    landing on the unreachable remote server when they're actually local.
    """
    tracker = MLflowTracker()
    tracker.configure(
        TrackingSettings(
            backend="mlflow", uri=UNREACHABLE_URI, on_connectivity_failure="fallback_local"
        )
    )

    with tracker:
        assert tracker.is_local() is True
        assert tracker.get_tracking_uri() == force_fallback_to.tracking_uri()
        assert tracker.get_tracking_uri() != UNREACHABLE_URI
