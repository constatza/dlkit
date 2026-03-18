"""Pytest fixtures for MLflow resource management and test isolation."""

import os
import mlflow
import pytest
from pathlib import Path
from typing import Generator
from unittest.mock import Mock

from dlkit.runtime.workflows.strategies.tracking.mlflow_resource_manager import (
    MLflowResourceManager,
)
from dlkit.runtime.workflows.strategies.tracking.mlflow_client_factory import MLflowClientFactory
from dlkit.tools.config.mlflow_settings import MLflowSettings


@pytest.fixture(autouse=True)
def mlflow_global_state_isolation(tmp_path: Path) -> Generator[None, None, None]:
    """Automatically isolate MLflow global state between tests.

    This fixture runs before and after each test to ensure clean state.
    MLflow 3.x changed the default tracking URI to ``sqlite:///mlflow.db``
    (relative to CWD).  Without an explicit override, any test that touches
    the tracking store without a URI would create that file in the project
    root.  We redirect the default to an isolated per-test path in
    ``tmp_path`` so all artifacts stay inside pytest's temporary tree.

    Both ``mlflow.set_tracking_uri`` and ``os.environ["MLFLOW_TRACKING_URI"]``
    are set so that ``select_backend()`` (which honours SQLite env vars) also
    resolves to the isolated path.

    Note:
        In MLflow 3.x, ``mlflow.set_tracking_uri(None)`` does **not** fall back
        to the ``MLFLOW_TRACKING_URI`` environment variable — it resets the
        internal state to the CWD-relative default (``sqlite:///mlflow.db``),
        which would create a stray DB in the project root.  This fixture
        therefore manages both the internal MLflow URI and the env var together,
        and re-pins them after every ``reset_global_state()`` call.
    """
    isolation_uri = f"sqlite:///{(tmp_path / 'mlflow_isolation.db').as_posix()}"
    _original_uri = os.environ.get("MLFLOW_TRACKING_URI")

    # Setup: pin env var + internal URI, reset any stale state, re-pin.
    os.environ["MLFLOW_TRACKING_URI"] = isolation_uri
    mlflow.set_tracking_uri(isolation_uri)
    MLflowResourceManager.reset_global_state()
    # reset_global_state() now preserves the env var, but re-pin to be explicit.
    os.environ["MLFLOW_TRACKING_URI"] = isolation_uri
    mlflow.set_tracking_uri(isolation_uri)

    yield

    # Teardown: same order — pin → reset → re-pin.
    os.environ["MLFLOW_TRACKING_URI"] = isolation_uri
    mlflow.set_tracking_uri(isolation_uri)
    MLflowResourceManager.reset_global_state()
    os.environ["MLFLOW_TRACKING_URI"] = isolation_uri
    mlflow.set_tracking_uri(isolation_uri)

    # Restore original env state.
    if _original_uri is None:
        os.environ.pop("MLFLOW_TRACKING_URI", None)
    else:
        os.environ["MLFLOW_TRACKING_URI"] = _original_uri


@pytest.fixture
def mock_mlflow_client():
    """Provide a mock MLflow client for testing."""
    mock_client = Mock()
    mock_client.search_experiments.return_value = []
    mock_client.create_experiment.return_value = "test_experiment_id"
    mock_client.get_experiment_by_name.return_value = None
    mock_client.create_run.return_value = Mock(run_id="test_run_id")
    mock_client.log_metric.return_value = None
    mock_client.log_param.return_value = None
    mock_client.log_artifact.return_value = None
    mock_client.set_tag.return_value = None
    mock_client.set_terminated.return_value = None
    return mock_client


@pytest.fixture
def mlflow_test_settings():
    """Provide test MLflow settings."""
    return MLflowSettings(
        experiment_name="test_experiment",
        run_name="test_run",
    )


@pytest.fixture
def mlflow_resource_manager(mlflow_test_settings, tmp_path: Path) -> Generator[MLflowResourceManager, None, None]:
    """Provide a properly managed MLflow resource manager for testing.

    This fixture ensures proper setup and cleanup of MLflow resources.
    """
    from dlkit.runtime.workflows.strategies.tracking.backend import LocalSqliteBackend

    db_path = tmp_path / "mlruns" / "mlflow.db"
    backend = LocalSqliteBackend(db_path=db_path)
    with MLflowResourceManager(mlflow_test_settings, backend) as manager:
        yield manager


@pytest.fixture
def mock_mlflow_resource_manager(mock_mlflow_client, mlflow_test_settings, tmp_path: Path):
    """Provide a mocked MLflow resource manager for unit testing."""
    from dlkit.runtime.workflows.strategies.tracking.backend import LocalSqliteBackend

    db_path = tmp_path / "mlruns" / "mlflow.db"
    backend = LocalSqliteBackend(db_path=db_path)
    manager = MLflowResourceManager(mlflow_test_settings, backend)

    # Mock the client creation to return our mock
    original_create_client = MLflowClientFactory.create_client
    MLflowClientFactory.create_client = lambda *args, **kwargs: mock_mlflow_client

    try:
        with manager:
            yield manager
    finally:
        # Restore original method
        MLflowClientFactory.create_client = original_create_client


@pytest.fixture
def isolated_mlflow_tracker():
    """Provide an MLflow tracker with proper isolation for testing."""
    from dlkit.runtime.workflows.strategies.tracking.mlflow_tracker import MLflowTracker

    # Create tracker with autostart disabled for testing
    tracker = MLflowTracker(disable_autostart=True)

    try:
        yield tracker
    finally:
        # Ensure tracker context is closed if a test left it open
        try:
            tracker.__exit__(None, None, None)
        except Exception:
            pass

@pytest.fixture
def process_leak_detector():
    """Detect and report process leaks during tests."""
    import psutil
    import os

    # Get initial process count
    initial_processes = set()
    try:
        current_process = psutil.Process(os.getpid())
        initial_processes = {p.pid for p in current_process.children(recursive=True)}
    except Exception:
        pass

    yield

    # Check for leaked processes
    try:
        current_process = psutil.Process(os.getpid())
        final_processes = {p.pid for p in current_process.children(recursive=True)}
        leaked_processes = final_processes - initial_processes

        if leaked_processes:
            import warnings

            warnings.warn(
                f"Detected {len(leaked_processes)} potentially leaked child processes: {leaked_processes}",
                ResourceWarning,
            )
    except Exception:
        pass


@pytest.fixture
def thread_leak_detector():
    """Detect and report thread leaks during tests."""
    import threading

    initial_thread_count = threading.active_count()
    initial_threads = set(threading.enumerate())

    yield

    final_thread_count = threading.active_count()
    final_threads = set(threading.enumerate())

    if final_thread_count > initial_thread_count:
        leaked_threads = final_threads - initial_threads
        thread_names = [t.name for t in leaked_threads if t.is_alive()]

        if thread_names:
            import warnings

            warnings.warn(
                f"Detected {len(thread_names)} potentially leaked threads: {thread_names}",
                ResourceWarning,
            )


@pytest.fixture
def resource_leak_detection(process_leak_detector, thread_leak_detector):
    """Combined resource leak detection for comprehensive monitoring."""
    yield
