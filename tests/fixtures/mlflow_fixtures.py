"""Pytest fixtures for MLflow resource management and test isolation."""

import mlflow
import pytest
from pathlib import Path
from typing import Generator
from unittest.mock import Mock

from dlkit.runtime.workflows.strategies.tracking.mlflow_resource_manager import MLflowResourceManager
from dlkit.runtime.workflows.strategies.tracking.mlflow_client_factory import MLflowClientFactory
from dlkit.tools.config.mlflow_settings import MLflowSettings, MLflowClientSettings, MLflowServerSettings


@pytest.fixture(autouse=True)
def mlflow_global_state_isolation(tmp_path: Path) -> Generator[None, None, None]:
    """Automatically isolate MLflow global state between tests.

    This fixture runs before and after each test to ensure clean state.
    MLflow 3.x changed the default tracking URI to ``sqlite:///mlflow.db``
    (relative to CWD).  Without an explicit override, any test that touches
    the tracking store without a URI would create that file in the project
    root.  We redirect the default to an isolated per-test path in
    ``tmp_path`` so all artifacts stay inside pytest's temporary tree.
    """
    isolation_uri = f"sqlite:///{(tmp_path / 'mlflow_isolation.db').as_posix()}"

    # Reset state before test, then pin the default to tmp_path
    MLflowResourceManager.reset_global_state()
    mlflow.set_tracking_uri(isolation_uri)

    yield

    # Reset state after test, then pin again so teardown doesn't leave a
    # CWD-relative URI behind for the next test's setup phase
    MLflowResourceManager.reset_global_state()
    mlflow.set_tracking_uri(isolation_uri)


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
    """Provide test MLflow settings without server auto-start."""
    return MLflowSettings(
        enabled=True,
        client=MLflowClientSettings(
            tracking_uri="http://localhost:5000",
            experiment_name="test_experiment",
            run_name="test_run",
        ),
        # No server settings - prevents auto-start
        server=None,
    )


@pytest.fixture
def mlflow_server_test_settings():
    """Provide test MLflow settings with server configuration."""
    return MLflowSettings(
        enabled=True,
        client=MLflowClientSettings(
            tracking_uri="http://localhost:5000",
            experiment_name="test_experiment",
            run_name="test_run",
        ),
        server=MLflowServerSettings(
            host="127.0.0.1",
            port=5000,
            backend_store_uri=None,
            artifacts_destination=None,
            health_timeout=10.0,
            request_timeout=2.0,
            poll_interval=0.5,
        ),
    )


@pytest.fixture
def mlflow_resource_manager(mlflow_test_settings) -> Generator[MLflowResourceManager, None, None]:
    """Provide a properly managed MLflow resource manager for testing.

    This fixture ensures proper setup and cleanup of MLflow resources.
    """
    manager = MLflowResourceManager(mlflow_test_settings)

    try:
        with manager:
            yield manager
    except Exception as e:
        # Ensure cleanup even if test fails
        try:
            manager.__exit__(None, None, None)
        except Exception:
            pass
        raise e


@pytest.fixture
def mock_mlflow_resource_manager(mock_mlflow_client, mlflow_test_settings):
    """Provide a mocked MLflow resource manager for unit testing."""
    manager = MLflowResourceManager(mlflow_test_settings)

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
    tracker = MLflowTracker(disable_autostart=True, skip_health_checks=True)

    try:
        yield tracker
    finally:
        # Ensure cleanup
        try:
            tracker.cleanup_server()
        except Exception:
            pass


@pytest.fixture(scope="session")
def subprocess_manager_cleanup():
    """Session-scoped fixture to ensure subprocess manager cleanup.

    This helps prevent process leaks between test sessions.
    """
    from dlkit.interfaces.servers.process_manager import SubprocessManager

    # Keep track of all managers created during tests
    managers = []

    original_init = SubprocessManager.__init__

    def tracked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        managers.append(self)

    SubprocessManager.__init__ = tracked_init

    try:
        yield
    finally:
        # Cleanup all managers
        for manager in managers:
            try:
                manager.cleanup_all_processes()
            except Exception:
                pass

        # Restore original init
        SubprocessManager.__init__ = original_init


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
                ResourceWarning
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
                ResourceWarning
            )


@pytest.fixture
def resource_leak_detection(process_leak_detector, thread_leak_detector):
    """Combined resource leak detection for comprehensive monitoring."""
    yield