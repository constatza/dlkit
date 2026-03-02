"""Unit tests for MLflow resource manager cleanup behavior."""

from __future__ import annotations

from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

import pytest

from dlkit.tools.config.mlflow_settings import (
    MLflowSettings,
    MLflowServerSettings,
    MLflowClientSettings,
)


@pytest.fixture
def mock_mlflow_client():
    """Create a mock MLflow client for testing."""
    mock_client = Mock()
    mock_client.search_experiments.return_value = []
    mock_client.get_experiment_by_name.return_value = None
    mock_client.create_experiment.return_value = "exp_1"
    mock_client.create_run.return_value = Mock(info=Mock(run_id="run_1", experiment_id="exp_1"))
    mock_client.set_terminated.return_value = None
    return mock_client


@pytest.fixture
def mlflow_config(tmp_path: Path) -> MLflowSettings:
    """Create test MLflow configuration with HTTP tracking."""
    return MLflowSettings(
        enabled=True,
        server=MLflowServerSettings(
            host="127.0.0.1",
            port=5678,
            backend_store_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
            artifacts_destination=f"file://{(tmp_path / 'artifacts').as_posix()}",
        ),
        client=MLflowClientSettings(
            tracking_uri="http://127.0.0.1:5678",
            experiment_name="test_experiment",
        ),
    )


class TestMLflowResourceManagerCleanup:
    """Test MLflow resource manager cleanup behavior."""

    def test_resource_manager_cleans_up_server_on_exit(
        self, mlflow_config: MLflowSettings, mock_mlflow_client: Mock
    ) -> None:
        """Ensure server context is properly cleaned up on resource manager exit."""
        from dlkit.runtime.workflows.strategies.tracking.mlflow_resource_manager import (
            MLflowResourceManager,
        )

        mock_server_context = Mock()
        mock_server_info = Mock(url="http://127.0.0.1:5678", pid=12345)
        mock_server_context.start_server.return_value = mock_server_info

        with (
            patch(
                "dlkit.runtime.workflows.strategies.tracking.mlflow_resource_manager.MLflowServerContext",
                return_value=mock_server_context,
            ),
            patch(
                "dlkit.runtime.workflows.strategies.tracking.mlflow_resource_manager.MLflowClientFactory"
            ) as mock_factory,
            patch(
                "dlkit.runtime.workflows.strategies.tracking.mlflow_resource_manager.mlflow"
            ) as mock_mlflow,
        ):
            mock_factory.create_client_from_server_info.return_value = mock_mlflow_client
            mock_factory.validate_client_connectivity.return_value = True
            mock_mlflow.active_run.return_value = None

            # Use context manager
            with MLflowResourceManager(mlflow_config) as manager:
                assert manager._state.server_context is mock_server_context
                assert manager._state.server_info is mock_server_info

            # After exit, cleanup should have been called
            mock_server_context.stop_server.assert_called_once()

    def test_resource_manager_handles_server_start_failure(
        self, mlflow_config: MLflowSettings
    ) -> None:
        """Ensure resource manager handles server startup failures gracefully."""
        from dlkit.runtime.workflows.strategies.tracking.mlflow_resource_manager import (
            MLflowResourceManager,
        )

        mock_server_context = Mock()
        mock_server_context.start_server.side_effect = RuntimeError("Server failed to start")

        with (
            patch(
                "dlkit.runtime.workflows.strategies.tracking.mlflow_resource_manager.MLflowServerContext",
                return_value=mock_server_context,
            ),
            pytest.raises(RuntimeError, match="Failed to initialize MLflow server"),
        ):
            with MLflowResourceManager(mlflow_config):
                pass

    def test_resource_manager_cleans_up_run_on_exit(
        self, mlflow_config: MLflowSettings, mock_mlflow_client: Mock
    ) -> None:
        """Ensure active run context is cleaned up on resource manager exit."""
        from dlkit.runtime.workflows.strategies.tracking.mlflow_resource_manager import (
            MLflowResourceManager,
        )

        mock_server_context = Mock()
        mock_server_info = Mock(url="http://127.0.0.1:5678", pid=12345)
        mock_server_context.start_server.return_value = mock_server_info

        with (
            patch(
                "dlkit.runtime.workflows.strategies.tracking.mlflow_resource_manager.MLflowServerContext",
                return_value=mock_server_context,
            ),
            patch(
                "dlkit.runtime.workflows.strategies.tracking.mlflow_resource_manager.MLflowClientFactory"
            ) as mock_factory,
            patch(
                "dlkit.runtime.workflows.strategies.tracking.mlflow_resource_manager.mlflow"
            ) as mock_mlflow,
        ):
            mock_factory.create_client_from_server_info.return_value = mock_mlflow_client
            mock_factory.validate_client_connectivity.return_value = True
            mock_factory.get_or_create_experiment.return_value = "exp_1"
            mock_mlflow.active_run.return_value = None

            with MLflowResourceManager(mlflow_config) as manager:
                # Create a run
                with manager.create_run(experiment_name="test"):
                    # Run is active (check stack has run)
                    assert len(manager._state.active_run_stack) > 0

            # Fluent run lifecycle should have been entered and stack must be clean.
            mock_mlflow.start_run.assert_called()
            assert len(manager._state.active_run_stack) == 0
            mock_server_context.stop_server.assert_called_once()

    def test_resource_manager_handles_cleanup_errors_gracefully(
        self, mlflow_config: MLflowSettings, mock_mlflow_client: Mock
    ) -> None:
        """Ensure resource manager continues cleanup even if some operations fail."""
        from dlkit.runtime.workflows.strategies.tracking.mlflow_resource_manager import (
            MLflowResourceManager,
        )

        mock_server_context = Mock()
        mock_server_info = Mock(url="http://127.0.0.1:5678", pid=12345)
        mock_server_context.start_server.return_value = mock_server_info
        # Make stop_server raise an exception
        mock_server_context.stop_server.side_effect = RuntimeError("Cleanup failed")

        with (
            patch(
                "dlkit.runtime.workflows.strategies.tracking.mlflow_resource_manager.MLflowServerContext",
                return_value=mock_server_context,
            ),
            patch(
                "dlkit.runtime.workflows.strategies.tracking.mlflow_resource_manager.MLflowClientFactory"
            ) as mock_factory,
            patch(
                "dlkit.runtime.workflows.strategies.tracking.mlflow_resource_manager.mlflow"
            ) as mock_mlflow,
        ):
            mock_factory.create_client_from_server_info.return_value = mock_mlflow_client
            mock_factory.validate_client_connectivity.return_value = True
            mock_mlflow.active_run.return_value = None

            # Should not raise even though cleanup fails
            with MLflowResourceManager(mlflow_config):
                pass

            # Cleanup should have been attempted
            mock_server_context.stop_server.assert_called_once()


class TestMLflowServerHealthChecking:
    """Test server health checking behavior."""

    def test_server_adapter_waits_for_api_readiness(self, tmp_path: Path) -> None:
        """Ensure server adapter waits for MLflow API to be ready, not just HTTP."""
        from dlkit.interfaces.servers.mlflow_adapter import MLflowServerAdapter
        from dlkit.interfaces.servers.protocols import ServerStatus
        from dlkit.tools.config.mlflow_settings import MLflowServerSettings

        config = MLflowServerSettings(
            host="127.0.0.1",
            port=5679,
            backend_store_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
            artifacts_destination=f"file://{(tmp_path / 'artifacts').as_posix()}",
        )

        mock_process = Mock(pid=12345)
        mock_process_manager = Mock()
        mock_process_manager.start_process.return_value = mock_process

        mock_health_checker = Mock()
        # First call: server not running
        # Second call (after start): HTTP is up
        # Third call (wait_for_health): API is ready
        mock_health_checker.check_health.side_effect = [
            ServerStatus(
                is_running=False,
                url="http://127.0.0.1:5679",
                response_time=None,
                error_message="Not running",
            ),
            ServerStatus(
                is_running=False,
                url="http://127.0.0.1:5679",
                response_time=None,
                error_message="Not running",
            ),
        ]
        mock_health_checker.wait_for_health.return_value = True

        adapter = MLflowServerAdapter(
            process_manager=mock_process_manager,
            health_checker=mock_health_checker,
        )

        with patch.object(adapter._storage_ensurer, "ensure_storage"):
            server_info = adapter.start_server(config)

        assert server_info.pid == 12345
        # wait_for_health should have been called to wait for API readiness
        mock_health_checker.wait_for_health.assert_called_once()

    def test_composite_health_checker_checks_all_endpoints(self) -> None:
        """Ensure composite health checker validates both HTTP and API endpoints."""
        from dlkit.interfaces.servers.health_checker import CompositeHealthChecker
        from dlkit.interfaces.servers.protocols import ServerStatus

        mock_http_checker = Mock()
        mock_api_checker = Mock()

        # HTTP passes
        mock_http_checker.check_health.return_value = ServerStatus(
            is_running=True,
            url="http://127.0.0.1:5000",
            response_time=0.1,
            error_message=None,
        )

        # API fails
        mock_api_checker.check_health.return_value = ServerStatus(
            is_running=False,
            url="http://127.0.0.1:5000",
            response_time=None,
            error_message="API not ready",
        )

        checker = CompositeHealthChecker(mock_http_checker, mock_api_checker)
        status = checker.check_health("http://127.0.0.1:5000")

        # Should fail because API is not ready
        assert status.is_running is False
        assert status.error_message == "API not ready"


class TestMLflowServerContextCleanup:
    """Test MLflow server context cleanup behavior."""

    def test_server_context_stops_server_on_exit(self, tmp_path: Path) -> None:
        """Ensure server context stops server when exiting context manager."""
        from dlkit.interfaces.servers.mlflow_adapter import MLflowServerContext
        from dlkit.tools.config.mlflow_settings import MLflowServerSettings

        config = MLflowServerSettings(
            host="127.0.0.1",
            port=5680,
            backend_store_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
            artifacts_destination=f"file://{(tmp_path / 'artifacts').as_posix()}",
        )

        mock_adapter = Mock()
        mock_server_info = Mock(url="http://127.0.0.1:5680", pid=12345)
        mock_adapter.start_server.return_value = mock_server_info
        mock_adapter.stop_server.return_value = True

        context = MLflowServerContext(config, adapter=mock_adapter)

        with context as server_info:
            assert server_info is mock_server_info
            mock_adapter.start_server.assert_called_once()

        # After exit, stop should have been called
        mock_adapter.stop_server.assert_called_once_with(mock_server_info)

    def test_server_context_handles_stop_failure_gracefully(self, tmp_path: Path) -> None:
        """Ensure server context handles stop failures without raising."""
        from dlkit.interfaces.servers.mlflow_adapter import MLflowServerContext
        from dlkit.tools.config.mlflow_settings import MLflowServerSettings

        config = MLflowServerSettings(
            host="127.0.0.1",
            port=5681,
            backend_store_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
            artifacts_destination=f"file://{(tmp_path / 'artifacts').as_posix()}",
        )

        mock_adapter = Mock()
        mock_server_info = Mock(url="http://127.0.0.1:5681", pid=12345)
        mock_adapter.start_server.return_value = mock_server_info
        mock_adapter.stop_server.side_effect = RuntimeError("Stop failed")

        context = MLflowServerContext(config, adapter=mock_adapter)

        # Should not raise even though stop fails
        with context:
            pass

        # Stop should have been attempted
        mock_adapter.stop_server.assert_called_once()
