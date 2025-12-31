"""Tests for MLflow server adapter functionality."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from dlkit.interfaces.servers.mlflow_adapter import MLflowServerAdapter
from dlkit.interfaces.servers.protocols import ServerInfo, ServerStatus
from dlkit.tools.config.mlflow_settings import MLflowServerSettings


@pytest.fixture
def mock_process_manager() -> Mock:
    """Create mock process manager."""
    manager = Mock()
    manager.start_process.return_value = Mock(pid=12345)
    manager.stop_process.return_value = True
    return manager


@pytest.fixture
def mock_health_checker() -> Mock:
    """Create mock health checker."""
    checker = Mock()
    checker.check_health.return_value = ServerStatus(
        is_running=True, url="http://localhost:5000", response_time=0.1, error_message=None
    )
    checker.wait_for_health.return_value = True
    return checker


@pytest.fixture
def mlflow_server_config(tmp_path_factory: pytest.TempPathFactory) -> MLflowServerSettings:
    """Create MLflow server configuration."""
    artifacts_dir = tmp_path_factory.mktemp("mlflow_adapter") / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return MLflowServerSettings(
        scheme="http",
        host="localhost",
        port=5000,
        backend_store_uri="sqlite:///test.db",
        artifacts_destination=artifacts_dir.resolve().as_uri(),
    )


@pytest.fixture
def mlflow_adapter(mock_process_manager: Mock, mock_health_checker: Mock) -> MLflowServerAdapter:
    """Create MLflow server adapter with mocked dependencies."""
    return MLflowServerAdapter(
        process_manager=mock_process_manager,
        health_checker=mock_health_checker,
        scheme="http",
        health_timeout=0.1,  # Very fast for tests
        request_timeout=0.05,
        poll_interval=0.01,
    )


class TestMLflowServerAdapterInit:
    """Test MLflow server adapter initialization."""

    def test_init_with_defaults(self) -> None:
        """Test adapter initialization with default parameters."""
        adapter = MLflowServerAdapter()

        assert adapter._scheme == "http"
        # Don't check exact timeout as it may be overridden by environment variables
        assert adapter._health_timeout > 0
        assert adapter._process_manager is not None
        assert adapter._health_checker is not None
        assert adapter._current_server_info is None

    def test_init_with_custom_parameters(
        self, mock_process_manager: Mock, mock_health_checker: Mock
    ) -> None:
        """Test adapter initialization with custom parameters."""
        adapter = MLflowServerAdapter(
            process_manager=mock_process_manager,
            health_checker=mock_health_checker,
            scheme="https",
            health_timeout=0.5,  # Fast for tests
        )

        assert adapter._scheme == "https"
        assert adapter._health_timeout == 0.5
        assert adapter._process_manager is mock_process_manager
        assert adapter._health_checker is mock_health_checker

    def test_init_uses_default_small_timeout(self) -> None:
        """Adapter should use MLflowServerSettings default timeout (single source of truth)."""
        adapter = MLflowServerAdapter()
        assert adapter._health_timeout == 30.0  # From MLflowServerSettings default

    def test_init_with_custom_timeout_parameter(self) -> None:
        """Explicit timeout parameter should be respected."""
        adapter = MLflowServerAdapter(health_timeout=0.3)  # Fast for tests
        assert adapter._health_timeout == 0.3


class TestMLflowServerAdapterStartServer:
    """Test MLflow server start functionality."""

    def test_start_server_successful(
        self,
        mlflow_adapter: MLflowServerAdapter,
        mlflow_server_config: MLflowServerSettings,
        mock_process_manager: Mock,
        mock_health_checker: Mock,
    ) -> None:
        """Test successful server startup."""
        # Mock server not already running
        mock_health_checker.check_health.return_value = ServerStatus(
            is_running=False,
            url="http://localhost:5000",
            response_time=None,
            error_message="Connection refused",
        )

        with patch.object(mlflow_adapter._storage_ensurer, "ensure_storage") as mock_ensure:
            server_info = mlflow_adapter.start_server(mlflow_server_config)

        assert isinstance(server_info, ServerInfo)
        assert server_info.url == "http://localhost:5000"
        assert server_info.host == "localhost"
        assert server_info.port == 5000
        assert server_info.pid == 12345
        assert server_info.process is not None

        expected_config = mlflow_adapter._config_normalizer.normalize(mlflow_server_config)
        mock_ensure.assert_called_once_with(expected_config)
        mock_process_manager.start_process.assert_called_once_with(expected_config)
        mock_health_checker.wait_for_health.assert_called_once_with(
            "http://localhost:5000", timeout=0.1
        )

    def test_start_server_already_running(
        self,
        mlflow_adapter: MLflowServerAdapter,
        mlflow_server_config: MLflowServerSettings,
        mock_process_manager: Mock,
        mock_health_checker: Mock,
    ) -> None:
        """Test starting server when already running."""
        # Mock server already running
        mock_health_checker.check_health.return_value = ServerStatus(
            is_running=True, url="http://localhost:5000", response_time=0.1, error_message=None
        )

        server_info = mlflow_adapter.start_server(mlflow_server_config)

        assert server_info.url == "http://localhost:5000"
        assert server_info.process is None  # Not started by us

        # Should not attempt to start new process
        mock_process_manager.start_process.assert_not_called()

    def test_start_server_with_overrides(
        self,
        mlflow_adapter: MLflowServerAdapter,
        mlflow_server_config: MLflowServerSettings,
        mock_health_checker: Mock,
    ) -> None:
        """Test starting server with configuration overrides."""
        # Mock server not running
        mock_health_checker.check_health.return_value = ServerStatus(
            is_running=False,
            url="http://localhost:8080",
            response_time=None,
            error_message="Connection refused",
        )

        overrides = {"host": "0.0.0.0", "port": 8080}

        with patch.object(mlflow_adapter._config_applier, "apply_overrides") as mock_apply:
            mock_apply.return_value = mlflow_server_config.model_copy(update=overrides)
            with patch.object(mlflow_adapter._storage_ensurer, "ensure_storage"):
                server_info = mlflow_adapter.start_server(mlflow_server_config, **overrides)

            assert server_info.host == overrides["host"]
            assert server_info.port == overrides["port"]
            mock_apply.assert_called_once_with(mlflow_server_config, overrides)

    def test_start_server_health_check_fails(
        self,
        mlflow_adapter: MLflowServerAdapter,
        mlflow_server_config: MLflowServerSettings,
        mock_process_manager: Mock,
        mock_health_checker: Mock,
    ) -> None:
        """Test server start failure when health check fails."""
        # Mock server not running initially
        mock_health_checker.check_health.return_value = ServerStatus(
            is_running=False,
            url="http://localhost:5000",
            response_time=None,
            error_message="Connection refused",
        )
        # Mock health check failure after start
        mock_health_checker.wait_for_health.return_value = False

        with patch.object(mlflow_adapter._storage_ensurer, "ensure_storage"):
            with pytest.raises(RuntimeError, match="MLflow server failed health check"):
                mlflow_adapter.start_server(mlflow_server_config)

        # Should attempt to stop the failed process
        mock_process_manager.stop_process.assert_called_once()

    def test_start_server_process_start_fails(
        self,
        mlflow_adapter: MLflowServerAdapter,
        mlflow_server_config: MLflowServerSettings,
        mock_process_manager: Mock,
        mock_health_checker: Mock,
    ) -> None:
        """Test server start failure when process start fails."""
        # Mock server not running
        mock_health_checker.check_health.return_value = ServerStatus(
            is_running=False,
            url="http://localhost:5000",
            response_time=None,
            error_message="Connection refused",
        )
        # Mock process start failure
        mock_process_manager.start_process.side_effect = RuntimeError("Process failed to start")

        with patch.object(mlflow_adapter._storage_ensurer, "ensure_storage"):
            with pytest.raises(RuntimeError, match="Failed to start MLflow server"):
                mlflow_adapter.start_server(mlflow_server_config)


class TestMLflowServerAdapterStopServer:
    """Test MLflow server stop functionality."""

    def test_stop_server_successful(
        self, mlflow_adapter: MLflowServerAdapter, mock_process_manager: Mock
    ) -> None:
        """Test successful server shutdown."""
        mock_process = Mock(pid=12345)
        server_info = ServerInfo(
            process=mock_process,
            url="http://localhost:5000",
            host="localhost",
            port=5000,
            pid=12345,
        )

        result = mlflow_adapter.stop_server(server_info)

        assert result is True
        mock_process_manager.stop_process.assert_called_once_with(mock_process)

    def test_stop_server_no_process_handle(self, mlflow_adapter: MLflowServerAdapter) -> None:
        """Test stopping server without process handle (idempotent - already stopped)."""
        server_info = ServerInfo(
            process=None, url="http://localhost:5000", host="localhost", port=5000
        )

        # Should return True (idempotent - server already stopped or externally managed)
        result = mlflow_adapter.stop_server(server_info)
        assert result is True

    def test_stop_server_process_stop_fails(
        self, mlflow_adapter: MLflowServerAdapter, mock_process_manager: Mock
    ) -> None:
        """Test server stop failure when process stop fails (operational failure)."""
        mock_process = Mock(pid=12345)
        server_info = ServerInfo(
            process=mock_process,
            url="http://localhost:5000",
            host="localhost",
            port=5000,
            pid=12345,
        )

        mock_process_manager.stop_process.return_value = False

        # Should return False (operational failure - process exists but won't stop)
        result = mlflow_adapter.stop_server(server_info)
        assert result is False

    def test_stop_server_exception_handling(
        self, mlflow_adapter: MLflowServerAdapter, mock_process_manager: Mock
    ) -> None:
        """Test server stop exception handling."""
        mock_process = Mock(pid=12345)
        server_info = ServerInfo(
            process=mock_process,
            url="http://localhost:5000",
            host="localhost",
            port=5000,
            pid=12345,
        )

        mock_process_manager.stop_process.side_effect = Exception("Unexpected error")

        with pytest.raises(RuntimeError, match="Failed to stop MLflow server"):
            mlflow_adapter.stop_server(server_info)


class TestMLflowServerAdapterCheckServer:
    """Test MLflow server status checking functionality."""

    def test_check_server_successful(
        self, mlflow_adapter: MLflowServerAdapter, mock_health_checker: Mock
    ) -> None:
        """Test successful server status check."""
        expected_status = ServerStatus(
            is_running=True, url="http://localhost:5000", response_time=0.15, error_message=None
        )
        mock_health_checker.check_health.return_value = expected_status

        status = mlflow_adapter.check_server("localhost", 5000)

        assert status is expected_status
        mock_health_checker.check_health.assert_called_once_with("http://localhost:5000")

    def test_check_server_with_custom_host_port(
        self, mlflow_adapter: MLflowServerAdapter, mock_health_checker: Mock
    ) -> None:
        """Test server status check with custom host and port."""
        status = mlflow_adapter.check_server("192.168.1.100", 8080)

        assert isinstance(status, ServerStatus)
        mock_health_checker.check_health.assert_called_once_with("http://192.168.1.100:8080")

    def test_check_server_exception_handling(
        self, mlflow_adapter: MLflowServerAdapter, mock_health_checker: Mock
    ) -> None:
        """Test server check exception handling."""
        mock_health_checker.check_health.side_effect = Exception("Network error")

        with pytest.raises(RuntimeError, match="Failed to check server status"):
            mlflow_adapter.check_server("localhost", 5000)


class TestMLflowServerAdapterUtility:
    """Test MLflow server adapter utility methods."""

    def test_get_server_url(self, mlflow_adapter: MLflowServerAdapter) -> None:
        """Test server URL construction."""
        url = mlflow_adapter.get_server_url("localhost", 5000)
        assert url == "http://localhost:5000"

    def test_get_server_url_with_https_scheme(self) -> None:
        """Test server URL construction with HTTPS scheme."""
        adapter = MLflowServerAdapter(scheme="https")
        url = adapter.get_server_url("mlflow.example.com", 443)
        assert url == "https://mlflow.example.com:443"

    @patch("dlkit.interfaces.servers.storage_ensurer.mkdir_for_local")
    def test_ensure_local_storage(
        self,
        mock_mkdir: Mock,
        mlflow_adapter: MLflowServerAdapter,
        mlflow_server_config: MLflowServerSettings,
    ) -> None:
        """Test local storage directory creation via storage ensurer service."""
        mlflow_adapter._storage_ensurer.ensure_storage(mlflow_server_config)

        # Should create directories for local paths
        calls = mock_mkdir.call_args_list
        assert len(calls) >= 1  # At least artifacts destination

    def test_apply_server_overrides(self, mlflow_server_config: MLflowServerSettings) -> None:
        """Test server configuration override application via config applier service."""
        from dlkit.interfaces.servers.config_applier import ServerConfigApplier

        overrides = {
            "host": "0.0.0.0",
            "port": 8080,
            "backend_store_uri": "postgresql://user:pass@localhost/mlflow",
        }

        updated_config = ServerConfigApplier.apply_overrides(mlflow_server_config, overrides)

        assert updated_config.host == "0.0.0.0"
        assert updated_config.port == 8080
        assert updated_config.backend_store_uri == "postgresql://user:pass@localhost/mlflow"
        assert updated_config.artifacts_destination == mlflow_server_config.artifacts_destination


class TestMLflowServerAdapterContextManager:
    """Test MLflow server adapter context manager functionality."""

    def test_context_manager_not_implemented(self, mlflow_adapter: MLflowServerAdapter) -> None:
        """Test context manager raises NotImplementedError without settings."""
        with pytest.raises(NotImplementedError, match="Context manager requires settings"):
            with mlflow_adapter:
                pass

    def test_context_manager_already_started(self, mlflow_adapter: MLflowServerAdapter) -> None:
        """Test context manager with server already started."""
        # Simulate server already started
        mlflow_adapter._current_server_info = ServerInfo(
            process=Mock(), url="http://localhost:5000", host="localhost", port=5000
        )

        with pytest.raises(RuntimeError, match="Server already started in this context"):
            with mlflow_adapter:
                pass

    def test_context_manager_exit_cleanup(self, mlflow_adapter: MLflowServerAdapter) -> None:
        """Test context manager cleanup on exit."""
        mock_server_info = ServerInfo(
            process=Mock(), url="http://localhost:5000", host="localhost", port=5000
        )
        mlflow_adapter._current_server_info = mock_server_info

        with patch.object(mlflow_adapter, "stop_server") as mock_stop:
            mlflow_adapter.__exit__(None, None, None)
            mock_stop.assert_called_once_with(mock_server_info)

        assert mlflow_adapter._current_server_info is None

    def test_context_manager_exit_with_stop_error(
        self, mlflow_adapter: MLflowServerAdapter
    ) -> None:
        """Test context manager cleanup handles stop errors gracefully."""
        mock_server_info = ServerInfo(
            process=Mock(), url="http://localhost:5000", host="localhost", port=5000
        )
        mlflow_adapter._current_server_info = mock_server_info

        with patch.object(mlflow_adapter, "stop_server", side_effect=Exception("Stop failed")):
            # Should not raise exception, just log error
            mlflow_adapter.__exit__(None, None, None)

        assert mlflow_adapter._current_server_info is None


class TestMLflowServerAdapterIntegration:
    """Integration tests for MLflow server adapter."""

    def test_full_server_lifecycle_mocked(
        self,
        mlflow_adapter: MLflowServerAdapter,
        mlflow_server_config: MLflowServerSettings,
        mock_health_checker: Mock,
    ) -> None:
        """Test complete server lifecycle with mocked components."""
        # Mock server not running initially
        mock_health_checker.check_health.return_value = ServerStatus(
            is_running=False,
            url="http://localhost:5000",
            response_time=None,
            error_message="Connection refused",
        )

        with patch.object(mlflow_adapter._storage_ensurer, "ensure_storage"):
            # Start server
            server_info = mlflow_adapter.start_server(mlflow_server_config)
            assert server_info.url == "http://localhost:5000"

            # Check server status
            mock_health_checker.check_health.return_value = ServerStatus(
                is_running=True, url="http://localhost:5000", response_time=0.1, error_message=None
            )
            status = mlflow_adapter.check_server("localhost", 5000)
            assert status.is_running is True

            # Stop server
            result = mlflow_adapter.stop_server(server_info)
            assert result is True
