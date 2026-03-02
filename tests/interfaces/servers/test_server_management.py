"""Tests for complete server management scenarios."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from dlkit.interfaces.servers.factory import create_mlflow_adapter
from dlkit.interfaces.servers.mlflow_adapter import MLflowServerAdapter
from dlkit.interfaces.servers.protocols import ServerStatus
from dlkit.tools.config.mlflow_settings import MLflowServerSettings


@pytest.fixture
def test_server_config(tmp_path: Path) -> MLflowServerSettings:
    """Create test server configuration."""
    return MLflowServerSettings(
        scheme="http",
        host="localhost",
        port=5555,  # Use non-standard port for tests
        backend_store_uri=f"sqlite:///{(tmp_path / 'test_mlflow.db').as_posix()}",
        artifacts_destination=str(tmp_path / "artifacts"),
        num_workers=1,
        keep_alive_interval=30,
        shutdown_timeout=1,
    )


class TestServerLifecycleManagement:
    """Test complete server lifecycle management scenarios."""

    @patch("dlkit.interfaces.servers.mlflow_adapter.SubprocessManager")
    @patch("dlkit.interfaces.servers.mlflow_adapter.HTTPHealthChecker")
    def test_complete_server_lifecycle(
        self,
        mock_health_checker_class: Mock,
        mock_process_manager_class: Mock,
        test_server_config: MLflowServerSettings,
    ) -> None:
        """Test complete server start → check → stop lifecycle."""
        # Setup mocks
        mock_process = Mock(pid=12345)
        mock_process_manager = Mock()
        mock_process_manager.start_process.return_value = mock_process
        mock_process_manager.stop_process.return_value = True
        mock_process_manager_class.return_value = mock_process_manager

        mock_health_checker = Mock()
        mock_health_checker_class.return_value = mock_health_checker

        # Create adapter
        adapter = MLflowServerAdapter()

        # Phase 1: Server not running initially
        mock_health_checker.check_health.return_value = ServerStatus(
            is_running=False,
            url="http://localhost:5555",
            response_time=None,
            error_message="Connection refused",
        )
        mock_health_checker.wait_for_health.return_value = True

        with patch.object(adapter._storage_ensurer, "ensure_storage"):
            # Start server
            server_info = adapter.start_server(test_server_config)

            assert server_info.url == "http://localhost:5555"
            assert server_info.host == "localhost"
            assert server_info.port == 5555
            assert server_info.pid == 12345
            assert server_info.process is mock_process

        # Phase 2: Server running and healthy
        mock_health_checker.check_health.return_value = ServerStatus(
            is_running=True, url="http://localhost:5555", response_time=0.12, error_message=None
        )

        status = adapter.check_server("localhost", 5555)
        assert status.is_running is True
        assert status.response_time == 0.12

        # Phase 3: Stop server
        result = adapter.stop_server(server_info)
        assert result is True

        mock_process_manager.start_process.assert_called_once()
        mock_process_manager.stop_process.assert_called_once_with(mock_process)

    def test_server_already_running_scenario(
        self, test_server_config: MLflowServerSettings
    ) -> None:
        """Test scenario where server is already running."""
        with (
            patch("dlkit.interfaces.servers.mlflow_adapter.SubprocessManager") as mock_pm_class,
            patch("dlkit.interfaces.servers.mlflow_adapter.HTTPHealthChecker") as mock_hc_class,
        ):
            mock_process_manager = Mock()
            mock_pm_class.return_value = mock_process_manager

            mock_health_checker = Mock()
            mock_health_checker.check_health.return_value = ServerStatus(
                is_running=True, url="http://localhost:5555", response_time=0.05, error_message=None
            )
            mock_hc_class.return_value = mock_health_checker

            adapter = MLflowServerAdapter()

            # Start server (should detect existing server)
            server_info = adapter.start_server(test_server_config)

            assert server_info.url == "http://localhost:5555"
            assert server_info.process is None  # Not started by us

            # Should not have attempted to start new process
            mock_process_manager.start_process.assert_not_called()

    def test_server_startup_failure_recovery(
        self, test_server_config: MLflowServerSettings
    ) -> None:
        """Test server startup failure and recovery."""
        with (
            patch("dlkit.interfaces.servers.mlflow_adapter.SubprocessManager") as mock_pm_class,
            patch("dlkit.interfaces.servers.mlflow_adapter.HTTPHealthChecker") as mock_hc_class,
        ):
            mock_process = Mock(pid=12345)
            mock_process_manager = Mock()
            mock_process_manager.start_process.return_value = mock_process
            mock_process_manager.stop_process.return_value = True
            mock_pm_class.return_value = mock_process_manager

            mock_health_checker = Mock()
            # Server not running initially
            mock_health_checker.check_health.return_value = ServerStatus(
                is_running=False,
                url="http://localhost:5555",
                response_time=None,
                error_message="Connection refused",
            )
            # Health check fails after startup
            mock_health_checker.wait_for_health.return_value = False
            mock_hc_class.return_value = mock_health_checker

            adapter = MLflowServerAdapter()

            with patch.object(adapter._storage_ensurer, "ensure_storage"):
                with pytest.raises(RuntimeError, match="MLflow server failed health check"):
                    adapter.start_server(test_server_config)

            # Should have attempted cleanup
            mock_process_manager.stop_process.assert_called_once_with(mock_process)

    def test_concurrent_server_management(self, test_server_config: MLflowServerSettings) -> None:
        """Test handling concurrent server management operations."""
        # This would test scenarios like:
        # - Multiple adapters trying to start server on same port
        # - Starting server while another is shutting down
        # - Health checking during server transitions

        # Setup mocks BEFORE creating adapters
        mock_process_manager1 = Mock()
        mock_process_manager1.start_process.return_value = Mock(pid=12345)
        mock_process_manager2 = Mock()

        # First adapter succeeds in starting server
        mock_health_checker1 = Mock()
        mock_health_checker1.check_health.return_value = ServerStatus(
            is_running=False,
            url="http://localhost:5555",
            response_time=None,
            error_message="Connection refused",
        )
        mock_health_checker1.wait_for_health.return_value = True

        # Second adapter sees server already running
        mock_health_checker2 = Mock()
        mock_health_checker2.check_health.return_value = ServerStatus(
            is_running=True, url="http://localhost:5555", response_time=0.1, error_message=None
        )

        # Create adapters with explicit mocks
        adapter1 = MLflowServerAdapter(
            process_manager=mock_process_manager1, health_checker=mock_health_checker1
        )
        adapter2 = MLflowServerAdapter(
            process_manager=mock_process_manager2, health_checker=mock_health_checker2
        )

        with (
            patch.object(adapter1._storage_ensurer, "ensure_storage"),
            patch.object(adapter2._storage_ensurer, "ensure_storage"),
        ):
            # First adapter starts server
            server_info1 = adapter1.start_server(test_server_config)
            assert server_info1.process is not None

            # Second adapter finds server already running
            server_info2 = adapter2.start_server(test_server_config)
            assert server_info2.process is None  # Not started by adapter2

        # Only first adapter should have started process
        mock_process_manager1.start_process.assert_called_once()
        mock_process_manager2.start_process.assert_not_called()


class TestServerConfigurationExtraction:
    """Test server configuration extraction and validation."""

    def test_server_management_service_creates_contexts(self) -> None:
        """Test that server management service can create contexts."""
        from dlkit.interfaces.servers import ServerManagementService
        from dlkit.tools.config.mlflow_settings import MLflowSettings, MLflowServerSettings
        from dlkit.tools.config.mlflow_settings import MLflowClientSettings

        service = ServerManagementService()

        mlflow_settings = MLflowSettings(
            enabled=True,
            server=MLflowServerSettings(
                host="0.0.0.0",
                port=6000,
                backend_store_uri="postgresql://user:pass@db:5432/mlflow",
                artifacts_destination="/shared/artifacts",
            ),
            client=MLflowClientSettings(tracking_uri="http://0.0.0.0:6000", experiment_name="test"),
        )

        server_context = service.create_server_context(mlflow_settings)

        # Context should be created successfully
        assert server_context is not None


class TestServerFactory:
    """Test server adapter factory functionality."""

    def test_create_mlflow_adapter(self) -> None:
        """Test creating MLflow adapter."""
        adapter = create_mlflow_adapter()

        assert isinstance(adapter, MLflowServerAdapter)
        assert adapter._scheme == "http"
        assert adapter._health_timeout > 0
        assert adapter._process_manager is not None
        assert adapter._health_checker is not None

    def test_create_mlflow_adapter_with_custom_timeout(self) -> None:
        """Test creating MLflow adapter with custom timeout."""
        adapter = create_mlflow_adapter(health_timeout=0.5)  # Fast for tests
        assert adapter._health_timeout == 0.5

    def test_factory_creates_independent_instances(self) -> None:
        """Test that factory creates independent adapter instances."""
        adapter1 = create_mlflow_adapter()
        adapter2 = create_mlflow_adapter()

        assert adapter1 is not adapter2
        assert adapter1._process_manager is not adapter2._process_manager
        assert adapter1._health_checker is not adapter2._health_checker


class TestServerErrorRecovery:
    """Test server error recovery scenarios."""

    def test_server_process_dies_during_operation(
        self, test_server_config: MLflowServerSettings
    ) -> None:
        """Test handling server process death during operation."""
        with (
            patch("dlkit.interfaces.servers.mlflow_adapter.SubprocessManager") as mock_pm_class,
            patch("dlkit.interfaces.servers.mlflow_adapter.HTTPHealthChecker") as mock_hc_class,
        ):
            mock_process = Mock(pid=12345)
            mock_process_manager = Mock()
            mock_process_manager.start_process.return_value = mock_process
            mock_pm_class.return_value = mock_process_manager

            mock_health_checker = Mock()
            mock_hc_class.return_value = mock_health_checker

            adapter = MLflowServerAdapter()

            # Start server successfully
            mock_health_checker.check_health.return_value = ServerStatus(
                is_running=False,
                url="http://localhost:5555",
                response_time=None,
                error_message="Connection refused",
            )
            mock_health_checker.wait_for_health.return_value = True

            with patch.object(adapter._storage_ensurer, "ensure_storage"):
                server_info = adapter.start_server(test_server_config)

            assert server_info.url == "http://localhost:5555"

            # Later, process dies and health check shows server down
            mock_health_checker.check_health.return_value = ServerStatus(
                is_running=False,
                url="http://localhost:5555",
                response_time=None,
                error_message="Connection refused",
            )

            status = adapter.check_server("localhost", 5555)
            assert status.is_running is False

    def test_server_stop_with_process_cleanup_failure(
        self, test_server_config: MLflowServerSettings
    ) -> None:
        """Test server stop when process cleanup fails."""
        with (
            patch("dlkit.interfaces.servers.mlflow_adapter.SubprocessManager") as mock_pm_class,
            patch("dlkit.interfaces.servers.mlflow_adapter.HTTPHealthChecker") as mock_hc_class,
        ):
            mock_process = Mock(pid=12345)
            mock_process_manager = Mock()
            mock_process_manager.start_process.return_value = mock_process
            mock_process_manager.stop_process.return_value = False  # Cleanup fails
            mock_pm_class.return_value = mock_process_manager

            mock_health_checker = Mock()
            mock_health_checker.check_health.return_value = ServerStatus(
                is_running=False,
                url="http://localhost:5555",
                response_time=None,
                error_message="Connection refused",
            )
            mock_health_checker.wait_for_health.return_value = True
            mock_hc_class.return_value = mock_health_checker

            adapter = MLflowServerAdapter()

            with patch.object(adapter._storage_ensurer, "ensure_storage"):
                server_info = adapter.start_server(test_server_config)

            # Stop should return False due to process cleanup failure (operational failure)
            result = adapter.stop_server(server_info)
            assert result is False

    def test_server_health_check_intermittent_failures(self) -> None:
        """Test handling intermittent health check failures."""
        with patch("dlkit.interfaces.servers.mlflow_adapter.HTTPHealthChecker") as mock_hc_class:
            mock_health_checker = Mock()
            mock_hc_class.return_value = mock_health_checker

            adapter = MLflowServerAdapter()

            # Simulate intermittent failures
            health_responses = [
                ServerStatus(
                    is_running=True,
                    url="http://localhost:5000",
                    response_time=0.1,
                    error_message=None,
                ),
                ServerStatus(
                    is_running=False,
                    url="http://localhost:5000",
                    response_time=None,
                    error_message="Timeout",
                ),
                ServerStatus(
                    is_running=True,
                    url="http://localhost:5000",
                    response_time=0.2,
                    error_message=None,
                ),
            ]

            mock_health_checker.check_health.side_effect = health_responses

            # Multiple checks should return different results
            status1 = adapter.check_server("localhost", 5000)
            assert status1.is_running is True

            status2 = adapter.check_server("localhost", 5000)
            assert status2.is_running is False

            status3 = adapter.check_server("localhost", 5000)
            assert status3.is_running is True


class TestServerConfigurationValidation:
    """Test server configuration validation scenarios."""

    def test_server_config_with_invalid_paths(self, tmp_path: Path) -> None:
        """Test server configuration with invalid paths."""
        # Test with non-existent parent directory
        invalid_path = tmp_path / "nonexistent" / "nested" / "path"

        config = MLflowServerSettings(
            host="localhost",
            port=5000,
            backend_store_uri=f"sqlite:///{invalid_path.as_posix()}/mlflow.db",
            artifacts_destination=str(invalid_path / "artifacts"),
        )

        adapter = MLflowServerAdapter()

        # storage_ensurer should handle creating directories
        with patch("dlkit.interfaces.servers.storage_ensurer.mkdir_for_local") as mock_mkdir:
            adapter._storage_ensurer.ensure_storage(config)
            # Should have called mkdir_for_local for paths that need it
            assert mock_mkdir.called

    def test_server_config_with_remote_storage(self) -> None:
        """Test server configuration with remote storage URIs."""
        config = MLflowServerSettings(
            host="localhost",
            port=5000,
            backend_store_uri="postgresql://user:pass@db:5432/mlflow",
            artifacts_destination="s3://mlflow-bucket/artifacts",
        )

        adapter = MLflowServerAdapter()

        # Remote URIs should not trigger local directory creation
        with patch("dlkit.interfaces.servers.storage_ensurer.mkdir_for_local") as mock_mkdir:
            adapter._storage_ensurer.ensure_storage(config)
            # Should not create directories for remote storage
            mock_mkdir.assert_not_called()

    def test_server_config_override_validation(
        self, test_server_config: MLflowServerSettings
    ) -> None:
        """Test server configuration override validation via config applier service."""
        from dlkit.interfaces.servers.config_applier import ServerConfigApplier

        # Test valid overrides
        valid_overrides = {
            "host": "0.0.0.0",
            "port": 8080,
            "backend_store_uri": "sqlite:///new_path.db",
        }

        updated_config = ServerConfigApplier.apply_overrides(test_server_config, valid_overrides)

        assert updated_config.host == "0.0.0.0"
        assert updated_config.port == 8080
        assert updated_config.backend_store_uri == "sqlite:///new_path.db"

        # Original config should be unchanged
        assert test_server_config.host == "localhost"
        assert test_server_config.port == 5555


class TestServerResourceManagement:
    """Test server resource management and cleanup."""

    def test_server_context_manager_resource_cleanup(self) -> None:
        """Test proper resource cleanup with context manager."""
        adapter = MLflowServerAdapter()

        # Test that context manager requires external configuration
        with pytest.raises(NotImplementedError):
            with adapter:
                pass

    def test_multiple_adapters_resource_isolation(self) -> None:
        """Test that multiple adapters maintain resource isolation."""
        adapter1 = MLflowServerAdapter(health_timeout=0.1)
        adapter2 = MLflowServerAdapter(health_timeout=0.2)

        # Adapters should be independent
        assert adapter1._health_timeout == 0.1
        assert adapter2._health_timeout == 0.2
        assert adapter1._process_manager is not adapter2._process_manager
        assert adapter1._health_checker is not adapter2._health_checker

    def test_adapter_cleanup_on_exception(self, test_server_config: MLflowServerSettings) -> None:
        """Test adapter cleanup when exceptions occur."""
        with (
            patch("dlkit.interfaces.servers.mlflow_adapter.SubprocessManager") as mock_pm_class,
            patch("dlkit.interfaces.servers.mlflow_adapter.HTTPHealthChecker") as mock_hc_class,
        ):
            mock_process = Mock(pid=12345)
            mock_process_manager = Mock()
            mock_process_manager.start_process.return_value = mock_process
            mock_process_manager.stop_process.return_value = True
            mock_pm_class.return_value = mock_process_manager

            mock_health_checker = Mock()
            mock_health_checker.check_health.return_value = ServerStatus(
                is_running=False,
                url="http://localhost:5555",
                response_time=None,
                error_message="Connection refused",
            )
            # Health check times out, triggering cleanup
            mock_health_checker.wait_for_health.return_value = False
            mock_hc_class.return_value = mock_health_checker

            adapter = MLflowServerAdapter()

            with patch.object(adapter._storage_ensurer, "ensure_storage"):
                with pytest.raises(RuntimeError, match="MLflow server failed health check"):
                    adapter.start_server(test_server_config)

            # Should have attempted cleanup even though startup failed
            mock_process_manager.stop_process.assert_called_once_with(mock_process)
