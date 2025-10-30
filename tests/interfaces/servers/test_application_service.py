"""Tests for server application service."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import pytest

from dlkit.interfaces.servers.application_service import ServerApplicationService
from dlkit.tools.config.protocols import TrainingSettingsProtocol


class TestServerApplicationService:
    """Test application service orchestration."""

    @pytest.fixture
    def mock_adapter(self) -> Mock:
        """Create mock server adapter."""
        return Mock()

    @pytest.fixture
    def mock_management_service(self) -> Mock:
        """Create mock server management service."""
        return Mock()

    @pytest.fixture
    def app_service(
        self, mock_adapter: Mock, mock_management_service: Mock
    ) -> ServerApplicationService:
        """Create application service with mocked dependencies."""
        return ServerApplicationService(mock_adapter, mock_management_service)

    def test_application_service_uses_dependency_injection(self) -> None:
        """Test that application service uses dependency injection."""
        mock_adapter = Mock()
        mock_management = Mock()

        app_service = ServerApplicationService(mock_adapter, mock_management)

        assert app_service._server_adapter is mock_adapter
        assert app_service._server_management is mock_management

    def test_application_service_creates_defaults_when_no_dependencies(self) -> None:
        """Test that application service creates defaults when no dependencies provided."""
        app_service = ServerApplicationService()

        assert app_service._server_adapter is not None
        assert app_service._server_management is not None

    def test_start_server_with_config_file(
        self,
        app_service: ServerApplicationService,
        mock_adapter: Mock,
        mock_management_service: Mock,
        tmp_path: Path,
    ) -> None:
        """Test starting server with configuration file."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[MLFLOW]\nenabled = true")

        mock_server_config = Mock()
        mock_management_service.create_server_context.return_value = mock_server_config
        mock_management_service.ensure_storage_setup.return_value = mock_server_config

        mock_server_info = Mock()
        mock_server_info.pid = 12345
        mock_server_info.host = "localhost"
        mock_server_info.port = 5000
        mock_adapter.start_server.return_value = mock_server_info

        result = app_service.start_server(config_path=config_path)

        assert result is mock_server_info
        # create_server_context is no longer called - we return settings directly
        mock_management_service.ensure_storage_setup.assert_called_once()
        mock_management_service.track_server.assert_called_once_with("localhost", 5000, 12345)

    def test_start_server_without_config_creates_defaults(
        self,
        app_service: ServerApplicationService,
        mock_adapter: Mock,
        mock_management_service: Mock,
    ) -> None:
        """Test starting server without config creates default settings."""
        mock_server_config = Mock()
        mock_management_service.ensure_storage_setup.return_value = mock_server_config

        mock_server_info = Mock()
        mock_server_info.pid = None
        mock_adapter.start_server.return_value = mock_server_info

        result = app_service.start_server(host="0.0.0.0", port=8080)

        assert result is mock_server_info
        # Should not track server without PID
        mock_management_service.track_server.assert_not_called()

    def test_start_server_with_overrides(
        self,
        app_service: ServerApplicationService,
        mock_adapter: Mock,
        mock_management_service: Mock,
        tmp_path: Path,
    ) -> None:
        """Test starting server with parameter overrides."""
        mock_server_config = Mock()
        mock_management_service.ensure_storage_setup.return_value = mock_server_config

        mock_server_info = Mock()
        mock_server_info.pid = 12345
        mock_server_info.host = "0.0.0.0"
        mock_server_info.port = 8080
        mock_adapter.start_server.return_value = mock_server_info

        artifacts_destination = str((tmp_path / "artifacts").resolve())

        app_service.start_server(
            host="0.0.0.0",
            port=8080,
            backend_store_uri="sqlite:///test.db",
            artifacts_destination=artifacts_destination,
        )

        # Verify overrides passed to adapter
        mock_adapter.start_server.assert_called_once()
        call_kwargs = mock_adapter.start_server.call_args[1]
        assert call_kwargs["host"] == "0.0.0.0"
        assert call_kwargs["port"] == 8080
        assert call_kwargs["backend_store_uri"] == "sqlite:///test.db"
        assert Path(call_kwargs["artifacts_destination"]) == Path(artifacts_destination)

    def test_stop_server_checks_status_first(
        self,
        app_service: ServerApplicationService,
        mock_adapter: Mock,
        mock_management_service: Mock,
    ) -> None:
        """Test that stop_server checks server status first."""
        mock_status = Mock()
        mock_status.is_running = True
        mock_adapter.check_server.return_value = mock_status
        mock_management_service.stop_server_processes.return_value = (True, ["Stopped"])

        success, messages = app_service.stop_server("localhost", 5000, False)

        assert success is True
        mock_adapter.check_server.assert_called_once_with("localhost", 5000)
        mock_management_service.stop_server_processes.assert_called_once_with(
            "localhost", 5000, False
        )
        mock_management_service.untrack_server.assert_called_once_with("localhost", 5000)

    def test_stop_server_skips_check_when_force(
        self,
        app_service: ServerApplicationService,
        mock_adapter: Mock,
        mock_management_service: Mock,
    ) -> None:
        """Test that stop_server skips status check when force=True."""
        mock_management_service.stop_server_processes.return_value = (True, ["Force stopped"])

        success, messages = app_service.stop_server("localhost", 5000, True)

        assert success is True
        # Should not check status when force=True
        mock_adapter.check_server.assert_not_called()
        mock_management_service.stop_server_processes.assert_called_once_with(
            "localhost", 5000, True
        )

    def test_stop_server_returns_early_if_not_running(
        self,
        app_service: ServerApplicationService,
        mock_adapter: Mock,
        mock_management_service: Mock,
    ) -> None:
        """Test that stop_server returns early if server is not running."""
        mock_status = Mock()
        mock_status.is_running = False
        mock_adapter.check_server.return_value = mock_status

        success, messages = app_service.stop_server("localhost", 5000, False)

        assert success is True
        assert "No server found running" in messages[0]
        # Should not attempt to stop processes
        mock_management_service.stop_server_processes.assert_not_called()

    def test_stop_server_handles_check_failure(
        self,
        app_service: ServerApplicationService,
        mock_adapter: Mock,
        mock_management_service: Mock,
    ) -> None:
        """Test that stop_server handles status check failure."""
        mock_adapter.check_server.side_effect = Exception("Connection failed")

        success, messages = app_service.stop_server("localhost", 5000, False)

        assert success is False
        assert "Could not check server status" in messages[0]

    def test_check_server_status_delegates_to_adapter(
        self,
        app_service: ServerApplicationService,
        mock_adapter: Mock,
    ) -> None:
        """Test that check_server_status delegates to adapter."""
        mock_status = Mock()
        mock_adapter.check_server.return_value = mock_status

        result = app_service.check_server_status("localhost", 8080)

        assert result is mock_status
        mock_adapter.check_server.assert_called_once_with("localhost", 8080)

    def test_get_server_configuration_info_without_config_path(
        self, app_service: ServerApplicationService
    ) -> None:
        """Test getting configuration info without config path."""
        result = app_service.get_server_configuration_info(None)

        assert result["configured"] is False
        assert "No configuration file provided" in result["message"]

    @patch("dlkit.interfaces.cli.adapters.config_adapter.load_config")
    def test_get_server_configuration_info_with_valid_config(
        self,
        mock_load_config: Mock,
        app_service: ServerApplicationService,
        tmp_path: Path,
    ) -> None:
        """Test getting configuration info with valid config file."""
        config_path = tmp_path / "config.toml"

        # Mock settings with MLflow configuration
        # Use MagicMock with spec to satisfy TrainingSettingsProtocol isinstance check
        mock_settings = MagicMock(spec=TrainingSettingsProtocol)

        mock_mlflow = Mock()
        mock_mlflow.enabled = True
        mock_mlflow.server = Mock()
        mock_mlflow.server.host = "localhost"
        mock_mlflow.server.port = 5000
        mock_mlflow.server.backend_store_uri = "sqlite:///test.db"
        mock_mlflow.server.artifacts_destination = str((tmp_path / "artifacts").resolve())
        mock_mlflow.client = Mock()
        mock_mlflow.client.tracking_uri = "http://localhost:5000"
        mock_mlflow.client.experiment_name = "test_experiment"
        mock_settings.MLFLOW = mock_mlflow
        mock_load_config.return_value = mock_settings

        result = app_service.get_server_configuration_info(config_path)

        assert result["configured"] is True
        assert result["server"]["host"] == "localhost"
        assert result["server"]["port"] == 5000
        assert result["client"]["experiment"] == "test_experiment"

    @patch("dlkit.interfaces.cli.adapters.config_adapter.load_config")
    def test_get_server_configuration_info_with_inactive_mlflow(
        self,
        mock_load_config: Mock,
        app_service: ServerApplicationService,
        tmp_path: Path,
    ) -> None:
        """Test getting configuration info with inactive MLflow."""
        config_path = tmp_path / "config.toml"

        # Use MagicMock with spec to satisfy TrainingSettingsProtocol isinstance check
        mock_settings = MagicMock(spec=TrainingSettingsProtocol)
        mock_settings.MLFLOW = None
        mock_load_config.return_value = mock_settings

        result = app_service.get_server_configuration_info(config_path)

        assert result["configured"] is False
        assert "not configured" in result["message"]

    @patch("dlkit.interfaces.cli.adapters.config_adapter.load_config")
    def test_get_server_configuration_info_handles_load_error(
        self,
        mock_load_config: Mock,
        app_service: ServerApplicationService,
        tmp_path: Path,
    ) -> None:
        """Test getting configuration info handles load errors."""
        config_path = tmp_path / "config.toml"
        mock_load_config.side_effect = Exception("Parse error")

        result = app_service.get_server_configuration_info(config_path)

        assert result["configured"] is False
        assert "Error loading configuration" in result["message"]

    def test_build_overrides_dict_filters_none_values(
        self, app_service: ServerApplicationService
    ) -> None:
        """Test that override dict building filters None values."""
        overrides = app_service._build_overrides_dict(
            host="localhost",
            port=None,
            backend_store_uri="sqlite:///test.db",
            artifacts_destination=None,
        )

        assert overrides == {"host": "localhost", "backend_store_uri": "sqlite:///test.db"}
        assert "port" not in overrides
        assert "artifacts_destination" not in overrides

    def test_load_server_configuration_creates_defaults_without_config(
        self,
        app_service: ServerApplicationService,
        tmp_path: Path,
    ) -> None:
        """Test that server configuration loading creates defaults without config."""
        artifacts_destination = str((tmp_path / "artifacts").resolve())

        result = app_service._load_server_configuration(
            None, "0.0.0.0", 8080, "sqlite:///test.db", artifacts_destination
        )

        # Should create MLflowServerSettings with provided values
        assert result.host == "0.0.0.0"
        assert result.port == 8080
        assert result.backend_store_uri == "sqlite:///test.db"
        assert Path(result.artifacts_destination) == Path(artifacts_destination)

    def test_load_server_configuration_uses_defaults_for_missing_values(
        self, app_service: ServerApplicationService
    ) -> None:
        """Test that server configuration uses defaults for missing values."""
        result = app_service._load_server_configuration(None, None, None, None, None)

        # Should use defaults
        assert result.host == "127.0.0.1"
        assert result.port == 5000
        assert result.backend_store_uri is None
        assert result.artifacts_destination is None


class TestApplicationServiceIntegration:
    """Test application service integration with dependencies."""

    def test_application_service_orchestrates_server_start_workflow(self) -> None:
        """Test that application service orchestrates complete server start workflow."""
        # This is an integration test showing the complete workflow
        with patch(
            "dlkit.interfaces.servers.application_service.create_mlflow_adapter"
        ) as mock_adapter_factory:
            with patch(
                "dlkit.interfaces.servers.application_service.ServerManagementService"
            ) as mock_service_class:
                mock_adapter = Mock()
                mock_management = Mock()
                mock_adapter_factory.return_value = mock_adapter
                mock_service_class.return_value = mock_management

                # Mock the workflow
                mock_server_config = Mock()
                mock_management.ensure_storage_setup.return_value = mock_server_config
                mock_server_info = Mock()
                mock_server_info.pid = 12345
                mock_server_info.host = "localhost"
                mock_server_info.port = 5000
                mock_adapter.start_server.return_value = mock_server_info

                # Execute workflow
                app_service = ServerApplicationService()
                result = app_service.start_server(host="localhost", port=5000)

                # Verify complete workflow
                assert result is mock_server_info
                mock_management.ensure_storage_setup.assert_called_once()
                mock_adapter.start_server.assert_called_once()
                mock_management.track_server.assert_called_once_with("localhost", 5000, 12345)

    def test_application_service_orchestrates_server_stop_workflow(self) -> None:
        """Test that application service orchestrates complete server stop workflow."""
        with patch(
            "dlkit.interfaces.servers.application_service.create_mlflow_adapter"
        ) as mock_adapter_factory:
            with patch(
                "dlkit.interfaces.servers.application_service.ServerManagementService"
            ) as mock_service_class:
                mock_adapter = Mock()
                mock_management = Mock()
                mock_adapter_factory.return_value = mock_adapter
                mock_service_class.return_value = mock_management

                # Mock the workflow
                mock_status = Mock()
                mock_status.is_running = True
                mock_adapter.check_server.return_value = mock_status
                mock_management.stop_server_processes.return_value = (True, ["Stopped"])

                # Execute workflow
                app_service = ServerApplicationService()
                success, messages = app_service.stop_server("localhost", 5000)

                # Verify complete workflow
                assert success is True
                mock_adapter.check_server.assert_called_once_with("localhost", 5000)
                mock_management.stop_server_processes.assert_called_once_with(
                    "localhost", 5000, False
                )
                mock_management.untrack_server.assert_called_once_with("localhost", 5000)

    def test_application_service_follows_single_responsibility_principle(self) -> None:
        """Test that application service only orchestrates, doesn't implement domain logic."""
        app_service = ServerApplicationService()

        # Should only have orchestration methods, not domain logic
        assert hasattr(app_service, "start_server")
        assert hasattr(app_service, "stop_server")
        assert hasattr(app_service, "check_server_status")
        assert hasattr(app_service, "get_server_configuration_info")

        # Should not have low-level implementation methods
        assert not hasattr(app_service, "track_server")
        assert not hasattr(app_service, "kill_processes")
        assert not hasattr(app_service, "create_directories")

    def test_application_service_uses_dependency_inversion(self) -> None:
        """Test that application service depends on abstractions."""
        # Verify constructor accepts abstractions
        mock_adapter = Mock()
        mock_management = Mock()

        app_service = ServerApplicationService(mock_adapter, mock_management)

        # Should accept any object with correct interface
        assert app_service._server_adapter is mock_adapter
        assert app_service._server_management is mock_management
