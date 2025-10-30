"""Tests to verify server properly reads config file settings without explicit overrides."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch
from typing import Any

import pytest

from dlkit.interfaces.servers.application_service import ServerApplicationService
from dlkit.interfaces.servers.storage_adapter import MLflowStorageSetup
from dlkit.interfaces.servers.server_configuration import should_use_default_storage


class TestServerConfigReading:
    """Test server configuration reading behavior."""

    @pytest.fixture
    def config_with_backend_and_artifacts(self, tmp_path: Path) -> Path:
        """Create config file with both backend_store_uri and artifacts_destination."""
        config_path = tmp_path / "config.toml"
        config_content = """
[SESSION]
name = "test_session"
inference = false

[MLFLOW]
enabled = true

[MLFLOW.server]
backend_store_uri = "sqlite:///custom/path/mlflow.db"
artifacts_destination = "/custom/artifacts/path"
host = "0.0.0.0"
port = 8080

[MLFLOW.client]
experiment_name = "test_experiment"
"""
        config_path.write_text(config_content)
        return config_path

    @pytest.fixture
    def config_with_only_backend_uri(self, tmp_path: Path) -> Path:
        """Create config file with only backend_store_uri."""
        config_path = tmp_path / "config.toml"
        config_content = """
[MLFLOW]
enabled = true

[MLFLOW.server]
backend_store_uri = "sqlite:///custom/path/mlflow.db"
host = "localhost"
port = 5000
"""
        config_path.write_text(config_content)
        return config_path

    @pytest.fixture
    def config_with_only_artifacts_destination(self, tmp_path: Path) -> Path:
        """Create config file with only artifacts_destination."""
        config_path = tmp_path / "config.toml"
        config_content = """
[MLFLOW]
enabled = true

[MLFLOW.server]
artifacts_destination = "/custom/artifacts/path"
host = "localhost"
port = 5000
"""
        config_path.write_text(config_content)
        return config_path

    @pytest.fixture
    def config_without_storage_settings(self, tmp_path: Path) -> Path:
        """Create config file without storage settings."""
        config_path = tmp_path / "config.toml"
        config_content = """
[MLFLOW]
enabled = true

[MLFLOW.server]
host = "localhost"
port = 5000
"""
        config_path.write_text(config_content)
        return config_path

    def test_should_use_default_storage_respects_config_backend_uri(self) -> None:
        """Test that should_use_default_storage returns False when config has backend_store_uri."""
        # Create mock server config with backend_store_uri
        server_config = Mock()
        server_config.backend_store_uri = "sqlite:///custom/path/mlflow.db"
        server_config.artifacts_destination = None

        overrides: dict[str, Any] = {}

        result = should_use_default_storage(server_config, overrides)

        assert result is False, "Should not use default storage when config has backend_store_uri"

    def test_should_use_default_storage_respects_config_artifacts_destination(self) -> None:
        """Test that should_use_default_storage returns False when config has artifacts_destination."""
        # Create mock server config with artifacts_destination
        server_config = Mock()
        server_config.backend_store_uri = None
        server_config.artifacts_destination = "/custom/artifacts/path"

        overrides: dict[str, Any] = {}

        result = should_use_default_storage(server_config, overrides)

        assert result is False, (
            "Should not use default storage when config has artifacts_destination"
        )

    def test_should_use_default_storage_respects_config_both_settings(self) -> None:
        """Test that should_use_default_storage returns False when config has both storage settings."""
        # Create mock server config with both settings
        server_config = Mock()
        server_config.backend_store_uri = "sqlite:///custom/path/mlflow.db"
        server_config.artifacts_destination = "/custom/artifacts/path"

        overrides: dict[str, Any] = {}

        result = should_use_default_storage(server_config, overrides)

        assert result is False, (
            "Should not use default storage when config has both storage settings"
        )

    def test_should_use_default_storage_when_no_config_settings(self) -> None:
        """Test that should_use_default_storage returns True when config has no storage settings."""
        # Create mock server config without storage settings
        server_config = Mock()
        server_config.backend_store_uri = None
        server_config.artifacts_destination = None

        overrides: dict[str, Any] = {}

        result = should_use_default_storage(server_config, overrides)

        assert result is True, "Should use default storage when config has no storage settings"

    def test_explicit_overrides_take_precedence_over_config(self) -> None:
        """Test that explicit overrides take precedence over config settings."""
        # Create mock server config with backend_store_uri
        server_config = Mock()
        server_config.backend_store_uri = "sqlite:///config/path/mlflow.db"
        server_config.artifacts_destination = "/config/artifacts/path"

        # But provide explicit override
        overrides = {"backend_store_uri": "sqlite:///override/path/mlflow.db"}

        result = should_use_default_storage(server_config, overrides)

        assert result is False, "Should not use default storage when explicit override provided"

    @patch("dlkit.interfaces.servers.storage_adapter.MLflowStorageSetup.ensure_storage_setup")
    def test_application_service_uses_config_settings_without_overrides(
        self, mock_ensure_storage: Mock, config_with_backend_and_artifacts: Path
    ) -> None:
        """Test that ApplicationService uses config settings when no overrides provided."""
        # Mock the storage setup to return the config unchanged
        mock_ensure_storage.side_effect = lambda config, overrides: config

        # Mock the server adapter to avoid actual server startup
        mock_server_adapter = Mock()
        mock_server_info = Mock()
        mock_server_info.host = "0.0.0.0"
        mock_server_info.port = 8080
        mock_server_info.pid = None
        mock_server_adapter.start_server.return_value = mock_server_info

        # Create application service with mocked adapter
        app_service = ServerApplicationService(server_adapter=mock_server_adapter)

        # Start server with config but no overrides
        result = app_service.start_server(
            config_path=config_with_backend_and_artifacts,
            host=None,
            port=None,
            backend_store_uri=None,
            artifacts_destination=None,
        )

        # Verify storage setup was called
        mock_ensure_storage.assert_called_once()

        # Get the arguments passed to ensure_storage_setup
        call_args = mock_ensure_storage.call_args
        server_config, overrides = call_args[0]

        # Verify config has the expected values from file
        # The server_config is actually a MLflowServerSettings object
        assert str(server_config.backend_store_uri) == "sqlite:///custom/path/mlflow.db"
        assert str(server_config.artifacts_destination) == "/custom/artifacts/path"

        # Verify no overrides were passed
        assert overrides == {}
        assert result.host == "0.0.0.0"
        assert result.port == 8080

    @patch("dlkit.interfaces.servers.infrastructure_adapters.TyperUserInteraction")
    @patch("dlkit.interfaces.servers.infrastructure_adapters.StandardFileSystemOperations")
    def test_storage_adapter_skips_default_storage_prompt_when_config_has_settings(
        self,
        mock_file_system: Mock,
        mock_user_interaction: Mock,
        config_with_backend_and_artifacts: Path,
    ) -> None:
        """Test that storage adapter doesn't prompt for default storage when config has settings."""
        # Setup mocks
        mock_file_system_instance = Mock()
        mock_user_interaction_instance = Mock()
        mock_file_system.return_value = mock_file_system_instance
        mock_user_interaction.return_value = mock_user_interaction_instance

        # Create storage adapter
        storage_adapter = MLflowStorageSetup(
            mock_user_interaction_instance, mock_file_system_instance
        )

        # Create server config with storage settings
        server_config = Mock()
        server_config.backend_store_uri = "sqlite:///custom/path/mlflow.db"
        server_config.artifacts_destination = "/custom/artifacts/path"

        overrides: dict[str, Any] = {}

        # Call ensure_storage_setup
        result = storage_adapter.ensure_storage_setup(server_config, overrides)

        # Verify it returns the config unchanged
        assert result == server_config

        # Verify no user interaction occurred (no prompts)
        mock_user_interaction_instance.show_message.assert_not_called()
        mock_user_interaction_instance.confirm_action.assert_not_called()

        # Verify no file system operations occurred
        mock_file_system_instance.directory_exists.assert_not_called()
        mock_file_system_instance.create_directory.assert_not_called()

    @patch("dlkit.interfaces.servers.infrastructure_adapters.TyperUserInteraction")
    @patch("dlkit.interfaces.servers.infrastructure_adapters.StandardFileSystemOperations")
    def test_storage_adapter_prompts_for_default_storage_when_config_lacks_settings(
        self,
        mock_file_system: Mock,
        mock_user_interaction: Mock,
        config_without_storage_settings: Path,
    ) -> None:
        """Test that storage adapter prompts for default storage when config lacks settings."""
        # Setup mocks
        mock_file_system_instance = Mock()
        mock_user_interaction_instance = Mock()
        mock_file_system.return_value = mock_file_system_instance
        mock_user_interaction.return_value = mock_user_interaction_instance

        # Mock file system to report mlruns doesn't exist
        mock_file_system_instance.directory_exists.return_value = False
        # Mock user to confirm creating default storage
        mock_user_interaction_instance.confirm_action.return_value = True

        # Create storage adapter
        storage_adapter = MLflowStorageSetup(
            mock_user_interaction_instance, mock_file_system_instance
        )

        # Create server config without storage settings
        server_config = Mock()
        server_config.backend_store_uri = None
        server_config.artifacts_destination = None

        overrides: dict[str, Any] = {}

        # Call ensure_storage_setup
        result = storage_adapter.ensure_storage_setup(server_config, overrides)

        # Storage adapter no longer prompts users - it's pure business logic
        # User interaction is handled at the CLI layer
        # So we don't expect these to be called
        mock_user_interaction_instance.show_message.assert_not_called()
        mock_user_interaction_instance.confirm_action.assert_not_called()

        # Verify file system operations occurred
        mock_file_system_instance.directory_exists.assert_called_once()
        mock_file_system_instance.create_directory.assert_called_once()
        assert result is server_config

    def test_config_with_only_backend_uri_should_not_use_default_storage(self) -> None:
        """Test that config with only backend_store_uri should not use default storage."""
        server_config = Mock()
        server_config.backend_store_uri = "sqlite:///custom/path/mlflow.db"
        server_config.artifacts_destination = None

        overrides: dict[str, Any] = {}

        result = should_use_default_storage(server_config, overrides)

        assert result is False, "Should not use default storage when config has backend_store_uri"

    def test_config_with_only_artifacts_destination_should_not_use_default_storage(self) -> None:
        """Test that config with only artifacts_destination should not use default storage."""
        server_config = Mock()
        server_config.backend_store_uri = None
        server_config.artifacts_destination = "/custom/artifacts/path"

        overrides: dict[str, Any] = {}

        result = should_use_default_storage(server_config, overrides)

        assert result is False, (
            "Should not use default storage when config has artifacts_destination"
        )

    @patch("dlkit.interfaces.servers.storage_adapter.should_use_default_storage")
    def test_storage_adapter_calls_should_use_default_storage_with_correct_params(
        self, mock_should_use_default: Mock
    ) -> None:
        """Test that storage adapter calls should_use_default_storage with correct parameters."""
        # Setup mock to return False (don't use default storage)
        mock_should_use_default.return_value = False

        # Create mocks for dependencies
        mock_user_interaction = Mock()
        mock_file_system = Mock()

        # Create storage adapter
        storage_adapter = MLflowStorageSetup(mock_user_interaction, mock_file_system)

        # Create test dataflow
        server_config = Mock()
        overrides = {"host": "localhost"}

        # Call ensure_storage_setup
        storage_adapter.ensure_storage_setup(server_config, overrides)

        # Verify should_use_default_storage was called with correct parameters
        mock_should_use_default.assert_called_once_with(server_config, overrides)


class TestServerConfigIntegration:
    """Integration tests for server config behavior."""

    @pytest.fixture
    def temp_config_file(self, tmp_path: Path) -> Path:
        """Create temporary config file for integration tests."""
        config_path = tmp_path / "integration_config.toml"
        config_content = """
[MLFLOW]
enabled = true

[MLFLOW.server]
backend_store_uri = "sqlite:///integration/test/mlflow.db"
artifacts_destination = "/integration/test/artifacts"
host = "0.0.0.0" 
port = 9000
"""
        config_path.write_text(config_content)
        return config_path

    def test_full_integration_config_reading_without_overrides(
        self, temp_config_file: Path
    ) -> None:
        """Integration test: verify config values are used when no overrides provided."""
        # Mock server adapter to avoid actual server startup
        mock_server_adapter = Mock()
        mock_server_info = Mock()
        mock_server_info.host = "0.0.0.0"
        mock_server_info.port = 9000
        mock_server_info.pid = None
        mock_server_adapter.start_server.return_value = mock_server_info

        # Create application service with mocked adapter
        app_service = ServerApplicationService(server_adapter=mock_server_adapter)

        # Start server with config file but no CLI overrides
        result = app_service.start_server(
            config_path=temp_config_file,
            host=None,  # No CLI override
            port=None,  # No CLI override
            backend_store_uri=None,  # No CLI override
            artifacts_destination=None,  # No CLI override
        )

        # Verify start_server was called on the adapter
        mock_server_adapter.start_server.assert_called_once()

        # Get the call arguments to verify config values were passed
        call_args = mock_server_adapter.start_server.call_args

        # The server_config should have the values from the file
        server_config = call_args.kwargs["server_config"]
        # The server_config is actually a MLflowServerSettings object
        assert str(server_config.backend_store_uri) == "sqlite:///integration/test/mlflow.db"
        assert str(server_config.artifacts_destination) == "/integration/test/artifacts"
        assert server_config.host == "0.0.0.0"
        assert server_config.port == 9000
        assert result.host == "0.0.0.0"
        assert result.port == 9000

    def test_full_integration_cli_overrides_take_precedence(self, temp_config_file: Path) -> None:
        """Integration test: verify CLI overrides take precedence over config values."""
        # Mock server adapter to avoid actual server startup
        mock_server_adapter = Mock()
        mock_server_info = Mock()
        mock_server_info.host = "localhost"
        mock_server_info.port = 8080
        mock_server_info.pid = None
        mock_server_adapter.start_server.return_value = mock_server_info

        # Create application service with mocked adapter
        app_service = ServerApplicationService(server_adapter=mock_server_adapter)

        # Start server with config file AND CLI overrides
        result = app_service.start_server(
            config_path=temp_config_file,
            host="localhost",  # Override config host
            port=8080,  # Override config port
            backend_store_uri="sqlite:///override/path/mlflow.db",  # Override config backend
            artifacts_destination="/override/artifacts",  # Override config artifacts
        )

        # Verify start_server was called with overrides
        mock_server_adapter.start_server.assert_called_once()

        # Get the call arguments to verify overrides were applied
        call_args = mock_server_adapter.start_server.call_args
        call_kwargs = call_args.kwargs

        # Verify the CLI overrides were passed as keyword arguments
        assert call_kwargs.get("host") == "localhost"
        assert call_kwargs.get("port") == 8080
        assert call_kwargs.get("backend_store_uri") == "sqlite:///override/path/mlflow.db"
        assert call_kwargs.get("artifacts_destination") == "/override/artifacts"
        assert result.host == "localhost"
        assert result.port == 8080
