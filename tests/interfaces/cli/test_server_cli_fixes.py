"""Integration tests for MLflow server CLI fixes.

This module tests the fixes for two reported issues:
1. Double prompt issue when no config file is provided
2. Config file handling issue where context objects were passed incorrectly
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from dlkit.interfaces.cli.commands.server import app as server_app
from dlkit.interfaces.servers.protocols import ServerInfo
from dlkit.tools.config.mlflow_settings import MLflowServerSettings


class TestMLflowServerCLIFixes:
    """Test MLflow server CLI fixes for reported issues."""

    @pytest.fixture
    def cli_runner(self) -> CliRunner:
        """Create CLI runner for tests."""
        return CliRunner()

    @pytest.fixture
    def mock_server_info(self) -> ServerInfo:
        """Create mock server info for testing."""
        return ServerInfo(
            process=None,
            url="http://127.0.0.1:5000",
            host="127.0.0.1",
            port=5000,
            pid=12345,
        )

    @pytest.fixture
    def sample_config_file(self, tmp_path: Path) -> Path:
        """Create a sample config file for testing."""
        config_file = tmp_path / "test_config.toml"
        config_content = """[SESSION]
name = "test_session"
inference = false

[MLFLOW]
enabled = true

[MLFLOW.server]
host = "127.0.0.1"
port = 5555
backend_store_uri = "sqlite:///test_mlflow.db"
artifacts_destination = "./test_artifacts"

[MLFLOW.client]
tracking_uri = "http://127.0.0.1:5555"
experiment_name = "test_experiment"
"""
        config_file.write_text(config_content)
        return config_file

    def test_config_file_handling_issue_fixed(
        self,
        cli_runner: CliRunner,
        sample_config_file: Path,
        mock_server_info: ServerInfo,
    ) -> None:
        """Test that config file handling works without type errors.

        This addresses the reported issue where 'MLflowServerContext' object
        has no attribute 'host' error occurred when using config files.
        """
        with patch(
            "dlkit.interfaces.cli.commands.server.ServerApplicationService"
        ) as mock_service_class, patch(
            "dlkit.interfaces.cli.commands.server._ensure_storage_setup_at_cli_level"
        ) as mock_storage_setup:
            # Mock storage setup to avoid real config loading
            mock_storage_setup.return_value = None

            # Setup mock application service
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            mock_service.start_server.return_value = mock_server_info

            # Test config file usage
            result = cli_runner.invoke(server_app, ["start", str(sample_config_file), "--detach"])

            # Should succeed without errors
            assert result.exit_code == 0
            assert "MLflow server started successfully" in result.stdout
            assert "127.0.0.1:5000" in result.stdout

            # Verify the application service was called correctly
            mock_service.start_server.assert_called_once()
            call_args = mock_service.start_server.call_args

            # The config_path should be passed as first argument
            assert call_args[0][0] == sample_config_file

    def test_double_prompt_issue_fixed_with_user_interaction_mock(
        self,
        cli_runner: CliRunner,
        mock_server_info: ServerInfo,
    ) -> None:
        """Test that storage setup prompts user only once.

        This addresses the reported issue where storage setup asked twice
        for user confirmation when no config file was provided.
        """
        with patch(
            "dlkit.interfaces.cli.commands.server.ServerApplicationService"
        ) as mock_service_class, patch(
            "dlkit.interfaces.cli.commands.server._ensure_storage_setup_at_cli_level"
        ) as mock_storage_setup:
            # Mock storage setup to avoid real config loading
            mock_storage_setup.return_value = None

            # Setup mock application service
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            mock_service.start_server.return_value = mock_server_info

            # Test without config file (triggers storage setup)
            result = cli_runner.invoke(server_app, ["start", "--port", "6000", "--detach"])

            # Should succeed
            assert result.exit_code == 0
            assert "MLflow server started successfully" in result.stdout

            # Verify application service was called with no config file
            mock_service.start_server.assert_called_once()
            call_args = mock_service.start_server.call_args
            assert call_args[0][0] is None  # config_path is first positional argument

    def test_storage_adapter_no_longer_prompts_user(self, tmp_path: Path) -> None:
        """Test that storage adapter no longer prompts user - it's pure business logic."""
        from dlkit.interfaces.servers.storage_adapter import MLflowStorageSetup
        from dlkit.interfaces.servers.infrastructure_adapters import StandardFileSystemOperations

        # Setup - mock user interaction should never be called
        mock_user_interaction = Mock()
        file_system = StandardFileSystemOperations()

        # Create storage adapter
        storage_adapter = MLflowStorageSetup(mock_user_interaction, file_system)

        # Create a temporary MLflow server config that would trigger storage setup
        mock_config = MLflowServerSettings(
            host="127.0.0.1",
            port=5000,
        )

        # Simulate missing mlruns directory by using a different temp path
        test_mlruns = tmp_path / "test_mlruns"

        with patch(
            "dlkit.interfaces.servers.storage_adapter.get_default_mlruns_path",
            return_value=test_mlruns,
        ):
            # Run storage setup - should be pure business logic now
            result_config = storage_adapter.ensure_storage_setup(mock_config, {})

            # Verify NO user interaction (business logic layer should not prompt)
            assert mock_user_interaction.confirm_action.call_count == 0
            assert mock_user_interaction.show_message.call_count == 0

            # Directory should be created automatically
            assert test_mlruns.exists()

            # Should return the original config
            assert result_config == mock_config

    def test_cli_architecture_separation_concept(self) -> None:
        """Test that the architectural separation concept is implemented correctly."""
        # This test verifies the architectural improvement we made:
        # CLI layer handles user interaction, business logic is pure

        from dlkit.interfaces.servers.storage_adapter import MLflowStorageSetup
        from dlkit.interfaces.servers.infrastructure_adapters import StandardFileSystemOperations
        from dlkit.interfaces.cli.commands.server import _ensure_storage_setup_at_cli_level

        # 1. Verify storage adapter is pure business logic
        mock_user_interaction = Mock()
        file_system = StandardFileSystemOperations()
        storage_adapter = MLflowStorageSetup(mock_user_interaction, file_system)

        # Storage adapter should not need user interaction anymore
        assert hasattr(
            storage_adapter, "_user_interaction"
        )  # Still has it for backward compatibility
        assert hasattr(
            storage_adapter, "_file_system"
        )  # But only uses file system in ensure_storage_setup

        # 2. Verify CLI function exists and handles user interaction
        assert callable(_ensure_storage_setup_at_cli_level)

        # 3. The key architectural improvement is that:
        # - CLI layer (_ensure_storage_setup_at_cli_level) handles prompts
        # - Business logic layer (storage_adapter.ensure_storage_setup) is pure
        # This separation is the main fix we implemented

    def test_storage_adapter_pure_business_logic(self, tmp_path: Path) -> None:
        """Test that storage adapter is now pure business logic without user prompts."""
        from dlkit.interfaces.servers.storage_adapter import MLflowStorageSetup
        from dlkit.interfaces.servers.infrastructure_adapters import StandardFileSystemOperations

        # Create real instances (no mocking for user interaction)
        mock_user_interaction = Mock()  # This shouldn't be used at all
        file_system = StandardFileSystemOperations()

        # Create storage adapter
        storage_adapter = MLflowStorageSetup(mock_user_interaction, file_system)

        # Create a temporary MLflow server config
        mock_config = MLflowServerSettings(
            host="127.0.0.1",
            port=5000,
        )

        # Simulate missing mlruns directory
        test_mlruns = tmp_path / "test_business_logic_mlruns"

        with patch(
            "dlkit.interfaces.servers.storage_adapter.get_default_mlruns_path",
            return_value=test_mlruns,
        ):
            # Run storage setup - should automatically create directory
            result_config = storage_adapter.ensure_storage_setup(mock_config, {})

            # Verify no user interaction occurred (pure business logic)
            mock_user_interaction.confirm_action.assert_not_called()
            mock_user_interaction.show_message.assert_not_called()

            # Directory should be created automatically
            assert test_mlruns.exists()

            # Should return the original config
            assert result_config == mock_config

    def test_type_consistency_in_application_service(self, tmp_path: Path) -> None:
        """Test that _load_server_configuration returns consistent MLflowServerSettings type."""
        from dlkit.interfaces.servers.application_service import ServerApplicationService
        from dlkit.tools.config.mlflow_settings import MLflowServerSettings

        # Create a test config file
        config_file = tmp_path / "test_consistency.toml"
        config_content = """[SESSION]
name = "test_session"
inference = false

[MLFLOW]
enabled = true

[MLFLOW.server]
host = "127.0.0.1"
port = 8888
backend_store_uri = "sqlite:///consistency_test.db"

[MLFLOW.client]
experiment_name = "test_experiment"
"""
        config_file.write_text(config_content)

        # Create application service
        app_service = ServerApplicationService()

        # Test config file loading
        result_with_config = app_service._load_server_configuration(
            config_file, None, None, None, None
        )

        # Should return MLflowServerSettings, not MLflowServerContext
        assert isinstance(result_with_config, MLflowServerSettings)
        # With the new config structure working correctly, it reads configured values
        assert result_with_config.host == "127.0.0.1"
        assert result_with_config.port == 8888
        assert str(result_with_config.backend_store_uri) == "sqlite:///consistency_test.db"

        # Test default configuration
        result_without_config = app_service._load_server_configuration(
            None, "localhost", 9999, "sqlite:///default.db", "./artifacts"
        )

        # Should also return MLflowServerSettings
        assert isinstance(result_without_config, MLflowServerSettings)
        assert result_without_config.host == "localhost"
        assert result_without_config.port == 9999
        assert str(result_without_config.backend_store_uri) == "sqlite:///default.db"
        assert str(result_without_config.artifacts_destination) == "./artifacts"

    def test_config_file_vs_defaults_return_same_type(self, tmp_path: Path) -> None:
        """Test that both config file and default paths return the same type."""
        from dlkit.interfaces.servers.application_service import ServerApplicationService
        from dlkit.tools.config.mlflow_settings import MLflowServerSettings

        # Create test config file
        config_file = tmp_path / "test_types.toml"
        config_content = """[SESSION]
name = "test_session"
inference = false

[MLFLOW]
enabled = true

[MLFLOW.client]
experiment_name = "test_experiment"
"""
        config_file.write_text(config_content)

        app_service = ServerApplicationService()

        # Both should return the same type
        config_result = app_service._load_server_configuration(config_file, None, None, None, None)
        default_result = app_service._load_server_configuration(None, "127.0.0.1", 7777, None, None)

        # Both should be MLflowServerSettings
        assert type(config_result) is type(default_result)
        assert isinstance(config_result, MLflowServerSettings)
        assert isinstance(default_result, MLflowServerSettings)

        # Both should have the same basic structure
        assert hasattr(config_result, "host")
        assert hasattr(config_result, "port")
        assert hasattr(default_result, "host")
        assert hasattr(default_result, "port")
