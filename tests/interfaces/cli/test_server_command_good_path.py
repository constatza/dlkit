"""Tests for server commands (cli/commands/server.py)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

from typer.testing import CliRunner

from dlkit.interfaces.cli.commands.server import app as server_app


class TestServerCommand:
    """Test server command structure and basic functionality."""

    def test_server_help_displays_command_help(
        self,
        cli_runner: CliRunner,
    ) -> None:
        """Test that server --help returns success."""
        result = cli_runner.invoke(server_app, ["--help"])
        assert result.exit_code == 0

    def test_server_without_subcommand_shows_help(
        self,
        cli_runner: CliRunner,
    ) -> None:
        """Test server without subcommand shows help."""
        result = cli_runner.invoke(server_app, [])
        assert result.exit_code == 2  # Typer returns 2 for missing subcommand


class TestServerStartCommand:
    """Test server start functionality."""

    @patch("dlkit.interfaces.cli.commands.server.ServerApplicationService")
    @patch("dlkit.interfaces.cli.commands.server._ensure_storage_setup_at_cli_level")
    def test_start_server_with_valid_config_succeeds(
        self,
        mock_storage_setup: Mock,
        mock_app_service_class: Mock,
        cli_runner: CliRunner,
        sample_config_path: Path,
    ) -> None:
        """Test starting server with valid configuration succeeds."""

        # Mock storage setup to avoid real config loading
        mock_storage_setup.return_value = None

        # Mock application service
        mock_app_service = Mock()
        mock_server_info = Mock()
        mock_server_info.url = "http://localhost:5000"
        mock_server_info.host = "localhost"
        mock_server_info.port = 5000
        mock_server_info.pid = 123
        mock_server_info.process = None
        mock_app_service.start_server.return_value = mock_server_info
        mock_app_service_class.return_value = mock_app_service

        result = cli_runner.invoke(server_app, ["start", str(sample_config_path)])

        assert result.exit_code == 0
        # Verify application service was used
        mock_app_service_class.assert_called_once()
        mock_app_service.start_server.assert_called_once_with(
            sample_config_path, None, None, None, None
        )

    @patch("dlkit.interfaces.cli.commands.server.ServerApplicationService")
    @patch("dlkit.interfaces.cli.commands.server._ensure_storage_setup_at_cli_level")
    def test_start_server_with_missing_config_fails(
        self,
        mock_storage_setup: Mock,
        mock_app_service_class: Mock,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test starting server with missing config file fails."""
        # Mock storage setup to avoid real config loading
        mock_storage_setup.side_effect = Exception("Config not found")

        missing_config = tmp_path / "missing.toml"

        result = cli_runner.invoke(server_app, ["start", str(missing_config)])

        assert result.exit_code == 1

    @patch("dlkit.interfaces.cli.commands.server.ServerApplicationService")
    @patch("dlkit.interfaces.cli.commands.server._ensure_storage_setup_at_cli_level")
    def test_start_server_with_invalid_config_fails(
        self,
        mock_storage_setup: Mock,
        mock_app_service_class: Mock,
        cli_runner: CliRunner,
        sample_config_path: Path,
        mock_configuration_error: Mock,
    ) -> None:
        """Test starting server with invalid config fails gracefully."""
        # Mock storage setup to avoid real config loading
        mock_storage_setup.side_effect = mock_configuration_error

        result = cli_runner.invoke(server_app, ["start", str(sample_config_path)])

        assert result.exit_code == 1


class TestServerStopCommand:
    """Test server stop functionality."""

    @patch("dlkit.interfaces.cli.commands.server.ServerApplicationService")
    def test_stop_server_succeeds(
        self,
        mock_app_service_class: Mock,
        cli_runner: CliRunner,
    ) -> None:
        """Test stopping server succeeds."""
        mock_app_service = Mock()
        mock_app_service.stop_server.return_value = (True, ["Server stopped successfully"])
        mock_app_service_class.return_value = mock_app_service

        result = cli_runner.invoke(server_app, ["stop"])

        assert result.exit_code == 0
        mock_app_service.stop_server.assert_called_once_with("localhost", 5000, False)

    @patch("dlkit.interfaces.servers.application_service.create_mlflow_adapter")
    def test_stop_server_with_custom_host(
        self,
        mock_create_adapter: Mock,
        cli_runner: CliRunner,
    ) -> None:
        """Test stopping server with custom host."""
        mock_adapter = Mock()
        mock_status = Mock()
        mock_status.is_running = False
        mock_adapter.check_server.return_value = mock_status
        mock_create_adapter.return_value = mock_adapter

        result = cli_runner.invoke(server_app, ["stop", "--host", "192.168.1.100"])

        assert result.exit_code == 0
        # Verify check_server called with custom host first
        mock_adapter.check_server.assert_called_with("192.168.1.100", 5000)

    @patch("dlkit.interfaces.servers.application_service.create_mlflow_adapter")
    def test_stop_server_handles_failure(
        self,
        mock_create_adapter: Mock,
        cli_runner: CliRunner,
        mock_workflow_error: Mock,
    ) -> None:
        """Test stopping server handles failure gracefully."""
        # Mock actual stop failure (not just check failure)
        with patch(
            "dlkit.interfaces.cli.commands.server.ServerApplicationService"
        ) as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            mock_service.stop_server.return_value = False  # Operational failure

            result = cli_runner.invoke(server_app, ["stop"])

            assert result.exit_code == 1


class TestServerStatusCommand:
    """Test server status functionality."""

    @patch("dlkit.interfaces.servers.application_service.create_mlflow_adapter")
    def test_status_check_shows_running_server(
        self,
        mock_create_adapter: Mock,
        cli_runner: CliRunner,
    ) -> None:
        """Test status check shows running server information."""
        mock_adapter = Mock()
        mock_status = Mock()
        mock_status.is_running = True
        mock_status.url = "http://localhost:5000"
        mock_status.response_time = 0.01
        mock_status.error_message = None
        mock_adapter.check_server.return_value = mock_status
        mock_create_adapter.return_value = mock_adapter

        result = cli_runner.invoke(server_app, ["status"])

        assert result.exit_code == 0
        mock_adapter.check_server.assert_called_once()

    @patch("dlkit.interfaces.servers.application_service.create_mlflow_adapter")
    def test_status_check_shows_stopped_server(
        self,
        mock_create_adapter: Mock,
        cli_runner: CliRunner,
    ) -> None:
        """Test status check shows stopped server information."""
        mock_adapter = Mock()
        mock_status = Mock()
        mock_status.is_running = False
        mock_status.url = "http://localhost:5000"
        mock_status.response_time = None
        mock_status.error_message = "Unavailable"
        mock_adapter.check_server.return_value = mock_status
        mock_create_adapter.return_value = mock_adapter

        result = cli_runner.invoke(server_app, ["status"])

        assert result.exit_code == 1

    @patch("dlkit.interfaces.servers.application_service.create_mlflow_adapter")
    def test_status_check_with_custom_host_and_port(
        self,
        mock_create_adapter: Mock,
        cli_runner: CliRunner,
    ) -> None:
        """Test status check with custom host and port."""
        mock_adapter = Mock()
        mock_status = Mock()
        mock_status.is_running = True
        mock_status.url = "http://192.168.1.100:8080"
        mock_status.response_time = 0.01
        mock_status.error_message = None
        mock_adapter.check_server.return_value = mock_status
        mock_create_adapter.return_value = mock_adapter

        result = cli_runner.invoke(
            server_app, ["status", "--host", "192.168.1.100", "--port", "8080"]
        )

        assert result.exit_code == 0
        # Verify adapter called with custom host/port
        mock_adapter.check_server.assert_called_once_with("192.168.1.100", 8080)


class TestServerInfoCommand:
    """Test server info functionality."""

    def test_info_shows_general_server_information(
        self,
        cli_runner: CliRunner,
    ) -> None:
        """Test info command shows general server information."""
        result = cli_runner.invoke(server_app, ["info"])

        assert result.exit_code == 0

    @patch("dlkit.interfaces.cli.adapters.config_adapter.load_config")
    def test_info_with_config_file_shows_specific_settings(
        self,
        mock_load_config: Mock,
        cli_runner: CliRunner,
        sample_config_path: Path,
        sample_settings: Mock,
        tmp_path: Path,
    ) -> None:
        """Test info command with config file shows specific settings."""

        # Mock MLflow settings
        mock_mlflow_settings = Mock()
        mock_mlflow_settings.is_active = True
        mock_mlflow_settings.server = Mock()
        mock_mlflow_settings.server.host = "localhost"
        mock_mlflow_settings.server.port = 5000
        mock_mlflow_settings.server.backend_store_uri = "sqlite:///test.db"
        mock_mlflow_settings.server.artifacts_destination = str((tmp_path / "artifacts").resolve())
        mock_mlflow_settings.client = Mock()
        mock_mlflow_settings.client.tracking_uri = "http://localhost:5000"
        mock_mlflow_settings.client.experiment_name = "test_experiment"
        sample_settings.MLFLOW = mock_mlflow_settings

        mock_load_config.return_value = sample_settings

        result = cli_runner.invoke(server_app, ["info", str(sample_config_path)])

        assert result.exit_code == 0
        mock_load_config.assert_called_once()

    @patch("dlkit.interfaces.cli.adapters.config_adapter.load_config")
    def test_info_with_invalid_config_fails_gracefully(
        self,
        mock_load_config: Mock,
        cli_runner: CliRunner,
        sample_config_path: Path,
        mock_configuration_error: Mock,
    ) -> None:
        """Test info command with invalid config fails gracefully."""
        mock_load_config.side_effect = mock_configuration_error

        result = cli_runner.invoke(server_app, ["info", str(sample_config_path)])

        assert result.exit_code == 1
        mock_load_config.assert_called_once()


class TestServerCommandStructure:
    """Test server command internal structure."""

    def test_server_app_is_typer_instance(self) -> None:
        """Test that server app is properly configured Typer instance."""
        import typer

        assert isinstance(server_app, typer.Typer)
        assert server_app.info.name == "server"

    def test_server_command_has_correct_help_metadata(self) -> None:
        """Test that server command has correct help and metadata."""
        assert "server management" in server_app.info.help.lower()
        assert server_app.info.no_args_is_help is True

    def test_server_command_imports_required_modules(self) -> None:
        """Test that server command can import its required modules."""
        # These imports should not fail - testing CLI structure
        from dlkit.interfaces.cli.commands.server import (
            app,
            console,
            start_server,
            stop_server,
            check_server_status,
            show_server_info,
        )

        # Basic structural tests
        assert callable(start_server)
        assert callable(stop_server)
        assert callable(check_server_status)
        assert callable(show_server_info)
        assert hasattr(console, "print")  # Rich console
        assert app is server_app


class TestServerIntegration:
    """Integration-style tests for server command."""

    def test_server_command_loads_and_executes(
        self,
        cli_runner: CliRunner,
    ) -> None:
        """Test that server command can be loaded and attempted to execute."""
        # Test that the command can be invoked without import errors
        result = cli_runner.invoke(server_app, ["--help"])

        # Should successfully show help
        assert result.exit_code == 0

    def test_all_server_subcommands_have_help(
        self,
        cli_runner: CliRunner,
    ) -> None:
        """Test that all server subcommands have accessible help."""
        subcommands = ["start", "stop", "status", "info"]

        for subcmd in subcommands:
            result = cli_runner.invoke(server_app, [subcmd, "--help"])
            assert result.exit_code == 0

    def test_server_command_recognizes_common_arguments(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test server command accepts common server management arguments."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("[SESSION]\nname = 'test'")

        # Test argument parsing without execution
        test_cases = [
            ["start", "--help"],
            ["stop", "--help"],
            ["status", "--help"],
            ["info", "--help"],
        ]

        for args in test_cases:
            result = cli_runner.invoke(server_app, args)
            # Should not fail with CLI parsing error (exit code 2)
            assert result.exit_code == 0
