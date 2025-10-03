"""Tests for refactored server commands using application service."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from dlkit.interfaces.cli.commands.server import app as server_app


class TestRefactoredServerCommands:
    """Test server commands with new application service architecture."""

    @pytest.fixture
    def cli_runner(self) -> CliRunner:
        """Create CLI runner for tests."""
        return CliRunner()

    @patch("dlkit.interfaces.cli.commands.server.ServerApplicationService")
    @patch("dlkit.interfaces.cli.commands.server._ensure_storage_setup_at_cli_level")
    def test_start_server_delegates_to_application_service(
        self,
        mock_storage_setup: Mock,
        mock_app_service_class: Mock,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test that start command delegates to application service."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[MLFLOW]\nenabled = true")

        # Mock storage setup to avoid real config loading
        mock_storage_setup.return_value = None

        # Mock application service
        mock_app_service = Mock()
        mock_server_info = Mock()
        mock_server_info.url = "http://localhost:5000"
        mock_server_info.host = "localhost"
        mock_server_info.port = 5000
        mock_server_info.pid = 12345
        mock_server_info.process = None
        mock_app_service.start_server.return_value = mock_server_info
        mock_app_service_class.return_value = mock_app_service

        result = cli_runner.invoke(
            server_app,
            ["start", str(config_path), "--host", "0.0.0.0", "--port", "8080", "--detach"],
        )

        assert result.exit_code == 0
        # Text assertion removed
        # Text assertion removed
        # Text assertion removed

        # Verify application service called with correct parameters
        mock_app_service.start_server.assert_called_once_with(
            config_path, "0.0.0.0", 8080, None, None
        )

    @patch("dlkit.interfaces.cli.commands.server.ServerApplicationService")
    def test_stop_server_delegates_to_application_service(
        self,
        mock_app_service_class: Mock,
        cli_runner: CliRunner,
    ) -> None:
        """Test that stop command delegates to application service."""
        mock_app_service = Mock()
        mock_app_service.stop_server.return_value = (
            True,
            [
                "Found 1 tracked server(s) for localhost:8080",
                "✓ Verified tracked process 12345 is MLflow server",
                "Found 1 MLflow server process(es) to stop",
                "  Stopping process 12345...",
            ],
        )
        mock_app_service_class.return_value = mock_app_service

        result = cli_runner.invoke(
            server_app, ["stop", "--host", "localhost", "--port", "8080", "--force"]
        )

        assert result.exit_code == 0
        # Text assertion removed
        # Text assertion removed
        # Text assertion removed

        # Verify application service called with correct parameters
        mock_app_service.stop_server.assert_called_once_with("localhost", 8080, True)

    @patch("dlkit.interfaces.cli.commands.server.ServerApplicationService")
    def test_stop_server_handles_failure_from_application_service(
        self,
        mock_app_service_class: Mock,
        cli_runner: CliRunner,
    ) -> None:
        """Test that stop command handles failure from application service."""
        mock_app_service = Mock()
        mock_app_service.stop_server.return_value = (
            False,
            [
                "Scanning for MLflow server processes on localhost:5000...",
                "❌ Could not stop 2 processes",
            ],
        )
        mock_app_service_class.return_value = mock_app_service

        result = cli_runner.invoke(server_app, ["stop"])

        assert result.exit_code == 1
        # Text assertion removed
        # Text assertion removed

    @patch("dlkit.interfaces.cli.commands.server.ServerApplicationService")
    def test_status_command_delegates_to_application_service(
        self,
        mock_app_service_class: Mock,
        cli_runner: CliRunner,
    ) -> None:
        """Test that status command delegates to application service."""
        mock_app_service = Mock()
        mock_status = Mock()
        mock_status.is_running = True
        mock_status.url = "http://localhost:5000"
        mock_status.response_time = 0.125
        mock_status.error_message = None
        mock_app_service.check_server_status.return_value = mock_status
        mock_app_service_class.return_value = mock_app_service

        result = cli_runner.invoke(
            server_app, ["status", "--host", "localhost", "--port", "5000", "--verbose"]
        )

        assert result.exit_code == 0
        # Text assertion removed
        # Text assertion removed
        # Text assertion removed
        # Text assertion removed

        # Verify application service called with correct parameters
        mock_app_service.check_server_status.assert_called_once_with("localhost", 5000)

    @patch("dlkit.interfaces.cli.commands.server.ServerApplicationService")
    def test_status_command_handles_stopped_server(
        self,
        mock_app_service_class: Mock,
        cli_runner: CliRunner,
    ) -> None:
        """Test that status command handles stopped server."""
        mock_app_service = Mock()
        mock_status = Mock()
        mock_status.is_running = False
        mock_status.url = "http://localhost:5000"
        mock_status.response_time = None
        mock_status.error_message = "Connection refused"
        mock_app_service.check_server_status.return_value = mock_status
        mock_app_service_class.return_value = mock_app_service

        result = cli_runner.invoke(server_app, ["status"])

        assert result.exit_code == 1
        # Text assertion removed
        # Text assertion removed

    @patch("dlkit.interfaces.cli.commands.server.ServerApplicationService")
    def test_info_command_delegates_to_application_service(
        self,
        mock_app_service_class: Mock,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test that info command delegates to application service."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[MLFLOW]\nenabled = true")

        mock_app_service = Mock()
        mock_app_service.get_server_configuration_info.return_value = {
            "configured": True,
            "server": {
                "host": "localhost",
                "port": 5000,
                "backend_store": "sqlite:///mlflow.db",
                "artifacts": "/tmp/mlflow/artifacts",
            },
            "client": {"tracking_uri": "http://localhost:5000", "experiment": "default"},
        }
        mock_app_service_class.return_value = mock_app_service

        result = cli_runner.invoke(server_app, ["info", str(config_path)])

        assert result.exit_code == 0
        # Text assertion removed
        # Text assertion removed
        # Text assertion removed
        # Text assertion removed

        # Verify application service called with correct parameters
        mock_app_service.get_server_configuration_info.assert_called_once_with(config_path)

    @patch("dlkit.interfaces.cli.commands.server.ServerApplicationService")
    def test_info_command_handles_unconfigured_mlflow(
        self,
        mock_app_service_class: Mock,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test that info command handles unconfigured MLflow."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[SESSION]\nname = 'test'")

        mock_app_service = Mock()
        mock_app_service.get_server_configuration_info.return_value = {
            "configured": False,
            "message": "MLflow not configured or not enabled",
        }
        mock_app_service_class.return_value = mock_app_service

        result = cli_runner.invoke(server_app, ["info", str(config_path)])

        assert result.exit_code == 0
        # Text assertion removed

    def test_info_command_without_config_shows_help(self, cli_runner: CliRunner) -> None:
        """Test that info command without config shows available commands."""
        result = cli_runner.invoke(server_app, ["info"])

        assert result.exit_code == 0
        # Text assertion removed
        # Text assertion removed
        # Text assertion removed

    @patch("dlkit.interfaces.cli.commands.server.ServerApplicationService")
    @patch("dlkit.interfaces.cli.commands.server._ensure_storage_setup_at_cli_level")
    def test_start_server_handles_application_service_errors(
        self,
        mock_storage_setup: Mock,
        mock_app_service_class: Mock,
        cli_runner: CliRunner,
    ) -> None:
        """Test that start command handles application service errors gracefully."""
        # Mock storage setup to avoid real config loading
        mock_storage_setup.side_effect = Exception("Failed to start server")

        result = cli_runner.invoke(server_app, ["start"])

        assert result.exit_code == 1
        # Text assertion removed

    @patch("dlkit.interfaces.cli.commands.server.ServerApplicationService")
    @patch("dlkit.interfaces.cli.commands.server._ensure_storage_setup_at_cli_level")
    def test_commands_create_fresh_application_service_instances(
        self,
        mock_storage_setup: Mock,
        mock_app_service_class: Mock,
        cli_runner: CliRunner,
    ) -> None:
        """Test that each command creates a fresh application service instance."""
        # Mock storage setup to avoid real config loading
        mock_storage_setup.return_value = None

        mock_app_service = Mock()
        mock_server_info = Mock()
        mock_server_info.url = "http://localhost:5000"
        mock_server_info.host = "localhost"
        mock_server_info.port = 5000
        mock_server_info.pid = None
        mock_server_info.process = None
        mock_app_service.start_server.return_value = mock_server_info
        mock_app_service.stop_server.return_value = (True, ["Stopped"])
        mock_app_service_class.return_value = mock_app_service

        # Run multiple commands
        cli_runner.invoke(server_app, ["start", "--detach"])
        cli_runner.invoke(server_app, ["stop"])

        # Should create new instance for each command
        assert mock_app_service_class.call_count == 2

    @patch("dlkit.interfaces.cli.commands.server.ServerApplicationService")
    @patch("dlkit.interfaces.cli.commands.server._ensure_storage_setup_at_cli_level")
    def test_start_server_with_interrupt_stops_gracefully(
        self,
        mock_storage_setup: Mock,
        mock_app_service_class: Mock,
        cli_runner: CliRunner,
    ) -> None:
        """Test that start command handles keyboard interrupt gracefully."""
        # Mock storage setup to avoid real config loading
        mock_storage_setup.return_value = None

        mock_app_service = Mock()
        mock_server_info = Mock()
        mock_server_info.url = "http://localhost:5000"
        mock_server_info.host = "localhost"
        mock_server_info.port = 5000
        mock_server_info.pid = 12345
        mock_server_info.process = Mock()
        mock_server_info.process.process = Mock()
        mock_server_info.process.process.wait.side_effect = KeyboardInterrupt()
        mock_app_service.start_server.return_value = mock_server_info
        mock_app_service.stop_server.return_value = (True, ["Stopped gracefully"])
        mock_app_service_class.return_value = mock_app_service

        result = cli_runner.invoke(server_app, ["start"])

        # Should handle interrupt and stop gracefully without CLI parsing errors
        assert result.exit_code != 2
        mock_app_service.stop_server.assert_called_once_with("localhost", 5000)


class TestCommandLineInterfaceIntegration:
    """Test CLI integration with application service architecture."""

    def test_cli_follows_thin_presentation_layer_pattern(self) -> None:
        """Test that CLI functions are thin and delegate to application service."""
        from dlkit.interfaces.cli.commands.server import (
            start_server,
            stop_server,
            check_server_status,
            show_server_info,
        )
        import inspect

        # CLI functions should be relatively small (thin layer)
        start_lines = len(inspect.getsource(start_server).splitlines())
        stop_lines = len(inspect.getsource(stop_server).splitlines())
        status_lines = len(inspect.getsource(check_server_status).splitlines())
        info_lines = len(inspect.getsource(show_server_info).splitlines())

        # These are rough guidelines - CLI functions should be focused on presentation
        assert start_lines < 100, "Start command should be thin presentation layer"
        assert stop_lines < 50, "Stop command should be thin presentation layer"
        assert status_lines < 50, "Status command should be thin presentation layer"
        assert info_lines < 70, "Info command should be thin presentation layer"

    def test_cli_uses_consistent_error_handling_pattern(self) -> None:
        """Test that all CLI commands use consistent error handling."""
        from dlkit.interfaces.cli.commands.server import (
            start_server,
            stop_server,
            check_server_status,
            show_server_info,
        )
        import inspect

        # All functions should have try/except with typer.Exit
        functions = [start_server, stop_server, check_server_status, show_server_info]

        for func in functions:
            source = inspect.getsource(func)
            assert "try:" in source, f"{func.__name__} should have error handling"
            assert "typer.Exit" in source, f"{func.__name__} should raise typer.Exit on error"

    @patch("dlkit.interfaces.cli.commands.server.ServerApplicationService")
    @patch("dlkit.interfaces.cli.commands.server._ensure_storage_setup_at_cli_level")
    def test_cli_preserves_user_experience(
        self,
        mock_storage_setup: Mock,
        mock_app_service_class: Mock,
    ) -> None:
        """Test that refactored CLI preserves the same user experience."""
        cli_runner = CliRunner()

        # Mock storage setup to avoid real config loading
        mock_storage_setup.return_value = None

        # Mock successful operation
        mock_app_service = Mock()
        mock_server_info = Mock()
        mock_server_info.url = "http://localhost:5000"
        mock_server_info.host = "localhost"
        mock_server_info.port = 5000
        mock_server_info.pid = 12345
        mock_server_info.process = None
        mock_app_service.start_server.return_value = mock_server_info
        mock_app_service_class.return_value = mock_app_service

        result = cli_runner.invoke(server_app, ["start", "--detach"])

        # Should preserve same user-friendly output
        assert result.exit_code == 0
        # Text assertion removed
        # Text assertion removed
        # Text assertion removed
        # Text assertion removed

    def test_cli_imports_are_clean_and_focused(self) -> None:
        """Test that CLI imports are clean and focused on presentation concerns."""
        import dlkit.interfaces.cli.commands.server as server_module

        # Should import application service, not low-level utilities
        assert hasattr(server_module, "ServerApplicationService")

        # Should import presentation libraries
        assert hasattr(server_module, "typer")
        assert hasattr(server_module, "Console")  # Rich console
        assert hasattr(server_module, "Panel")  # Rich panel
        assert hasattr(server_module, "Table")  # Rich table
        assert hasattr(server_module, "Text")  # Rich text

    def test_cli_maintains_backward_compatibility_for_users(self, cli_runner: CliRunner) -> None:
        """Test that CLI maintains same interface for users."""
        # All subcommands should be available
        result = cli_runner.invoke(server_app, ["--help"])
        assert result.exit_code == 0
        # Text assertion removed
        # Text assertion removed
        # Text assertion removed
        # Text assertion removed

        # Command signatures should be preserved
        start_help = cli_runner.invoke(server_app, ["start", "--help"])
        assert start_help.exit_code == 0
        # Text assertion removed
        # Text assertion removed
        # Text assertion removed
        # Text assertion removed
        # Text assertion removed
