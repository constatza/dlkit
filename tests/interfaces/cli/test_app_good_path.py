"""Test CLI application core functionality."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from dlkit.interfaces.cli.app import app as cli_app


@pytest.fixture
def cli_runner() -> CliRunner:
    """CLI runner for tests."""
    return CliRunner()


def test_cli_app_no_args_shows_help(cli_runner: CliRunner) -> None:
    """Test that CLI shows help when invoked without arguments."""
    result = cli_runner.invoke(cli_app, [])
    assert result.exit_code == 2  # Typer help exit code


def test_info_command_shows_system_info(cli_runner: CliRunner) -> None:
    """Test info command returns success."""
    result = cli_runner.invoke(cli_app, ["info"])
    assert result.exit_code == 0


def test_help_flag_displays_help(cli_runner: CliRunner) -> None:
    """Test --help flag returns success."""
    result = cli_runner.invoke(cli_app, ["--help"])
    assert result.exit_code == 0
    assert "dlkit" in result.output.lower()


def test_invalid_command_returns_error(cli_runner: CliRunner) -> None:
    """Test invalid command returns non-zero exit code."""
    result = cli_runner.invoke(cli_app, ["nonexistent-command"])
    assert result.exit_code != 0


def test_verbose_flag_with_valid_command(cli_runner: CliRunner) -> None:
    """Test --verbose flag works with valid commands."""
    result = cli_runner.invoke(cli_app, ["--verbose", "info"])
    assert result.exit_code == 0


def test_subcommands_are_available(cli_runner: CliRunner) -> None:
    """Test that main subcommands are available and show help."""
    subcommands = ["train", "config", "predict", "optimize"]

    for subcmd in subcommands:
        result = cli_runner.invoke(cli_app, [subcmd, "--help"])
        assert result.exit_code == 0, f"Subcommand {subcmd} should have help available"
        assert "usage" in result.output.lower(), f"Help for {subcmd} should show usage"
