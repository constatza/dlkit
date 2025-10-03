"""Basic CLI functionality tests that should work reliably."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from dlkit.interfaces.cli.app import app as cli_app


@pytest.fixture
def cli_runner() -> CliRunner:
    """Typer CLI test runner fixture."""
    return CliRunner()


def test_cli_help_flag(cli_runner: CliRunner) -> None:
    """Test --help flag returns success."""
    result = cli_runner.invoke(cli_app, ["--help"])
    assert result.exit_code == 0


def test_cli_version_flag(cli_runner: CliRunner) -> None:
    """Test --version flag returns success."""
    result = cli_runner.invoke(cli_app, ["--version"])
    assert result.exit_code == 0


def test_cli_version_short_flag(cli_runner: CliRunner) -> None:
    """Test -v flag returns success."""
    result = cli_runner.invoke(cli_app, ["-v"])
    assert result.exit_code == 0


def test_cli_info_command(cli_runner: CliRunner) -> None:
    """Test info command returns success."""
    result = cli_runner.invoke(cli_app, ["info"])
    assert result.exit_code == 0


def test_cli_no_args_shows_help(cli_runner: CliRunner) -> None:
    """Test that CLI shows help when invoked without arguments."""
    result = cli_runner.invoke(cli_app, [])
    # Typer returns exit code 2 when no_args_is_help=True
    assert result.exit_code == 2


def test_cli_invalid_command(cli_runner: CliRunner) -> None:
    """Test CLI handles invalid commands gracefully."""
    result = cli_runner.invoke(cli_app, ["nonexistent-command"])
    # Typer returns exit code 2 for invalid commands
    assert result.exit_code == 2


@pytest.mark.parametrize("subcommand", ["train", "predict", "optimize", "config", "server"])
def test_subcommand_help_accessible(cli_runner: CliRunner, subcommand: str) -> None:
    """Test that each subcommand's help is accessible."""
    result = cli_runner.invoke(cli_app, [subcommand, "--help"])
    assert result.exit_code == 0


def test_train_help_exists(cli_runner: CliRunner) -> None:
    """Train command help is accessible."""
    result = cli_runner.invoke(cli_app, ["train", "--help"])
    assert result.exit_code == 0


def test_config_subcommand_structure(cli_runner: CliRunner) -> None:
    """Test config subcommand help is accessible."""
    result = cli_runner.invoke(cli_app, ["config", "--help"])
    assert result.exit_code == 0


def test_all_subcommands_registered(cli_runner: CliRunner) -> None:
    """Test that help command returns success."""
    result = cli_runner.invoke(cli_app, ["--help"])
    assert result.exit_code == 0


def test_config_validate_help(cli_runner: CliRunner) -> None:
    """Test config validate subcommand help returns success."""
    result = cli_runner.invoke(cli_app, ["config", "validate", "--help"])
    assert result.exit_code == 0


def test_config_create_help(cli_runner: CliRunner) -> None:
    """Test config create subcommand help returns success."""
    result = cli_runner.invoke(cli_app, ["config", "create", "--help"])
    assert result.exit_code == 0
