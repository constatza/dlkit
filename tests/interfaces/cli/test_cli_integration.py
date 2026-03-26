"""Integration tests demonstrating CLI functionality end-to-end."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from dlkit.interfaces.cli.app import app as cli_app


@pytest.fixture
def cli_runner() -> CliRunner:
    """Typer CLI test runner fixture."""
    return CliRunner()


def test_cli_main_workflow_help_and_info(cli_runner: CliRunner) -> None:
    """Test main CLI workflow: help and info commands work."""
    # Test help
    result = cli_runner.invoke(cli_app, ["--help"])
    assert result.exit_code == 0

    # Test info
    result = cli_runner.invoke(cli_app, ["info"])
    assert result.exit_code == 0


def test_all_subcommand_groups_accessible(cli_runner: CliRunner) -> None:
    """Test that all subcommand groups are accessible and have help."""
    subcommands = ["train", "predict", "optimize", "config"]

    for subcommand in subcommands:
        result = cli_runner.invoke(cli_app, [subcommand, "--help"])
        assert result.exit_code == 0


def test_train_help(cli_runner: CliRunner) -> None:
    """Train command help is accessible."""
    result = cli_runner.invoke(cli_app, ["train", "--help"])
    assert result.exit_code == 0


def test_config_subcommands_structure(cli_runner: CliRunner) -> None:
    """Test config subcommands are accessible."""
    # Test config main help
    result = cli_runner.invoke(cli_app, ["config", "--help"])
    assert result.exit_code == 0

    # Test config validate help
    result = cli_runner.invoke(cli_app, ["config", "validate", "--help"])
    assert result.exit_code == 0

    # Test config show help
    result = cli_runner.invoke(cli_app, ["config", "show", "--help"])
    assert result.exit_code == 0

    # Test config create help
    result = cli_runner.invoke(cli_app, ["config", "create", "--help"])
    assert result.exit_code == 0


def test_config_create_template_basic_functionality(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test config create template functionality."""
    output_file = tmp_path / "test_config.toml"

    result = cli_runner.invoke(
        cli_app, ["config", "create", "--output", str(output_file), "--type", "training"]
    )

    assert result.exit_code == 0
    assert output_file.exists()


def test_error_handling_for_invalid_commands(cli_runner: CliRunner) -> None:
    """Test that invalid commands return non-zero exit codes."""
    # Test invalid main command
    result = cli_runner.invoke(cli_app, ["invalid-command"])
    assert result.exit_code == 2

    # Test invalid subcommand
    result = cli_runner.invoke(cli_app, ["train", "invalid-subcmd"])
    assert result.exit_code in [1, 2]

    # Test invalid option
    result = cli_runner.invoke(cli_app, ["--invalid-option"])
    assert result.exit_code == 2


def test_cli_no_args_behavior(cli_runner: CliRunner) -> None:
    """Test CLI behavior when called without arguments."""
    result = cli_runner.invoke(cli_app, [])
    # Should show help due to no_args_is_help=True
    assert result.exit_code == 2


def test_subcommand_no_args_behavior(cli_runner: CliRunner) -> None:
    """Test subcommand behavior when called without arguments."""
    # Train without args should show help
    result = cli_runner.invoke(cli_app, ["train"])
    assert result.exit_code in [0, 2]

    # Config without args should show help
    result = cli_runner.invoke(cli_app, ["config"])
    assert result.exit_code in [0, 2]


def test_cli_info_command_accessible(cli_runner: CliRunner) -> None:
    """Test that the info command returns success."""
    result = cli_runner.invoke(cli_app, ["info"])
    assert result.exit_code == 0


def test_rich_formatting_in_output(cli_runner: CliRunner) -> None:
    """Test that commands with Rich formatting return success."""
    # Test info command
    result = cli_runner.invoke(cli_app, ["info"])
    assert result.exit_code == 0


def test_help_consistency_across_subcommands(cli_runner: CliRunner) -> None:
    """Test that help commands return success across subcommands."""
    subcommands = ["train", "predict", "optimize", "config"]

    for subcommand in subcommands:
        result = cli_runner.invoke(cli_app, [subcommand, "--help"])
        assert result.exit_code == 0


def test_cli_entrypoint_integration(cli_runner: CliRunner) -> None:
    """Test that the CLI can be invoked through its main entrypoint."""
    # Test basic CLI integration through app object
    result = cli_runner.invoke(cli_app, ["info"])
    assert result.exit_code == 0
