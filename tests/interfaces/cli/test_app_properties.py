"""Test CLI application behavior properties and edge cases."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from dlkit.interfaces.cli.app import app as cli_app


@pytest.fixture
def cli_runner() -> CliRunner:
    """CLI runner for tests."""
    return CliRunner()


def test_cli_returns_standard_exit_codes(cli_runner: CliRunner) -> None:
    """Test that CLI returns standard Unix exit codes."""
    # Valid commands should return 0
    valid_commands = [["--help"], ["--version"], ["info"]]
    for args in valid_commands:
        result = cli_runner.invoke(cli_app, args)
        assert result.exit_code == 0, f"Valid command {args} should return exit code 0"

    # Invalid commands should return non-zero
    invalid_commands = [["nonexistent"], ["train", "invalid"], ["config", "invalid"]]
    for args in invalid_commands:
        result = cli_runner.invoke(cli_app, args)
        assert result.exit_code != 0, f"Invalid command {args} should return non-zero exit code"


def test_version_output_is_consistent(cli_runner: CliRunner) -> None:
    """Test that version output is consistent across different flags."""
    long_result = cli_runner.invoke(cli_app, ["--version"])
    short_result = cli_runner.invoke(cli_app, ["-v"])

    assert long_result.exit_code == 0
    assert short_result.exit_code == 0
    assert long_result.output == short_result.output


def test_help_is_comprehensive(cli_runner: CliRunner) -> None:
    """Test that help output contains necessary information."""
    result = cli_runner.invoke(cli_app, ["--help"])
    assert result.exit_code == 0

    output_lower = result.output.lower()
    required_elements = ["usage", "commands", "options", "dlkit"]
    for element in required_elements:
        assert element in output_lower, f"Help should contain '{element}'"


def test_all_subcommands_have_help(cli_runner: CliRunner) -> None:
    """Test that all main subcommands provide help documentation."""
    subcommands = ["train", "config", "predict", "optimize"]

    for subcmd in subcommands:
        result = cli_runner.invoke(cli_app, [subcmd, "--help"])
        assert result.exit_code == 0, f"Help for {subcmd} should be available"

        output_lower = result.output.lower()
        assert "usage" in output_lower, f"Help for {subcmd} should show usage"
        assert subcmd in output_lower, f"Help for {subcmd} should mention the command name"


def test_verbose_flag_works_with_commands(cli_runner: CliRunner) -> None:
    """Test that verbose flag can be combined with valid commands."""
    # Verbose should work with info command
    result = cli_runner.invoke(cli_app, ["--verbose", "info"])
    assert result.exit_code == 0, "Verbose flag should work with info command"

    # Verbose should work with version
    result = cli_runner.invoke(cli_app, ["--verbose", "--version"])
    assert result.exit_code == 0, "Verbose flag should work with version"


def test_command_chaining_validation(cli_runner: CliRunner) -> None:
    """Test that command arguments are properly validated."""
    # Multiple flags should be handled
    result = cli_runner.invoke(cli_app, ["--verbose", "--help"])
    assert result.exit_code == 0, "Multiple valid flags should work"

    # Conflicting arguments should be handled gracefully
    result = cli_runner.invoke(cli_app, ["--version", "--help"])
    # Should either work or fail gracefully (not crash)
    assert result.exit_code in {0, 1, 2}, "Conflicting flags should be handled gracefully"
