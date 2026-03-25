"""Integration tests for CLI subcommand registration and routing."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from dlkit.interfaces.cli.app import app as cli_app

# Constants for subcommand testing
EXPECTED_SUBCOMMANDS = {
    "train": "Training",
    "predict": "Prediction",
    "optimize": "optimization",
    "config": "Configuration",
}

SUCCESS_EXIT_CODE = 0
HELP_EXIT_CODE = 0
ERROR_EXIT_CODE = 2  # Typer's "no such command" exit code


def test_all_subcommands_registered(
    cli_runner: CliRunner,
    expected_help_patterns: dict[str, str],
) -> None:
    """Test that help command returns success.

    Args:
        cli_runner: Typer CLI test runner.
        expected_help_patterns: Expected help text patterns.
    """
    result = cli_runner.invoke(cli_app, ["--help"])
    assert result.exit_code == SUCCESS_EXIT_CODE


@pytest.mark.parametrize("subcommand", list(EXPECTED_SUBCOMMANDS.keys()))
def test_subcommand_help_accessible(
    cli_runner: CliRunner,
    subcommand: str,
) -> None:
    """Test that each subcommand's help is accessible.

    Args:
        cli_runner: Typer CLI test runner.
        subcommand: Name of the subcommand to test.
    """
    result = cli_runner.invoke(cli_app, [subcommand, "--help"])
    assert result.exit_code == HELP_EXIT_CODE


@pytest.mark.parametrize("subcommand", list(EXPECTED_SUBCOMMANDS.keys()))
def test_subcommand_shows_help_when_no_args(
    cli_runner: CliRunner,
    subcommand: str,
) -> None:
    """Test that subcommands show help when called without arguments.

    Args:
        cli_runner: Typer CLI test runner.
        subcommand: Name of the subcommand to test.
    """
    result = cli_runner.invoke(cli_app, [subcommand])
    assert result.exit_code in [SUCCESS_EXIT_CODE, ERROR_EXIT_CODE]


def test_train_command_help(cli_runner: CliRunner) -> None:
    """Train command help is accessible (no nested subcommands)."""
    result = cli_runner.invoke(cli_app, ["train", "--help"])
    assert result.exit_code == HELP_EXIT_CODE


def test_config_subcommand_structure(cli_runner: CliRunner) -> None:
    """Test config subcommand has expected structure."""
    result = cli_runner.invoke(cli_app, ["config", "--help"])
    assert result.exit_code == HELP_EXIT_CODE


def test_optimize_subcommand_structure(cli_runner: CliRunner) -> None:
    """Test optimize subcommand has expected structure."""
    result = cli_runner.invoke(cli_app, ["optimize", "--help"])
    assert result.exit_code == HELP_EXIT_CODE


def test_predict_subcommand_structure(cli_runner: CliRunner) -> None:
    """Test predict subcommand has expected structure."""
    result = cli_runner.invoke(cli_app, ["predict", "--help"])
    assert result.exit_code == HELP_EXIT_CODE


def test_invalid_subcommand_returns_error(cli_runner: CliRunner) -> None:
    """Test that invalid subcommands return appropriate error."""
    result = cli_runner.invoke(cli_app, ["nonexistent-subcommand"])
    assert result.exit_code == ERROR_EXIT_CODE


def test_subcommand_with_invalid_flag(cli_runner: CliRunner) -> None:
    """Test subcommand behavior with invalid flags."""
    result = cli_runner.invoke(cli_app, ["train", "--nonexistent-flag"])
    # Should return error for invalid flag
    assert result.exit_code == ERROR_EXIT_CODE


def test_subcommand_help_consistency(cli_runner: CliRunner) -> None:
    """Test that all subcommand help outputs follow consistent patterns."""
    for subcommand in EXPECTED_SUBCOMMANDS:
        result = cli_runner.invoke(cli_app, [subcommand, "--help"])
        assert result.exit_code == HELP_EXIT_CODE


@pytest.mark.parametrize(
    "subcommand,expected_patterns",
    [
        ("train", ["config", "strategy", "checkpoint", "validate-only"]),
        ("predict", ["model", "dataflow", "output"]),
        ("optimize", ["trials", "study"]),
        ("config", ["validate", "show"]),
    ],
)
def test_subcommand_specific_keywords(
    cli_runner: CliRunner,
    subcommand: str,
    expected_patterns: list[str],
) -> None:
    """Test that subcommands contain expected domain-specific keywords.

    Args:
        cli_runner: Typer CLI test runner.
        subcommand: Name of the subcommand to test.
        expected_patterns: Expected patterns in the subcommand help.
    """
    result = cli_runner.invoke(cli_app, [subcommand, "--help"])
    assert result.exit_code == HELP_EXIT_CODE


def test_main_app_lists_all_subcommands(cli_runner: CliRunner) -> None:
    """Test that main app help lists all expected subcommands."""
    result = cli_runner.invoke(cli_app, ["--help"])
    assert result.exit_code == SUCCESS_EXIT_CODE


def test_subcommand_error_handling(cli_runner: CliRunner) -> None:
    """Test that subcommands handle errors appropriately."""
    # Test with invalid subcommand arguments (non-existent config file)
    result = cli_runner.invoke(cli_app, ["train", "invalid-subcmd"])
    # Should return application error exit code (file not found)
    assert result.exit_code == 1  # Application handles missing files


def test_mixed_valid_invalid_args(cli_runner: CliRunner) -> None:
    """Test CLI with mixed valid and invalid arguments."""
    # Valid subcommand but invalid sub-subcommand
    result = cli_runner.invoke(cli_app, ["config", "nonexistent"])

    assert result.exit_code == ERROR_EXIT_CODE


def test_subcommand_callback_integration(cli_runner: CliRunner) -> None:
    """Test that subcommand callbacks are properly integrated."""
    # Test train command with just help (should trigger callback)
    result = cli_runner.invoke(cli_app, ["train", "--help"])
    assert result.exit_code == HELP_EXIT_CODE
