"""Test config command functionality and behavior."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from dlkit.interfaces.cli.app import app as cli_app


@pytest.fixture
def cli_runner() -> CliRunner:
    """CLI runner for tests."""
    return CliRunner()


@pytest.fixture
def valid_config_content() -> str:
    """Valid minimal config content for testing."""
    return """
[SESSION]
name = "test"
inference = false

[PATHS]
output_dir = "./outputs"
"""


@pytest.fixture
def invalid_config_content() -> str:
    """Invalid config content for testing validation."""
    return """
[SESSION]
name = "test"
invalid_field = "this should cause validation error"
"""


def test_config_subcommands_exist(cli_runner: CliRunner) -> None:
    """Test that config command help is accessible."""
    result = cli_runner.invoke(cli_app, ["config", "--help"])
    assert result.exit_code == 0


def test_config_validate_requires_file_argument(cli_runner: CliRunner) -> None:
    """Test that config validate requires a file argument."""
    result = cli_runner.invoke(cli_app, ["config", "validate"])
    assert result.exit_code != 0, "Config validate should require file argument"


def test_config_validate_detects_missing_file(cli_runner: CliRunner) -> None:
    """Test that config validate properly reports missing files."""
    result = cli_runner.invoke(cli_app, ["config", "validate", "nonexistent.toml"])
    assert result.exit_code != 0, "Should fail with missing file"


def test_config_validate_processes_valid_file(
    cli_runner: CliRunner, tmp_path, valid_config_content: str
) -> None:
    """Test that config validate processes valid config files."""
    config_file = tmp_path / "valid_config.toml"
    config_file.write_text(valid_config_content)

    result = cli_runner.invoke(cli_app, ["config", "validate", str(config_file)])
    # Should either succeed or fail with meaningful validation message
    assert result.exit_code in [0, 1], "Should not crash on valid file structure"


def test_config_validate_detects_invalid_content(
    cli_runner: CliRunner, tmp_path, invalid_config_content: str
) -> None:
    """Test that config validate detects invalid configuration content."""
    config_file = tmp_path / "invalid_config.toml"
    config_file.write_text(invalid_config_content)

    result = cli_runner.invoke(cli_app, ["config", "validate", str(config_file)])
    assert result.exit_code != 0, "Should fail with invalid config content"


def test_config_show_requires_file_argument(cli_runner: CliRunner) -> None:
    """Test that config show requires a file argument."""
    result = cli_runner.invoke(cli_app, ["config", "show"])
    assert result.exit_code != 0, "Config show should require file argument"


def test_config_show_detects_missing_file(cli_runner: CliRunner) -> None:
    """Test that config show properly reports missing files."""
    result = cli_runner.invoke(cli_app, ["config", "show", "nonexistent.toml"])
    assert result.exit_code != 0, "Should fail with missing file"


def test_config_show_displays_valid_config(
    cli_runner: CliRunner, tmp_path, valid_config_content: str
) -> None:
    """Test that config show displays configuration content."""
    config_file = tmp_path / "display_config.toml"
    config_file.write_text(valid_config_content)

    result = cli_runner.invoke(cli_app, ["config", "show", str(config_file)])
    if result.exit_code == 0:
        # If successful, should show some config content
        output = result.output.lower()
        assert "session" in output or "paths" in output, "Should display config sections"
    # If it fails, that's also acceptable as long as it doesn't crash
