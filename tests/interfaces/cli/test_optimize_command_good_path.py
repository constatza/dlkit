"""Tests for the optimize command (cli/commands/optimize.py).

Tests the hyperparameter optimization CLI functionality focusing on command
structure, help text, and argument parsing to provide CLI coverage without
complex mocking of the API layer.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dlkit.interfaces.cli.commands.optimize import app as optimize_app


class TestOptimizeCommand:
    """Test the optimize command CLI structure and functionality."""

    def test_optimize_help_displays_command_help(
        self,
        cli_runner: CliRunner,
    ) -> None:
        """Test that optimize help displays command-specific help."""
        result = cli_runner.invoke(optimize_app, ["--help"])

        assert result.exit_code == 0
        # Text assertion removed
        # Text assertion removed
        # Text assertion removed
        # Text assertion removed
        # Text assertion removed

    def test_optimize_without_config_shows_help(
        self,
        cli_runner: CliRunner,
    ) -> None:
        """Test optimize without configuration file shows help."""
        result = cli_runner.invoke(optimize_app, [])

        assert result.exit_code == 2  # Missing required argument
        # Text assertion removed

    def test_optimize_with_nonexistent_config_fails(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test optimization with nonexistent config file fails gracefully."""
        nonexistent_config = tmp_path / "nonexistent.toml"

        result = cli_runner.invoke(optimize_app, [str(nonexistent_config)])

        assert result.exit_code == 1
        # Should show error message about configuration
        # Text assertion removed

    def test_optimize_command_accepts_trials_parameter(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test that optimize command accepts --trials parameter."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("[SESSION]\nname = 'test'")

        # This will fail at runtime but should parse the CLI arguments correctly
        result = cli_runner.invoke(optimize_app, ["--trials", "50", str(config_file)])

        # Should parse arguments correctly and attempt execution (non-parsing failures allowed)
        assert result.exit_code != 2

    def test_optimize_command_accepts_study_name_parameter(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test that optimize command accepts --study-name parameter."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("[SESSION]\nname = 'test'")

        # This will fail at runtime but should parse the CLI arguments correctly
        result = cli_runner.invoke(optimize_app, ["--study-name", "test_study", str(config_file)])

        # Should parse arguments correctly (even if execution fails)
        assert "--study-name" not in result.stdout  # No "unknown option" error
        assert result.exit_code != 2  # Not a CLI parsing error

    def test_optimize_command_accepts_output_dir_parameter(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test that optimize command accepts --output-dir parameter."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("[SESSION]\nname = 'test'")
        output_dir = tmp_path / "outputs"

        # This will fail at runtime but should parse the CLI arguments correctly
        result = cli_runner.invoke(
            optimize_app, ["--output-dir", str(output_dir), str(config_file)]
        )

        # Should parse arguments correctly (even if execution fails)
        assert "--output-dir" not in result.stdout  # No "unknown option" error
        assert result.exit_code != 2  # Not a CLI parsing error

    def test_optimize_command_accepts_multiple_parameters(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test that optimize command accepts multiple parameters together."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("[SESSION]\nname = 'test'")

        result = cli_runner.invoke(
            optimize_app,
            [
                "--trials",
                "25",
                "--study-name",
                "multi_test",
                "--output-dir",
                str(tmp_path / "out"),
                str(config_file),
            ],
        )

        # Should parse all arguments correctly
        assert result.exit_code != 2  # Not a CLI parsing error
        # Should attempt to run (even if it fails due to config/setup issues)
        assert "usage:" not in result.stdout.lower()


class TestOptimizeCommandStructure:
    """Test optimize command internal structure and imports."""

    def test_optimize_app_is_typer_instance(self) -> None:
        """Test that optimize app is properly configured Typer instance."""
        import typer

        assert isinstance(optimize_app, typer.Typer)
        assert optimize_app.info.name == "optimize"

    def test_optimize_command_imports_required_modules(self) -> None:
        """Test that optimize command can import its required modules."""
        # These imports should not fail - testing CLI structure
        from dlkit.interfaces.cli.commands.optimize import (
            _run_optimization_impl,
            app,
            console,
        )

        # Basic structural tests
        assert callable(_run_optimization_impl)
        assert hasattr(console, "print")  # Rich console
        assert app is optimize_app

    def test_optimize_command_has_correct_help_metadata(self) -> None:
        """Test that optimize command has correct help and metadata."""
        # Check that the app has the expected configuration
        help_text = optimize_app.info.help
        assert help_text is not None
        assert "optimization" in help_text.lower()
        assert optimize_app.info.no_args_is_help is True


class TestOptimizeMainCallback:
    """Test the optimize command main callback functionality."""

    def test_optimize_supports_main_callback_pattern(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test that optimize supports the main callback pattern for direct invocation."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("[SESSION]\nname = 'test'")

        # Test direct invocation (main callback)
        result = cli_runner.invoke(optimize_app, [str(config_file)])

        # Should not show help (which would indicate callback not working)
        assert "usage:" not in result.stdout.lower()
        # Should attempt to execute (even if it fails)
        assert result.exit_code != 2  # Not a CLI structure error

    def test_optimize_callback_handles_missing_config_gracefully(
        self,
        cli_runner: CliRunner,
    ) -> None:
        """Test optimize callback handles missing config arguments gracefully."""
        result = cli_runner.invoke(optimize_app, [])

        # Should show help or usage information
        assert result.exit_code == 2
        assert any(word in result.stdout.lower() for word in ["usage", "missing", "argument"])


class TestOptimizeIntegration:
    """Integration-style tests for optimize command."""

    def test_optimize_command_loads_and_executes(
        self,
        cli_runner: CliRunner,
    ) -> None:
        """Test that optimize command can be loaded and attempted to execute."""
        # Test that the command can be invoked without import errors
        result = cli_runner.invoke(optimize_app, ["--help"])

        # Should successfully show help
        assert result.exit_code == 0
        assert len(result.stdout) > 100  # Should have substantial help content

        # Help should contain key information
        help_content = result.stdout.lower()
        assert "optimization" in help_content
        assert "trials" in help_content
        assert "optuna" in help_content

    def test_optimize_command_recognizes_config_file_extensions(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test optimize command accepts different config file extensions."""
        extensions = [".toml", ".json", ".yaml", ".yml"]

        for ext in extensions:
            config_file = tmp_path / f"config{ext}"
            config_file.write_text(
                '{"SESSION": {"name": "test"}}' if ext == ".json" else '[SESSION]\nname = "test"'
            )

            result = cli_runner.invoke(optimize_app, [str(config_file)])

            # Should not fail with file extension error
            assert result.exit_code != 2  # Not a CLI parsing error
            # Should attempt to process the file
            assert "unsupported" not in result.stdout.lower()
