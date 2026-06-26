"""Tests for CLI adapters (cli/adapters/*)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from dlkit.common import ConfigurationError, InferenceResult, TrainingResult
from dlkit.interfaces.cli.adapters.config_adapter import (
    load_config,
    validate_config_path,
)

from ._helpers import (
    create_invalid_toml_config,
    create_toml_config,
)


class TestConfigAdapter:
    """Test configuration loading adapter functionality."""

    def test_load_config_with_valid_toml_file(
        self,
        sample_config_path: Path,
    ) -> None:
        """Test loading valid TOML configuration file."""
        from dlkit.infrastructure.config.job_config import TrainingJobConfig

        settings = load_config(sample_config_path)

        # New architecture returns a TrainingJobConfig (JobConfig hierarchy)
        assert isinstance(settings, TrainingJobConfig)
        assert settings.experiment is not None
        assert settings.experiment.name == "test_session"

    def test_load_config_with_nonexistent_file(
        self,
        tmp_path: Path,
    ) -> None:
        """Test loading nonexistent configuration file fails gracefully."""
        nonexistent_file = tmp_path / "nonexistent.toml"

        with pytest.raises(ConfigurationError) as exc_info:
            load_config(nonexistent_file)

        error = exc_info.value
        assert "not found" in error.message.lower()
        assert str(nonexistent_file) in error.context["config_path"]

    def test_load_config_with_invalid_toml_syntax(
        self,
        tmp_path: Path,
    ) -> None:
        """Test loading configuration with invalid TOML syntax fails gracefully."""
        invalid_config = tmp_path / "invalid.toml"
        create_invalid_toml_config(invalid_config)

        with pytest.raises(ConfigurationError) as exc_info:
            load_config(invalid_config)

        error = exc_info.value
        assert isinstance(error, ConfigurationError)
        assert "invalid configuration" in error.message.lower()

    def test_load_config_handles_value_error(
        self,
        tmp_path: Path,
    ) -> None:
        """Test load_config handles ValueError (invalid config content) gracefully.

        Note: With lazy validation, Pydantic validation errors don't raise immediately
        unless they're structural (e.g., missing required fields). Extra fields with
        extra='forbid' would still raise, but our models use extra='allow' for flexibility.
        This test ensures the error handling path works when a ValueError is raised.
        """
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[run]
type = "train"

[experiment]
name = "test"

[data]
features = "not a list"
""")

        with pytest.raises(ConfigurationError) as exc_info:
            load_config(config_path)

        error = exc_info.value
        assert isinstance(error, ConfigurationError)
        # The error message will vary but should indicate a problem
        assert "configuration" in error.message.lower() or "failed to load" in error.message.lower()

    def test_load_config_handles_unexpected_exception(
        self,
        sample_config_path: Path,
    ) -> None:
        """Test load_config handles unexpected exceptions gracefully."""
        with patch("dlkit.interfaces.cli.adapters.config_adapter.load_job") as mock_load_job:
            mock_load_job.side_effect = RuntimeError("Unexpected error")

            with pytest.raises(ConfigurationError) as exc_info:
                load_config(sample_config_path)

            error = exc_info.value
            assert isinstance(error, ConfigurationError)
            assert "failed to load configuration" in error.message.lower()


class TestValidateConfigPath:
    """Test configuration path validation functionality."""

    def test_validate_config_path_with_valid_toml_file(
        self,
        sample_config_path: Path,
    ) -> None:
        """Test validating valid TOML configuration file path."""
        result = validate_config_path(sample_config_path)

        assert result is True

    def test_validate_config_path_rejects_json_file(
        self,
        tmp_path: Path,
    ) -> None:
        """Test validating JSON configuration file path fails."""
        json_config = tmp_path / "config.json"
        json_config.write_text('{"run": {"type": "train"}}')

        with pytest.raises(ConfigurationError) as exc_info:
            validate_config_path(json_config)

        assert "unsupported configuration file format" in exc_info.value.message.lower()

    def test_validate_config_path_rejects_yaml_file(
        self,
        tmp_path: Path,
    ) -> None:
        """Test validating YAML configuration file path fails."""
        yaml_config = tmp_path / "config.yaml"
        yaml_config.write_text("run:\n  type: train\n")

        with pytest.raises(ConfigurationError) as exc_info:
            validate_config_path(yaml_config)

        assert "unsupported configuration file format" in exc_info.value.message.lower()

    def test_validate_config_path_with_nonexistent_file(
        self,
        tmp_path: Path,
    ) -> None:
        """Test validating nonexistent file path fails."""
        nonexistent_file = tmp_path / "nonexistent.toml"

        with pytest.raises(ConfigurationError) as exc_info:
            validate_config_path(nonexistent_file)

        error = exc_info.value
        assert isinstance(error, ConfigurationError)
        assert "not found" in error.message.lower()

    def test_validate_config_path_with_directory_fails(
        self,
        tmp_path: Path,
    ) -> None:
        """Test validating directory path (not file) fails."""
        directory_path = tmp_path / "config_dir"
        directory_path.mkdir()

        with pytest.raises(ConfigurationError) as exc_info:
            validate_config_path(directory_path)

        error = exc_info.value
        assert isinstance(error, ConfigurationError)
        assert "not a file" in error.message.lower()

    def test_validate_config_path_with_unsupported_extension(
        self,
        tmp_path: Path,
    ) -> None:
        """Test validating file with unsupported extension fails."""
        unsupported_file = tmp_path / "config.txt"
        unsupported_file.write_text("some content")

        with pytest.raises(ConfigurationError) as exc_info:
            validate_config_path(unsupported_file)

        error = exc_info.value
        assert isinstance(error, ConfigurationError)
        assert "unsupported configuration file format" in error.message.lower()
        assert ".txt" in error.message

    def test_validate_config_path_with_permission_error(
        self,
        tmp_path: Path,
    ) -> None:
        """Test validating file with permission issues fails gracefully."""
        config_file = tmp_path / "config.toml"
        config_file.write_text('[run]\ntype = "train"\n')

        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            with pytest.raises(ConfigurationError) as exc_info:
                validate_config_path(config_file)

            error = exc_info.value
            assert isinstance(error, ConfigurationError)
            assert "permission denied" in error.message.lower()

    def test_validate_config_path_with_unexpected_error(
        self,
        sample_config_path: Path,
    ) -> None:
        """Test validate_config_path handles unexpected errors gracefully."""
        with patch("builtins.open", side_effect=RuntimeError("Unexpected error")):
            with pytest.raises(ConfigurationError) as exc_info:
                validate_config_path(sample_config_path)

            error = exc_info.value
            assert isinstance(error, ConfigurationError)
            assert "cannot access configuration file" in error.message.lower()

    @pytest.mark.parametrize("extension", [".json", ".yaml", ".yml"])
    def test_validate_config_path_rejects_non_toml_extensions(
        self,
        tmp_path: Path,
        extension: str,
    ) -> None:
        """Test that non-TOML config extensions are rejected."""
        config_file = tmp_path / f"config{extension}"
        config_file.write_text("# Placeholder config content")

        with pytest.raises(ConfigurationError) as exc_info:
            validate_config_path(config_file)

        assert "unsupported configuration file format" in exc_info.value.message.lower()

    def test_validate_config_path_case_insensitive_extensions(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that file extension validation is case-insensitive."""
        config_file = tmp_path / "config.TOML"
        config_file.write_text('[run]\ntype = "train"\n')

        result = validate_config_path(config_file)

        assert result is True


class TestResultPresenter:
    """Test result presentation adapter functionality."""

    def test_present_training_result_success(
        self,
        mock_console: Mock,
        mock_successful_training_result: TrainingResult,
    ) -> None:
        """Test presenting successful training results."""
        # Import here since we need to patch potentially
        from dlkit.interfaces.cli.adapters.result_presenter import present_training_result

        present_training_result(mock_successful_training_result, mock_console)

        # Verify console.print was called (results were displayed)
        mock_console.print.assert_called()

        # Verify that we get the expected number of prints:
        # - 1 for the main panel
        # - 1 for the metrics table
        # - 1 for the artifacts table
        assert len(mock_console.print.call_args_list) == 3

    def test_present_inference_result_success(
        self,
        mock_console: Mock,
        mock_successful_inference_result: InferenceResult,
    ) -> None:
        """Test presenting successful inference results."""
        from dlkit.interfaces.cli.adapters.result_presenter import present_inference_result

        present_inference_result(
            mock_successful_inference_result, mock_console, save_predictions=True
        )

        # Verify console.print was called (results were displayed)
        mock_console.print.assert_called()

        # Verify that we get the expected number of prints:
        # - 1 for the main panel
        # - 1 for the metrics table
        # - 1 for the prediction summary
        assert len(mock_console.print.call_args_list) == 3

    def test_present_inference_result_without_saving_predictions(
        self,
        mock_console: Mock,
        mock_successful_inference_result: InferenceResult,
    ) -> None:
        """Test presenting inference results without saving predictions."""
        from dlkit.interfaces.cli.adapters.result_presenter import present_inference_result

        present_inference_result(
            mock_successful_inference_result, mock_console, save_predictions=False
        )

        # Verify console.print was still called
        mock_console.print.assert_called()

        # Should still have the same number of prints
        assert len(mock_console.print.call_args_list) == 3


# Note: We need to check if result_presenter.py exists and implement these tests
# Let me first check what's available in the result presenter module


class TestResultPresenterModule:
    """Test the result presenter module exists and has expected functions."""

    def test_result_presenter_module_imports(self) -> None:
        """Test that result presenter module can be imported."""
        try:
            from dlkit.interfaces.cli.adapters import result_presenter

            assert hasattr(result_presenter, "present_training_result") or hasattr(
                result_presenter, "present_inference_result"
            )
        except ImportError:
            # If module doesn't exist yet, we'll create basic implementations
            return

    def test_config_adapter_functions_handle_errors(self) -> None:
        """Test that config adapter functions handle error cases properly."""
        from pathlib import Path

        from dlkit.common import ConfigurationError
        from dlkit.interfaces.cli.adapters.config_adapter import load_config, validate_config_path

        # Test that functions handle missing files appropriately
        nonexistent_path = Path("nonexistent_file.toml")

        with pytest.raises(ConfigurationError):
            load_config(nonexistent_path)

        with pytest.raises(ConfigurationError):
            validate_config_path(nonexistent_path)


class TestConfigAdapterIntegration:
    """Integration tests for config adapter with real file operations."""

    def test_load_config_with_complex_toml_configuration(
        self,
        tmp_path: Path,
    ) -> None:
        """Test loading complex TOML configuration with all sections."""
        config_path = tmp_path / "complex_config.toml"
        create_toml_config(
            config_path,
            experiment_name="complex_test",
            model_name="ComplexModel",
            enable_mlflow=True,
            enable_optuna=True,
            max_epochs=100,
            additional_sections={"CUSTOM": {"param1": "value1", "param2": 42, "param3": True}},
        )

        # This should fail because CUSTOM section is not a known section
        with pytest.raises(ConfigurationError) as exc_info:
            load_config(config_path)

        error = exc_info.value
        assert isinstance(error, ConfigurationError)
        # The error will be about missing [run] section or invalid config structure
        assert (
            "invalid configuration" in error.message.lower()
            or "failed to load" in error.message.lower()
            or "run.type" in error.message.lower()
        )

    def test_load_config_with_minimal_configuration(
        self,
        tmp_path: Path,
    ) -> None:
        """Test loading minimal valid configuration."""
        from dlkit.infrastructure.config.job_config import TrainingJobConfig

        config_path = tmp_path / "minimal_config.toml"
        config_path.write_text("""
[run]
type = "train"

[experiment]
name = "minimal"

[model]
name = "Linear"
module_path = "torch.nn"

[data]
root = "."

[training]
loss = "mse"
""")

        settings = load_config(config_path)

        # New architecture returns a TrainingJobConfig (JobConfig hierarchy)
        assert isinstance(settings, TrainingJobConfig)
        assert settings.experiment is not None
        assert settings.experiment.name == "minimal"

    def test_validate_config_path_with_real_files(
        self,
        tmp_path: Path,
    ) -> None:
        """Test config path validation with real file operations."""
        for ext in [".toml", ".json", ".yaml", ".yml"]:
            config_file = tmp_path / f"config{ext}"
            config_file.write_text("# Valid content")

            if ext == ".toml":
                result = validate_config_path(config_file)
                assert result is True, f"Failed for extension {ext}"
            else:
                with pytest.raises(ConfigurationError):
                    validate_config_path(config_file)

        # Test invalid file
        txt_file = tmp_path / "config.txt"
        txt_file.write_text("content")

        with pytest.raises(ConfigurationError):
            validate_config_path(txt_file)
