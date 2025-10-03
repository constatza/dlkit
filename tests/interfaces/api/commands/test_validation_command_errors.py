"""Test ValidationCommand error scenarios."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from dlkit.interfaces.api.commands.validation_command import (
    ValidationCommand,
    ValidationCommandInput,
)
from dlkit.interfaces.api.domain.errors import WorkflowError
from dlkit.tools.config import GeneralSettings


def test_execute_missing_model_section(missing_model_settings: Mock) -> None:
    """Test validation failure when MODEL section is missing."""
    command = ValidationCommand()
    input_data = ValidationCommandInput(dry_build=False)

    with pytest.raises(WorkflowError) as exc_info:
        command.execute(input_data, missing_model_settings)

    error = exc_info.value
    assert "Configuration validation failed (training)" in error.message
    assert "[MODEL] section is required" in error.message
    assert error.context["command"] == "validate_config"
    assert error.context["profile"] == "training"


def test_execute_missing_dataset_section(missing_dataset_settings: Mock) -> None:
    """Test validation failure when DATASET section is missing."""
    command = ValidationCommand()
    input_data = ValidationCommandInput(dry_build=False)

    with pytest.raises(WorkflowError) as exc_info:
        command.execute(input_data, missing_dataset_settings)

    error = exc_info.value
    assert "Configuration validation failed (training)" in error.message
    assert "[DATASET] section is required" in error.message
    assert error.context["command"] == "validate_config"
    assert error.context["profile"] == "training"


def test_execute_missing_datamodule_section(missing_datamodule_settings: Mock) -> None:
    """Test validation failure when DATAMODULE section is missing."""
    command = ValidationCommand()
    input_data = ValidationCommandInput(dry_build=False)

    with pytest.raises(WorkflowError) as exc_info:
        command.execute(input_data, missing_datamodule_settings)

    error = exc_info.value
    assert "Configuration validation failed (training)" in error.message
    assert "[DATAMODULE] section is required" in error.message
    assert error.context["command"] == "validate_config"
    assert error.context["profile"] == "training"


def test_execute_missing_training_section_in_training_mode(missing_training_settings: Mock) -> None:
    """Test validation failure when TRAINING section is missing in training mode."""
    command = ValidationCommand()
    input_data = ValidationCommandInput(dry_build=False)

    with pytest.raises(WorkflowError) as exc_info:
        command.execute(input_data, missing_training_settings)

    error = exc_info.value
    assert "Configuration validation failed (training)" in error.message
    assert "[TRAINING] section is required for training" in error.message
    assert error.context["command"] == "validate_config"
    assert error.context["profile"] == "training"


def test_execute_missing_checkpoint_in_inference_mode(
    missing_checkpoint_inference_settings: Mock,
) -> None:
    """Test validation failure when MODEL.checkpoint is missing in inference mode."""
    command = ValidationCommand()
    input_data = ValidationCommandInput(dry_build=False)

    with pytest.raises(WorkflowError) as exc_info:
        command.execute(input_data, missing_checkpoint_inference_settings)

    error = exc_info.value
    assert "Configuration validation failed (inference)" in error.message
    assert "[MODEL.checkpoint] is required for inference mode" in error.message
    assert error.context["command"] == "validate_config"
    assert error.context["profile"] == "inference"


def test_execute_mlflow_import_error(mlflow_active_settings: Mock) -> None:
    """Test validation failure when MLflow import fails."""
    command = ValidationCommand()
    input_data = ValidationCommandInput(dry_build=False)

    with patch("builtins.__import__") as mock_import:
        # Mock MLflow import failure
        mock_import.side_effect = ImportError("No module named 'mlflow'")

        with pytest.raises(WorkflowError) as exc_info:
            command.execute(input_data, mlflow_active_settings)

        error = exc_info.value
        # Test behavior: should raise WorkflowError when mlflow is missing
        assert isinstance(error, WorkflowError)


def test_execute_optuna_import_error(optuna_active_settings: Mock) -> None:
    """Test validation failure when Optuna import fails."""
    command = ValidationCommand()
    input_data = ValidationCommandInput(dry_build=False)

    with patch("builtins.__import__") as mock_import:
        # Mock Optuna import failure
        mock_import.side_effect = ImportError("No module named 'optuna'")

        with pytest.raises(WorkflowError) as exc_info:
            command.execute(input_data, optuna_active_settings)

        error = exc_info.value
        # Test behavior: should raise WorkflowError when optuna is missing
        assert isinstance(error, WorkflowError)


def test_execute_auto_detect_mlflow_import_error(mlflow_active_settings: Mock) -> None:
    """Test validation failure when auto-detected MLflow import fails."""
    command = ValidationCommand()
    input_data = ValidationCommandInput(dry_build=False)

    with patch("builtins.__import__") as mock_import:
        # Mock MLflow import failure
        mock_import.side_effect = ImportError("No module named 'mlflow'")

        with pytest.raises(WorkflowError) as exc_info:
            command.execute(input_data, mlflow_active_settings)

        error = exc_info.value
        # Test behavior: should raise WorkflowError when mlflow auto-detection fails
        assert isinstance(error, WorkflowError)


def test_execute_auto_detect_optuna_import_error(optuna_active_settings: Mock) -> None:
    """Test validation failure when auto-detected Optuna import fails."""
    command = ValidationCommand()
    input_data = ValidationCommandInput(dry_build=False)

    with patch("builtins.__import__") as mock_import:
        # Mock Optuna import failure
        mock_import.side_effect = ImportError("No module named 'optuna'")

        with pytest.raises(WorkflowError) as exc_info:
            command.execute(input_data, optuna_active_settings)

        error = exc_info.value
        # Test behavior: should raise WorkflowError when optuna auto-detection fails
        assert isinstance(error, WorkflowError)


def test_execute_dry_build_failure(valid_training_settings: Mock) -> None:
    """Test validation failure when dry build fails."""
    command = ValidationCommand()
    input_data = ValidationCommandInput(dry_build=True)

    with patch(
        "dlkit.runtime.workflows.factories.build_factory.BuildFactory"
    ) as mock_factory_class:
        mock_factory = Mock()
        mock_factory.build_components.side_effect = Exception("Component build failed")
        mock_factory_class.return_value = mock_factory

        with pytest.raises(WorkflowError) as exc_info:
            command.execute(input_data, valid_training_settings)

        error = exc_info.value
        assert "Configuration validation failed (training)" in error.message
        assert "Dry build failed: Component build failed" in error.message
        assert error.context["command"] == "validate_config"
        assert error.context["profile"] == "training"


def test_execute_generic_exception_wrapping(valid_training_settings: Mock) -> None:
    """Test that generic exceptions are wrapped in WorkflowError."""
    command = ValidationCommand()
    input_data = ValidationCommandInput(dry_build=False)

    # Mock a generic exception during execution
    with patch.object(command, "validate_input", side_effect=ValueError("Generic error")):
        with pytest.raises(WorkflowError) as exc_info:
            command.execute(input_data, valid_training_settings)

        error = exc_info.value
        assert "Validation execution failed: Generic error" in error.message
        assert error.context["command"] == "validate_config"
        assert error.context["error_type"] == "ValueError"
        assert isinstance(error.__cause__, ValueError)


def test_validate_input_exception_wrapping(valid_training_settings: Mock) -> None:
    """Test that validate_input exceptions are wrapped in WorkflowError."""
    command = ValidationCommand()
    input_data = ValidationCommandInput()

    # Mock an exception in validate_input
    with patch.object(command, "validate_input") as mock_validate:
        mock_validate.side_effect = ValueError("Validation error")

        with pytest.raises(WorkflowError) as exc_info:
            command.execute(input_data, valid_training_settings)

        error = exc_info.value
        assert "Validation execution failed: Validation error" in error.message
        assert error.context["error_type"] == "ValueError"


def test_execute_workflow_error_passthrough(missing_model_settings: Mock) -> None:
    """Test that WorkflowError exceptions are passed through without wrapping."""
    command = ValidationCommand()
    input_data = ValidationCommandInput(dry_build=False)

    # This should raise a WorkflowError directly, not wrapped
    with pytest.raises(WorkflowError) as exc_info:
        command.execute(input_data, missing_model_settings)

    error = exc_info.value
    assert "Configuration validation failed (training)" in error.message
    assert "[MODEL] section is required" in error.message
    # Should not have nested WorkflowError wrapping


def test_execute_import_exception_handling_robustness() -> None:
    """Test that import exception handling is robust to edge cases."""
    command = ValidationCommand()
    input_data = ValidationCommandInput(dry_build=False)

    # Create settings that might cause issues during import checks
    settings = Mock(spec=GeneralSettings)
    settings.MODEL = Mock()
    settings.DATASET = Mock()
    settings.DATAMODULE = Mock()
    settings.TRAINING = Mock()
    settings.SESSION = Mock()
    settings.SESSION.inference = False
    settings.MLFLOW = Mock()
    settings.MLFLOW.enabled = True
    settings.OPTUNA = Mock()
    settings.OPTUNA.enabled = False

    with patch("builtins.__import__") as mock_import:
        # Mock a different type of exception during import
        mock_import.side_effect = AttributeError("module has no attribute")

        with pytest.raises(WorkflowError) as exc_info:
            command.execute(input_data, settings)

        error = exc_info.value
        # Test behavior: should raise WorkflowError for import attribute errors
        assert isinstance(error, WorkflowError)
