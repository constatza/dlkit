"""Test ValidationCommand good path scenarios."""

from __future__ import annotations

from unittest.mock import Mock, patch


from dlkit.interfaces.api.commands.validation_command import (
    ValidationCommand,
    ValidationCommandInput,
)


def test_validation_command_initialization() -> None:
    """Test ValidationCommand initializes correctly."""
    command = ValidationCommand()
    assert command.command_name == "validate_config"

    custom_command = ValidationCommand("custom_validate")
    assert custom_command.command_name == "custom_validate"


def test_validation_command_input_dataclass_creation() -> None:
    """Test ValidationCommandInput dataclass creation with all parameters."""
    input_data = ValidationCommandInput(dry_build=True)

    assert input_data.dry_build is True


def test_validation_command_input_dataclass_defaults() -> None:
    """Test ValidationCommandInput dataclass with default values."""
    input_data = ValidationCommandInput()

    assert input_data.dry_build is False


def test_validate_input_passes_with_valid_data(valid_training_settings: Mock) -> None:
    """Test validate_input passes with valid input"""
    command = ValidationCommand()
    input_data = ValidationCommandInput(dry_build=False)

    # Should not raise any exception
    command.validate_input(input_data, valid_training_settings)


def test_execute_valid_training_configuration(valid_training_settings: Mock) -> None:
    """Test successful validation for training profile."""
    command = ValidationCommand()
    input_data = ValidationCommandInput(dry_build=False)

    result = command.execute(input_data, valid_training_settings)

    assert result is True


def test_execute_valid_inference_configuration(
    valid_inference_settings: Mock,
) -> None:
    """Test successful validation for inference profile."""
    command = ValidationCommand()
    input_data = ValidationCommandInput(dry_build=False)

    result = command.execute(input_data, valid_inference_settings)

    assert result is True


def test_execute_training_profile_without_plugins(valid_training_settings: Mock) -> None:
    """Test validation passes when no plugins are active."""
    command = ValidationCommand()
    input_data = ValidationCommandInput(dry_build=False)

    result = command.execute(input_data, valid_training_settings)

    assert result is True


def test_execute_with_mlflow_enabled(mlflow_active_settings: Mock) -> None:
    """Test validation when MLflow is enabled."""
    command = ValidationCommand()
    input_data = ValidationCommandInput(dry_build=False)

    with patch("builtins.__import__") as mock_import:
        # Mock successful MLflow import
        mock_import.return_value = Mock()

        result = command.execute(input_data, mlflow_active_settings)

        assert result is True


def test_execute_with_optuna_enabled(optuna_active_settings: Mock) -> None:
    """Test validation when Optuna is enabled."""
    command = ValidationCommand()
    input_data = ValidationCommandInput(dry_build=False)

    with patch("builtins.__import__") as mock_import:
        # Mock successful Optuna import
        mock_import.return_value = Mock()

        result = command.execute(input_data, optuna_active_settings)

        assert result is True


def test_execute_with_dry_build_success(valid_training_settings: Mock) -> None:
    """Test successful validation with dry build enabled."""
    command = ValidationCommand()
    input_data = ValidationCommandInput(dry_build=True)

    with patch(
        "dlkit.runtime.workflows.factories.build_factory.BuildFactory"
    ) as mock_factory_class:
        mock_factory = Mock()
        mock_factory.build_components.return_value = Mock()
        mock_factory_class.return_value = mock_factory

        result = command.execute(input_data, valid_training_settings)

        assert result is True
        mock_factory.build_components.assert_called_once_with(valid_training_settings)


def test_execute_with_none_session_settings(valid_training_settings: Mock) -> None:
    """Test validation when SESSION is None (defaults to training mode)."""
    # Modify settings to have None SESSION
    valid_training_settings.SESSION = None

    command = ValidationCommand()
    input_data = ValidationCommandInput(dry_build=False)

    result = command.execute(input_data, valid_training_settings)

    assert result is True


def test_execute_with_none_session_inference_attribute(valid_training_settings: Mock) -> None:
    """Test validation when SESSION exists but inference attribute is None."""
    # Modify settings to have SESSION without inference attribute
    valid_training_settings.SESSION = Mock()
    del valid_training_settings.SESSION.inference  # Remove the attribute

    command = ValidationCommand()
    input_data = ValidationCommandInput(dry_build=False)

    result = command.execute(input_data, valid_training_settings)

    assert result is True
