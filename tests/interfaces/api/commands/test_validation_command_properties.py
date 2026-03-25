"""Property-based tests for ValidationCommand using Hypothesis."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from hypothesis import given
from hypothesis import strategies as st

from dlkit.interfaces.api.commands.validation_command import (
    ValidationCommand,
    ValidationCommandInput,
)
from dlkit.interfaces.api.domain.errors import WorkflowError
from dlkit.tools.config import GeneralSettings

# Hypothesis strategies


@st.composite
def validation_command_inputs(draw) -> ValidationCommandInput:
    """Generate ValidationCommandInput instances with various parameter combinations."""
    dry_build = draw(st.booleans())

    return ValidationCommandInput(dry_build=dry_build)


@st.composite
def mock_settings_with_sections(draw) -> Mock:
    """Generate mock settings with various section presence combinations."""
    settings = Mock(spec=GeneralSettings)

    # Required sections - sometimes None
    settings.MODEL = draw(st.one_of(st.none(), st.just(Mock())))
    settings.DATASET = draw(st.one_of(st.none(), st.just(Mock())))
    settings.DATAMODULE = draw(st.one_of(st.none(), st.just(Mock())))

    # Training section
    settings.TRAINING = draw(st.one_of(st.none(), st.just(Mock())))

    # Session configuration
    has_session = draw(st.booleans())
    if has_session:
        settings.SESSION = Mock()
        settings.SESSION.inference = draw(st.booleans())
    else:
        settings.SESSION = None

    # Model checkpoint for inference
    if settings.MODEL is not None:
        has_checkpoint = draw(st.booleans())
        settings.MODEL.checkpoint = "/path/to/checkpoint.ckpt" if has_checkpoint else None

    mlflow_active = draw(st.booleans())
    settings.MLFLOW = Mock() if mlflow_active else None
    settings.OPTUNA = Mock()
    settings.OPTUNA.enabled = draw(st.booleans())

    return settings


# Property-based tests


@given(validation_command_inputs())
def test_validation_command_input_dataclass_invariants(input_data: ValidationCommandInput) -> None:
    """Test ValidationCommandInput dataclass invariants."""
    # Input should always be constructible
    assert isinstance(input_data, ValidationCommandInput)

    # dry_build should always be boolean
    assert isinstance(input_data.dry_build, bool)

    # Dataclass should be frozen (immutable)
    with pytest.raises(AttributeError):
        input_data.dry_build = False  # type: ignore


@given(st.text(min_size=1, max_size=50))
def test_command_initialization_with_various_names(command_name: str) -> None:
    """Test ValidationCommand initialization with various command names."""
    command = ValidationCommand(command_name)
    assert command.command_name == command_name
    assert isinstance(command, ValidationCommand)


@given(validation_command_inputs(), mock_settings_with_sections())
def test_structural_validation_properties(
    input_data: ValidationCommandInput, settings: Mock
) -> None:
    """Test structural validation properties across different configurations."""
    command = ValidationCommand()

    # Determine expected validation outcome based on settings structure
    should_pass = _should_structural_validation_pass(settings)

    if should_pass:
        with patch("builtins.__import__"):
            with patch("dlkit.runtime.workflows.factories.build_factory.BuildFactory"):
                # Should not raise any exception for valid configurations
                try:
                    result = command.execute(input_data, settings)
                    assert result is True
                except WorkflowError as e:
                    # Only acceptable failures are import errors or dry build failures
                    assert (
                        "not available" in e.message
                        or "Dry build failed" in e.message
                        or "Import error" in e.message
                    )
    else:
        # Should raise WorkflowError for invalid configurations
        with pytest.raises(WorkflowError) as exc_info:
            command.execute(input_data, settings)

        error = exc_info.value
        assert "Configuration validation failed" in error.message
        assert error.context["command"] == "validate_config"


@given(st.booleans(), st.booleans())
def test_plugin_activation_profiles(mlflow_active: bool, optuna_active: bool) -> None:
    """Test validation succeeds across plugin activation combinations."""
    command = ValidationCommand()
    input_data = ValidationCommandInput(dry_build=False)

    # Create settings with plugin states
    settings = Mock(spec=GeneralSettings)
    settings.MODEL = Mock()
    settings.DATASET = Mock()
    settings.DATAMODULE = Mock()
    settings.TRAINING = Mock()
    settings.SESSION = Mock()
    settings.SESSION.inference = False
    settings.MLFLOW = Mock() if mlflow_active else None
    settings.OPTUNA = Mock()
    settings.OPTUNA.enabled = optuna_active

    with patch("builtins.__import__"):
        result = command.execute(input_data, settings)
        assert result is True


@given(st.booleans())
def test_inference_mode_detection_robustness(has_session: bool) -> None:
    """Test inference mode detection with various SESSION configurations."""
    command = ValidationCommand()
    input_data = ValidationCommandInput(dry_build=False)

    settings = Mock(spec=GeneralSettings)
    settings.MODEL = Mock()
    settings.MODEL.checkpoint = "/path/to/checkpoint.ckpt"
    settings.DATASET = Mock()
    settings.DATAMODULE = Mock()
    settings.TRAINING = None if has_session else Mock()

    if has_session:
        settings.SESSION = Mock()
        settings.SESSION.inference = True
    else:
        settings.SESSION = None

    settings.MLFLOW = None
    settings.OPTUNA = Mock()
    settings.OPTUNA.enabled = False

    result = command.execute(input_data, settings)
    assert result is True


# Helper functions


def _should_structural_validation_pass(settings: Mock) -> bool:
    """Determine if structural validation should pass based on settings.

    Args:
        settings: Mock settings object

    Returns:
        bool: True if validation should pass, False otherwise
    """
    # Check required sections
    if not settings.MODEL or not settings.DATASET or not settings.DATAMODULE:
        return False

    # Check mode-specific requirements
    is_inference = (
        settings.SESSION
        and hasattr(settings.SESSION, "inference")
        and getattr(settings.SESSION, "inference", False)
    )

    if is_inference:
        # Inference mode requires MODEL.checkpoint
        return bool(settings.MODEL.checkpoint)
    # Training mode requires TRAINING section (unless SESSION is None or doesn't exist)
    if not (settings.SESSION and getattr(settings.SESSION, "inference", False)):
        return bool(settings.TRAINING)
    return True


@given(
    st.text(min_size=0, max_size=20),
    st.dictionaries(st.text(min_size=1, max_size=10), st.text(min_size=1, max_size=20)),
)
def test_workflow_error_context_preservation(
    error_message: str, error_context: dict[str, str]
) -> None:
    """Test that WorkflowError context is properly preserved."""
    command = ValidationCommand()
    input_data = ValidationCommandInput(dry_build=False)

    # Create settings that will fail validation (missing MODEL)
    settings = Mock(spec=GeneralSettings)
    settings.MODEL = None
    settings.DATASET = Mock()
    settings.DATAMODULE = Mock()
    settings.TRAINING = Mock()
    settings.SESSION = Mock()
    settings.SESSION.inference = False
    settings.MLFLOW = None
    settings.OPTUNA = Mock()
    settings.OPTUNA.enabled = False

    with pytest.raises(WorkflowError) as exc_info:
        command.execute(input_data, settings)

    error = exc_info.value
    assert isinstance(error.context, dict)
    assert "command" in error.context
    assert "profile" in error.context
    assert error.context["command"] == "validate_config"
