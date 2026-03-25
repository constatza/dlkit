"""Tests for error handling middleware with property-based testing."""

from __future__ import annotations

from unittest.mock import Mock

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from dlkit.interfaces.api.domain import (
    ConfigurationError,
    ModelStateError,
    PluginError,
    StrategyError,
    WorkflowError,
)
from dlkit.interfaces.cli.middleware.error_handler import (
    _get_error_suggestions,
    format_validation_error,
    handle_api_error,
    handle_keyboard_interrupt,
    handle_unexpected_error,
)


class TestHandleApiError:
    """Test API error handling with various error types."""

    def test_handle_configuration_error(
        self,
        mock_console: Mock,
    ) -> None:
        """Test handling configuration errors displays appropriate suggestions."""
        error = ConfigurationError(
            "Missing required field", {"field": "MODEL.name", "config_path": "/path/to/config.toml"}
        )

        handle_api_error(error, mock_console)

        # Verify console.print was called
        mock_console.print.assert_called_once()

        # Check that the error panel was created
        panel_arg = mock_console.print.call_args[0][0]
        assert hasattr(panel_arg, "title")

    def test_handle_strategy_error(
        self,
        mock_console: Mock,
    ) -> None:
        """Test handling strategy errors displays available strategies."""
        error = StrategyError(
            "Invalid strategy: unknown_strategy",
            {"strategy": "unknown_strategy", "available_modes": ["vanilla", "mlflow", "optuna"]},
        )

        handle_api_error(error, mock_console)

        mock_console.print.assert_called_once()

    def test_handle_plugin_error(
        self,
        mock_console: Mock,
    ) -> None:
        """Test handling plugin errors displays plugin-specific suggestions."""
        error = PluginError("MLflow plugin not configured", {"plugin": "mlflow"})

        handle_api_error(error, mock_console)

        mock_console.print.assert_called_once()

    def test_handle_model_state_error(
        self,
        mock_console: Mock,
    ) -> None:
        """Test handling model state errors."""
        error = ModelStateError("Model initialization failed", {"model_class": "TestModel"})

        handle_api_error(error, mock_console)

        mock_console.print.assert_called_once()

    def test_handle_workflow_error(
        self,
        mock_console: Mock,
    ) -> None:
        """Test handling workflow errors with strategy-specific suggestions."""
        error = WorkflowError(
            "Training failed: GPU out of memory", {"strategy": "mlflow", "gpu_memory": "8GB"}
        )

        handle_api_error(error, mock_console)

        mock_console.print.assert_called_once()


class TestGetErrorSuggestions:
    """Test error suggestion generation for different error types."""

    def test_configuration_error_suggestions(self) -> None:
        """Test suggestions for configuration errors."""
        error = ConfigurationError("Invalid configuration", {"config_path": "/path/to/config.toml"})

        suggestions = _get_error_suggestions(error)

        assert len(suggestions) > 0
        assert any("validate" in suggestion.lower() for suggestion in suggestions)
        assert any("template" in suggestion.lower() for suggestion in suggestions)
        assert any("/path/to/config.toml" in suggestion for suggestion in suggestions)

    def test_strategy_error_suggestions(self) -> None:
        """Test suggestions for strategy errors."""
        error = StrategyError(
            "Invalid strategy", {"available_modes": ["vanilla", "mlflow", "optuna"]}
        )

        suggestions = _get_error_suggestions(error)

        assert len(suggestions) > 0
        assert any("strategy" in suggestion.lower() for suggestion in suggestions)
        assert any("vanilla, mlflow, optuna" in suggestion for suggestion in suggestions)

    def test_plugin_error_suggestions(self) -> None:
        """Test suggestions for plugin errors."""
        error = PluginError("Plugin not enabled", {"plugin": "mlflow"})

        suggestions = _get_error_suggestions(error)

        assert len(suggestions) > 0
        assert any("mlflow" in suggestion.lower() for suggestion in suggestions)
        assert any("enable" in suggestion.lower() for suggestion in suggestions)

    def test_workflow_error_mlflow_suggestions(self) -> None:
        """Test MLflow-specific suggestions for workflow errors."""
        error = WorkflowError("MLflow connection failed", {"strategy": "mlflow"})

        suggestions = _get_error_suggestions(error)

        assert len(suggestions) > 0
        assert any("mlflow" in suggestion.lower() for suggestion in suggestions)

    def test_workflow_error_optuna_suggestions(self) -> None:
        """Test Optuna-specific suggestions for workflow errors."""
        error = WorkflowError("Study creation failed", {"strategy": "optuna"})

        suggestions = _get_error_suggestions(error)

        assert len(suggestions) > 0
        assert any("optuna" in suggestion.lower() for suggestion in suggestions)

    def test_model_state_error_suggestions(self) -> None:
        """Test suggestions for model state errors."""
        error = ModelStateError("Model failed to initialize", {})

        suggestions = _get_error_suggestions(error)

        assert len(suggestions) > 0
        assert any("model" in suggestion.lower() for suggestion in suggestions)
        assert any("configuration" in suggestion.lower() for suggestion in suggestions)

    def test_generic_workflow_error_suggestions(self) -> None:
        """Test generic suggestions for workflow errors without specific strategy."""
        error = WorkflowError("Training failed", {})

        suggestions = _get_error_suggestions(error)

        assert len(suggestions) > 0
        assert any("log" in suggestion.lower() for suggestion in suggestions)
        assert any("verbose" in suggestion.lower() for suggestion in suggestions)


class TestFormatValidationError:
    """Test validation error formatting."""

    def test_format_simple_error_message(self) -> None:
        """Test formatting simple error message."""
        error = ValueError("Simple error message")

        formatted = format_validation_error(error)

        assert formatted == "Simple error message"

    def test_format_validation_error_with_field_required(self) -> None:
        """Test formatting validation error with field required."""
        error_msg = "1 validation error\nfield required (type=value_error.missing)"
        error = ValueError(error_msg)

        formatted = format_validation_error(error)

        assert "Required field is missing" in formatted

    def test_format_validation_error_with_type_error(self) -> None:
        """Test formatting validation error with type error."""
        error_msg = "1 validation error\nvalue is not a valid integer (type=type_error.integer)"
        error = ValueError(error_msg)

        formatted = format_validation_error(error)

        assert "Invalid dataflow type" in formatted

    def test_format_validation_error_with_value_error(self) -> None:
        """Test formatting validation error with value error."""
        error_msg = "1 validation error\nensure this value is greater than 0 (type=value_error.number.not_gt)"
        error = ValueError(error_msg)

        formatted = format_validation_error(error)

        assert "Invalid value" in formatted


class TestHandleKeyboardInterrupt:
    """Test keyboard interrupt handling."""

    def test_handle_keyboard_interrupt_displays_message(
        self,
        mock_console: Mock,
    ) -> None:
        """Test keyboard interrupt displays cancellation message."""
        handle_keyboard_interrupt(mock_console)

        mock_console.print.assert_called_once()

        # Check that appropriate panel was created
        panel_arg = mock_console.print.call_args[0][0]
        assert hasattr(panel_arg, "title")


class TestHandleUnexpectedError:
    """Test unexpected error handling."""

    def test_handle_unexpected_error_displays_debug_info(
        self,
        mock_console: Mock,
    ) -> None:
        """Test unexpected error displays debugging information."""
        error = RuntimeError("Something went wrong")

        handle_unexpected_error(error, mock_console)

        mock_console.print.assert_called_once()

        # Check that error panel was created
        panel_arg = mock_console.print.call_args[0][0]
        assert hasattr(panel_arg, "title")

    def test_handle_unexpected_error_with_different_exception_types(
        self,
        mock_console: Mock,
    ) -> None:
        """Test unexpected error handling with different exception types."""
        errors = [
            ValueError("Invalid value"),
            TypeError("Type mismatch"),
            AttributeError("Missing attribute"),
            KeyError("Missing key"),
        ]

        for error in errors:
            mock_console.reset_mock()

            handle_unexpected_error(error, mock_console)

            mock_console.print.assert_called_once()


# Property-based tests using Hypothesis
class TestErrorHandlingProperties:
    """Property-based tests for error handling robustness."""

    @given(
        error_message=st.text(min_size=1, max_size=200),
        context_key=st.text(
            min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=["L", "N"])
        ),
        context_value=st.text(min_size=1, max_size=100),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_configuration_error_handling_is_robust(
        self,
        mock_console: Mock,
        error_message: str,
        context_key: str,
        context_value: str,
    ) -> None:
        """Test that configuration error handling works with arbitrary inputs."""
        mock_console.reset_mock()  # Reset mock for each hypothesis example
        error = ConfigurationError(error_message, {context_key: context_value})

        # Should not raise any exceptions
        handle_api_error(error, mock_console)

        # Should call console.print exactly once
        assert mock_console.print.call_count == 1

    @given(
        error_message=st.text(min_size=1, max_size=200),
        strategy=st.sampled_from(["vanilla", "mlflow", "optuna", "unknown", "custom"]),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_strategy_error_handling_is_robust(
        self,
        mock_console: Mock,
        error_message: str,
        strategy: str,
    ) -> None:
        """Test that strategy error handling works with various strategies."""
        mock_console.reset_mock()  # Reset mock for each hypothesis example
        error = StrategyError(
            error_message,
            {"strategy": strategy, "available_modes": ["vanilla", "mlflow", "optuna"]},
        )

        # Should not raise any exceptions
        handle_api_error(error, mock_console)

        # Should call console.print exactly once
        assert mock_console.print.call_count == 1

    @given(
        plugin_name=st.text(
            min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=["L", "N"])
        ),
        error_message=st.text(min_size=1, max_size=200),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_plugin_error_suggestions_always_include_plugin_name(
        self,
        plugin_name: str,
        error_message: str,
    ) -> None:
        """Test that plugin error suggestions always reference the plugin name."""
        error = PluginError(error_message, {"plugin": plugin_name})

        suggestions = _get_error_suggestions(error)

        # At least one suggestion should mention the plugin
        assert any(plugin_name in suggestion for suggestion in suggestions)

    @given(error_messages=st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=10))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_validation_error_formatting_handles_multiline_errors(
        self,
        error_messages: list[str],
    ) -> None:
        """Test validation error formatting with multiline error messages."""
        combined_message = "\n".join(error_messages)
        error = ValueError(combined_message)

        # Should not raise any exceptions
        formatted = format_validation_error(error)

        # Should return a string
        assert isinstance(formatted, str)
        assert len(formatted) > 0

    @given(
        context_dict=st.dictionaries(
            st.text(
                min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=["L", "N"])
            ),
            st.one_of(
                st.text(max_size=100),
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False),
                st.booleans(),
            ),
            min_size=0,
            max_size=5,
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_error_handling_with_various_context_types(
        self,
        mock_console: Mock,
        context_dict: dict,
    ) -> None:
        """Test error handling with various context value types."""
        mock_console.reset_mock()  # Reset mock for each hypothesis example
        error = WorkflowError("Test error", context_dict)

        # Should not raise any exceptions regardless of context types
        handle_api_error(error, mock_console)

        # Should call console.print exactly once
        assert mock_console.print.call_count == 1

    @given(
        exception_type=st.sampled_from(
            [
                ValueError,
                TypeError,
                AttributeError,
                KeyError,
                RuntimeError,
            ]
        ),
        message=st.text(min_size=1, max_size=200),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_unexpected_error_handling_with_various_exception_types(
        self,
        mock_console: Mock,
        exception_type: type,
        message: str,
    ) -> None:
        """Test unexpected error handling with various built-in exception types."""
        mock_console.reset_mock()  # Reset mock for each hypothesis example
        error = exception_type(message)

        # Should not raise any exceptions
        handle_unexpected_error(error, mock_console)

        # Should call console.print exactly once
        assert mock_console.print.call_count == 1
