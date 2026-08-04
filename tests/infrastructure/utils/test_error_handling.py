"""Tests for `raise_error`'s explicit error-class threading and logging."""

import pytest
from loguru import logger

from dlkit.common.errors import ConfigurationError, StrategyError, WorkflowError
from dlkit.infrastructure.utils.error_handling import raise_error


def test_raise_error_defaults_to_workflow_error() -> None:
    """No `error_class` given raises `WorkflowError`, matching prior default behavior."""
    with pytest.raises(WorkflowError):
        raise_error("Training execution failed")


def test_raise_error_uses_explicit_error_class() -> None:
    """Callers can request any exception type directly, no message sniffing involved."""
    with pytest.raises(ConfigurationError):
        raise_error("Could not load settings", error_class=ConfigurationError)


def test_raise_error_message_mentioning_config_is_not_misclassified() -> None:
    """A workflow failure whose message happens to mention "config" for an unrelated
    reason must not be silently reclassified as `ConfigurationError` -- the caller's
    explicit `error_class` is the only source of truth now.
    """
    with pytest.raises(WorkflowError) as exc_info:
        raise_error(
            "Training failed: could not load config-adjacent asset",
            error_class=WorkflowError,
        )
    assert not isinstance(exc_info.value, ConfigurationError)


def test_raise_error_accepts_any_dlkit_error_subclass() -> None:
    """`error_class` isn't limited to Workflow/Configuration -- any DLKitError works."""
    with pytest.raises(StrategyError):
        raise_error("Strategy selection failed", error_class=StrategyError)


def test_raise_error_chains_original_error() -> None:
    """The original exception is preserved via `raise ... from original_error`."""
    original = ValueError("root cause")

    with pytest.raises(WorkflowError) as exc_info:
        raise_error("Training execution failed", original)

    assert exc_info.value.__cause__ is original
    assert "root cause" in str(exc_info.value)


def test_raise_error_context_includes_correlation_id_and_component() -> None:
    """Context still carries a correlation ID plus the caller's component/operation,
    even though the exception type is no longer derived from them.
    """
    with pytest.raises(WorkflowError) as exc_info:
        raise_error("Training execution failed")

    context = exc_info.value.context
    assert context["correlation_id"]
    assert context["operation"] == "test_raise_error_context_includes_correlation_id_and_component"


def test_raise_error_context_includes_stage_when_given() -> None:
    with pytest.raises(WorkflowError) as exc_info:
        raise_error("Training execution failed", stage="tracking")

    assert exc_info.value.context["stage"] == "tracking"


def test_raise_error_logs_the_failure() -> None:
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record["message"]), level="ERROR")

    try:
        with pytest.raises(WorkflowError):
            raise_error("Training execution failed")
    finally:
        logger.remove(sink_id)

    assert any("Training execution failed" in message for message in messages)
