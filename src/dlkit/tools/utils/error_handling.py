"""Simplified unified error handling for DLKit.

Single function to handle all error raising with automatic logging and context.
"""

from __future__ import annotations

import inspect
import uuid

from dlkit.interfaces.api.domain.errors import WorkflowError, ConfigurationError
from dlkit.tools.utils.logging_config import get_logger

# Global logger for error handling
_error_logger = get_logger(__name__, "error_handler")


def raise_error(
    message: str, original_error: Exception | None = None, *, stage: str | None = None
) -> None:
    """Unified error raising with automatic context and logging.

    This is the single way to raise errors in DLKit. It automatically:
    - Detects the calling component from stack trace
    - Generates correlation ID for tracking
    - Logs the error with structured context
    - Raises appropriate exception type
    - Chains original exception for debugging

    Args:
        message: Error message describing what failed
        original_error: Original exception to chain (optional)

    Raises:
        WorkflowError: For workflow/execution errors
        ConfigurationError: For config-related errors
        DLKitError: For other domain errors
    """
    # Auto-detect component from call stack
    frame = inspect.currentframe()
    component = "unknown"
    operation = "unknown"

    try:
        if frame and frame.f_back:
            caller_frame = frame.f_back
            filename = caller_frame.f_code.co_filename
            function_name = caller_frame.f_code.co_name

            # Extract component from filename path
            if "dlkit" in filename:
                parts = filename.split("/")
                if "dlkit" in parts:
                    dlkit_idx = parts.index("dlkit")
                    if dlkit_idx + 1 < len(parts):
                        component = parts[dlkit_idx + 1]  # e.g., "workflows", "api", "adapters"

            operation = function_name
    finally:
        del frame  # Prevent reference cycles

    # Generate correlation ID
    correlation_id = str(uuid.uuid4())[:8]

    # Determine exception type based on component or message content
    error_class = WorkflowError  # Default
    if component == "io" or "config" in message.lower():
        error_class = ConfigurationError

    # Create context
    context = {"correlation_id": correlation_id, "component": component, "operation": operation}
    if stage:
        context["stage"] = stage

    # Add original error info if present
    if original_error:
        context["original_error"] = str(original_error)
        context["original_error_type"] = type(original_error).__name__

    # Compose final message to include original error detail when present
    final_message = message
    if original_error:
        try:
            final_message = f"{message}: {original_error}"
        except Exception:
            final_message = message

    # Keep the log line flat. The exception chain already preserves detail.
    if original_error:
        _error_logger.error(
            "Error in {}.{}: {}: {}",
            component,
            operation,
            message,
            original_error,
        )
    else:
        _error_logger.error("Error in {}.{}: {}", component, operation, message)

    # Raise the appropriate exception
    if original_error:
        raise error_class(final_message, context) from original_error
    else:
        raise error_class(final_message, context)
