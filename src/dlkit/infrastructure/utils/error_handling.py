"""Simplified unified error handling for DLKit.

Single function to handle all error raising with automatic logging and context.
"""

from __future__ import annotations

import inspect
import uuid
from typing import NamedTuple, NoReturn

from dlkit.common.errors import DLKitError, WorkflowError
from dlkit.infrastructure.utils.logging_config import get_logger

# Global logger for error handling
_error_logger = get_logger(__name__, "error_handler")


class _CallerContext(NamedTuple):
    """Component/operation identifying the frame that called `raise_error`."""

    component: str
    operation: str


def _resolve_caller_context() -> _CallerContext:
    """Detect the calling component and operation from the call stack.

    Walks two frames up from this helper (past `_resolve_caller_context`
    itself and `raise_error`) to find the frame that invoked `raise_error`.
    """
    frame = inspect.currentframe()
    component = "unknown"
    operation = "unknown"

    try:
        caller_frame = frame.f_back.f_back if frame and frame.f_back else None
        if caller_frame:
            filename = caller_frame.f_code.co_filename
            operation = caller_frame.f_code.co_name

            # Extract component from filename path, e.g. ".../dlkit/workflows/x.py" -> "workflows"
            parts = filename.split("/")
            if "dlkit" in parts:
                dlkit_idx = parts.index("dlkit")
                if dlkit_idx + 1 < len(parts):
                    component = parts[dlkit_idx + 1]
    finally:
        del frame  # Prevent reference cycles

    return _CallerContext(component=component, operation=operation)


def raise_error(
    message: str,
    original_error: Exception | None = None,
    *,
    error_class: type[Exception] = WorkflowError,
    stage: str | None = None,
) -> NoReturn:
    """Unified error raising with automatic context and logging.

    This is the single way to raise errors in DLKit. It automatically:
    - Detects the calling component from stack trace
    - Generates correlation ID for tracking
    - Logs the error with structured context
    - Raises the exception type the caller specifies
    - Chains original exception for debugging

    Callers must pass `error_class` explicitly when the default
    (`WorkflowError`) isn't the right type — e.g. `ConfigurationError` for a
    config-related failure. This is deliberate: the exception type is a
    property of what failed, which only the caller knows, not something to
    guess from the error message's wording.

    A `DLKitError` passed as `original_error` is re-raised unchanged instead
    of being wrapped in `error_class`: it's already the specific, actionable
    error a caller further up the stack needs (e.g. a `TrackingError` from a
    failed MLflow connectivity check) — wrapping it would only discard that
    specificity. Only non-`DLKitError` exceptions get wrapped.

    Args:
        message: Error message describing what failed
        original_error: Original exception to chain (optional)
        error_class: Exception type to raise (defaults to `WorkflowError`)
        stage: Optional workflow stage to record in the error context

    Raises:
        DLKitError: `original_error` unchanged, if it already was one.
        error_class: The exception type requested by the caller, otherwise.
    """
    if isinstance(original_error, DLKitError):
        raise original_error

    component, operation = _resolve_caller_context()

    # Generate correlation ID
    correlation_id = str(uuid.uuid4())[:8]

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

    # Raise the caller-specified exception
    if original_error:
        raise error_class(final_message, context) from original_error
    raise error_class(final_message, context)
