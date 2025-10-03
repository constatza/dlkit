"""CLI middleware for error handling and other cross-cutting concerns."""

from .error_handler import (
    format_validation_error,
    handle_api_error,
    handle_keyboard_interrupt,
    handle_unexpected_error,
)

__all__ = [
    "handle_api_error",
    "handle_keyboard_interrupt",
    "handle_unexpected_error",
    "format_validation_error",
]
