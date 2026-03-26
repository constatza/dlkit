"""Precision context and protocol for DLKit.

Thread-safe precision context management for runtime overrides.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol

from dlkit.tools.config.precision.strategy import PrecisionStrategy


class PrecisionProvider(Protocol):
    """Protocol for objects that can provide precision strategy.

    This follows the Interface Segregation Principle by defining a minimal
    interface for precision-aware components.
    """

    def get_precision_strategy(self) -> PrecisionStrategy:
        """Get the precision strategy for this provider.

        Returns:
            PrecisionStrategy to use for this component.
        """
        ...


class PrecisionContext:
    """Thread-safe precision context manager for runtime overrides.

    This class provides a thread-local context for precision overrides that
    allows the API layer to temporarily override precision settings without
    affecting global configuration or other concurrent operations.

    Thread Safety:
        Uses threading.local() to ensure each thread has its own precision context,
        preventing interference between concurrent training/inference operations.

    Usage:
        Direct usage:
            context = PrecisionContext()
            context.set_override(PrecisionStrategy.MIXED_16)
            # Operations use MIXED_16 precision
            context.clear_override()

        Context manager usage:
            with context.precision_override(PrecisionStrategy.TRUE_16):
                # Operations use TRUE_16 precision
                pass
            # Precision reverts to previous value

        Global usage:
            with precision_override(PrecisionStrategy.FULL_64):
                # All operations use FULL_64 precision
                pass
    """

    def __init__(self) -> None:
        """Initialize thread-local precision context."""
        self._local = threading.local()

    def set_override(self, precision: PrecisionStrategy) -> None:
        """Set precision override for current thread.

        Args:
            precision: Precision strategy to use as override.
        """
        self._local.override = precision

    def get_override(self) -> PrecisionStrategy | None:
        """Get current precision override for this thread.

        Returns:
            Current precision override or None if no override is set.
        """
        return getattr(self._local, "override", None)

    def has_override(self) -> bool:
        """Check if precision override is set for current thread.

        Returns:
            True if precision override is active, False otherwise.
        """
        return self.get_override() is not None

    def clear_override(self) -> None:
        """Clear precision override for current thread."""
        if hasattr(self._local, "override"):
            del self._local.override

    @contextmanager
    def precision_override(self, precision: PrecisionStrategy) -> Iterator[None]:
        """Context manager for temporary precision override.

        Args:
            precision: Precision strategy to use within context.

        Yields:
            None - context manager for use with 'with' statement.

        Examples:
            >>> context = PrecisionContext()
            >>> with context.precision_override(PrecisionStrategy.MIXED_16):
            ...     # Operations here use MIXED_16 precision
            ...     pass
            # Precision reverts to previous value
        """
        previous_override = self.get_override()
        try:
            self.set_override(precision)
            yield
        finally:
            if previous_override is not None:
                self.set_override(previous_override)
            else:
                self.clear_override()

    @classmethod
    @contextmanager
    def override(cls, precision: PrecisionStrategy) -> Iterator[PrecisionContext]:
        """Class-level context manager for precision override.

        Args:
            precision: Precision strategy to use within context.

        Yields:
            PrecisionContext instance for use within the context.

        Examples:
            >>> with PrecisionContext.override(PrecisionStrategy.TRUE_16) as ctx:
            ...     # Operations here use TRUE_16 precision
            ...     assert ctx.get_override() == PrecisionStrategy.TRUE_16
        """
        context = cls()
        with context.precision_override(precision):
            yield context

    def resolve_precision(self, default: PrecisionStrategy) -> PrecisionStrategy:
        """Resolve effective precision considering override.

        Args:
            default: Default precision strategy to use if no override is set.

        Returns:
            Override precision if set, otherwise the default precision.
        """
        override = self.get_override()
        return override if override is not None else default

    def __repr__(self) -> str:
        """String representation for debugging."""
        override = self.get_override()
        if override:
            return f"PrecisionContext(override={override.name})"
        return "PrecisionContext(no_override)"


# Global precision context instance for use throughout DLKit
_global_precision_context = PrecisionContext()


def get_global_precision_context() -> PrecisionContext:
    """Get the global precision context instance.

    Returns:
        Global PrecisionContext instance for application-wide use.
    """
    return _global_precision_context


def current_precision_override() -> PrecisionStrategy | None:
    """Get current thread's precision override.

    Convenience function to check precision override without creating context instance.

    Returns:
        Current precision override or None if no override is active.
    """
    return _global_precision_context.get_override()


@contextmanager
def precision_override(precision: PrecisionStrategy) -> Iterator[None]:
    """Global precision override context manager.

    Convenience function for temporary precision overrides using the global context.

    Args:
        precision: Precision strategy to use within context.

    Yields:
        None - context manager for use with 'with' statement.

    Examples:
        >>> from dlkit.tools.config.precision.context import precision_override, PrecisionStrategy
        >>> with precision_override(PrecisionStrategy.MIXED_16):
        ...     # All operations here use MIXED_16 precision
        ...     pass
    """
    with _global_precision_context.precision_override(precision):
        yield


def get_precision_context() -> PrecisionContext:
    """Get the global precision context instance.

    Alias for get_global_precision_context() for convenience.

    Returns:
        Global PrecisionContext instance for application-wide use.
    """
    return get_global_precision_context()
