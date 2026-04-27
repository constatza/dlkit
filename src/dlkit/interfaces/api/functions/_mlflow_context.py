"""Shared MLflow helpers for API function modules."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol, runtime_checkable


@runtime_checkable
class ITrackingContext(Protocol):
    """Protocol for managing MLflow tracking URI scope."""

    def enter(self, uri: str) -> None:
        """Enter tracking context with the given URI.

        Args:
            uri: The tracking URI to set.
        """
        ...

    def exit(self) -> None:
        """Exit tracking context and restore previous URI."""
        ...

    def __enter__(self) -> ITrackingContext:
        """Support context manager entry."""
        ...

    def __exit__(self, *args: object) -> None:
        """Support context manager exit."""
        ...


class MLflowTrackingContext:
    """Sets and restores the MLflow tracking URI.

    This implementation manages the global MLflow state in a reentrant-safe
    manner by saving and restoring the tracking URI.
    """

    def __init__(self) -> None:
        """Initialize with no previous URI saved."""
        self._previous_uri: str | None = None

    def enter(self, uri: str) -> None:
        """Enter tracking context with the given URI.

        Args:
            uri: The tracking URI to set.
        """
        import mlflow

        self._previous_uri = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(uri)

    def exit(self) -> None:
        """Exit tracking context and restore previous URI."""
        import mlflow

        if self._previous_uri is not None:
            mlflow.set_tracking_uri(self._previous_uri)

    def __enter__(self) -> MLflowTrackingContext:
        """Support context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Support context manager exit."""
        self.exit()


class StubTrackingContext:
    """Stub implementation for testing that captures calls without side effects.

    Useful for testing code that uses ITrackingContext without requiring
    an actual MLflow backend.
    """

    def __init__(self) -> None:
        """Initialize with empty call history."""
        self.entered_uris: list[str] = []
        self.exit_count: int = 0

    def enter(self, uri: str) -> None:
        """Record entering with the given URI.

        Args:
            uri: The tracking URI to set.
        """
        self.entered_uris.append(uri)

    def exit(self) -> None:
        """Record exit call."""
        self.exit_count += 1

    def __enter__(self) -> StubTrackingContext:
        """Support context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Support context manager exit."""
        self.exit()


def create_mlflow_client(tracking_uri: str | None = None):
    """Create an MLflow client, optionally bound to a tracking URI."""
    from mlflow.tracking import MlflowClient

    match tracking_uri:
        case str() as configured_uri if configured_uri:
            return MlflowClient(tracking_uri=configured_uri)
        case _:
            return MlflowClient()


@contextmanager
def tracking_uri_context(
    tracking_uri: str | None, ctx: ITrackingContext | None = None
) -> Iterator[None]:
    """Temporarily override MLflow tracking URI while loading models.

    Args:
        tracking_uri: The tracking URI to set, or None to skip context.
        ctx: The tracking context to use. Defaults to MLflowTrackingContext().

    Yields:
        None
    """
    match tracking_uri:
        case str() as configured_uri if configured_uri:
            if ctx is None:
                ctx = MLflowTrackingContext()
            with ctx:
                ctx.enter(configured_uri)
                yield
        case _:
            yield
