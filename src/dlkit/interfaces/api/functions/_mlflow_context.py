"""Shared MLflow helpers for API function modules."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


def create_mlflow_client(tracking_uri: str | None = None):
    """Create an MLflow client, optionally bound to a tracking URI."""
    from mlflow.tracking import MlflowClient

    match tracking_uri:
        case str() as configured_uri if configured_uri:
            return MlflowClient(tracking_uri=configured_uri)
        case _:
            return MlflowClient()


@contextmanager
def tracking_uri_context(tracking_uri: str | None) -> Iterator[None]:
    """Temporarily override MLflow tracking URI while loading models."""
    match tracking_uri:
        case str() as configured_uri if configured_uri:
            import mlflow

            previous_uri = mlflow.get_tracking_uri()
            mlflow.set_tracking_uri(configured_uri)
            try:
                yield
            finally:
                mlflow.set_tracking_uri(previous_uri)
        case _:
            yield
