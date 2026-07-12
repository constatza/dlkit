"""Tracking settings — backend connection configuration."""

from __future__ import annotations

from typing import Literal

from dlkit.infrastructure.config.core.base_settings import BasicSettings


class TrackingSettings(BasicSettings):
    """Tracking backend connection.

    Typically provided via a user-level profile (~/.config/dlkit/mlflow.toml).

    Args:
        backend: Tracking backend type.
        uri: Backend connection URI.
        max_retries: Maximum connection retries for transient errors.
        model_serialization_format: PyTorch model artifact serialization format.
    """

    backend: Literal["mlflow", "wandb", "none"] = "none"
    uri: str | None = None
    max_retries: int = 5
    model_serialization_format: Literal["pickle", "pt2"] = "pickle"
