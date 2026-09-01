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
        on_connectivity_failure: How to react when the configured tracking
            backend is unreachable. ``"raise"`` (default) fails fast with a
            ``TrackingError`` — an unattended run can't ask a human, and
            silently degrading risks the run landing somewhere other than
            where the user expects. ``"fallback_local"`` explicitly opts
            into tracking locally instead of losing the run entirely; the
            resulting run is tagged to make the degraded mode visible.
    """

    backend: Literal["mlflow", "wandb", "none"] = "none"
    uri: str | None = None
    max_retries: int = 5
    model_serialization_format: Literal["pickle", "pt2"] = "pickle"
    on_connectivity_failure: Literal["raise", "fallback_local"] = "raise"
