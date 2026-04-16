"""Typed settings for stage transition triggers."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .core.base_settings import BasicSettings


class EpochTriggerSettings(BasicSettings):
    """Settings for epoch-based stage transition.

    Transition occurs when the specified epoch is reached.

    Attributes:
        at_epoch: Epoch number (1-indexed) at which to trigger transition.
    """

    at_epoch: int = Field(..., description="Epoch number at which transition fires (1-indexed)")


class PlateauTriggerSettings(BasicSettings):
    """Settings for plateau-based stage transition.

    Transition occurs when monitored metric shows no improvement for patience epochs.

    Attributes:
        monitor: Metric to monitor. Defaults to "val_loss".
        patience: Epochs without improvement before transition. Defaults to 10.
        min_delta: Minimum change to qualify as improvement. Defaults to 1e-4.
        mode: Optimization direction ("min" or "max"). Defaults to "min".
    """

    monitor: str = Field(default="val_loss", description="Metric to monitor")
    patience: int = Field(
        default=10, description="Number of epochs with no improvement before transitioning"
    )
    min_delta: float = Field(default=1e-4, description="Minimum change to qualify as improvement")
    mode: Literal["min", "max"] = Field(
        default="min", description="Optimization direction (min or max)"
    )


type TriggerSettings = EpochTriggerSettings | PlateauTriggerSettings | None
"""Type alias for any valid trigger configuration, or None for no trigger."""
