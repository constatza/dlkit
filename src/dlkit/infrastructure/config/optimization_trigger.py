"""Typed settings for stage transition triggers."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from .core.base_settings import BasicSettings


class EpochTriggerSettings(BasicSettings):
    """Settings for epoch-based stage transition.

    Transition occurs when the specified epoch is reached.

    Attributes:
        kind: Discriminator tag — always ``"epoch"``.
        at_epoch: 0-indexed epoch number (Lightning epoch counter) at which to trigger transition.
    """

    kind: Literal["epoch"] = "epoch"
    at_epoch: int = Field(..., description="0-indexed epoch number at which transition fires")


class PlateauTriggerSettings(BasicSettings):
    """Settings for plateau-based stage transition.

    Transition occurs when monitored metric shows no improvement for patience epochs.

    Attributes:
        kind: Discriminator tag — always ``"plateau"``.
        monitor: Metric to monitor. Defaults to "val_loss".
        patience: Epochs without improvement before transition. Defaults to 10.
        min_delta: Minimum change to qualify as improvement. Defaults to 1e-4.
        mode: Optimization direction ("min" or "max"). Defaults to "min".
    """

    kind: Literal["plateau"] = "plateau"
    monitor: str = Field(default="val_loss", description="Metric to monitor")
    patience: int = Field(
        default=10, description="Number of epochs with no improvement before transitioning"
    )
    min_delta: float = Field(default=1e-4, description="Minimum change to qualify as improvement")
    mode: Literal["min", "max"] = Field(
        default="min", description="Optimization direction (min or max)"
    )


TriggerSpec = Annotated[
    EpochTriggerSettings | PlateauTriggerSettings,
    Field(discriminator="kind"),
]
"""Discriminated union of all trigger variants.

Use ``TriggerSpec | None`` as a field type where a trigger is optional.
Pydantic dispatches deserialization to the correct subclass via the
``kind`` discriminator field.
"""
