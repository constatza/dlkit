"""Typed settings for stage transition triggers."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .core.base_settings import BasicSettings


class TriggerSettings(BasicSettings):
    """Stage transition trigger — set at_epoch for epoch-based or patience for plateau-based.

    Exactly one of ``at_epoch`` or ``patience`` must be specified.

    Attributes:
        at_epoch: 0-indexed epoch at which to trigger transition. Use for epoch-based triggers.
        patience: Epochs without improvement before transition. Use for plateau-based triggers.
        monitor: Metric to monitor (plateau only). Defaults to ``"val_loss"``.
        min_delta: Minimum change to qualify as improvement (plateau only). Defaults to 1e-4.
        mode: Optimization direction (plateau only). Defaults to ``"min"``.
    """

    at_epoch: int | None = Field(
        default=None, description="0-indexed epoch at which transition fires (epoch trigger)"
    )
    patience: int | None = Field(
        default=None,
        description="Epochs without improvement before transitioning (plateau trigger)",
    )
    monitor: str = Field(default="val_loss", description="Metric to monitor (plateau only)")
    min_delta: float = Field(
        default=1e-4, description="Minimum change to qualify as improvement (plateau only)"
    )
    mode: Literal["min", "max"] = Field(
        default="min", description="Optimization direction (plateau only)"
    )

    @model_validator(mode="after")
    def _validate_trigger_type(self) -> TriggerSettings:
        """Enforce that exactly one trigger type is specified.

        Returns:
            Self after validation.

        Raises:
            ValueError: If both or neither of at_epoch and patience are set.
        """
        both_set = self.at_epoch is not None and self.patience is not None
        neither_set = self.at_epoch is None and self.patience is None
        if both_set or neither_set:
            raise ValueError(
                "Specify exactly one of at_epoch (epoch trigger) or patience (plateau trigger)."
            )
        return self
