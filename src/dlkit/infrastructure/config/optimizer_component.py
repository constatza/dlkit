"""Typed settings for individual optimizer and scheduler components."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from .core.base_settings import ComponentSettings
from .core.types import FloatHyperparameter, PositiveFloatHyperparameter


class OptimizerComponentSettings(ComponentSettings):
    """Settings for an individual optimizer component.

    This is the canonical single-optimizer config used within optimization stages.
    It supports both simple optimizer selection (by name) and advanced configurations
    (custom callables, dicts with hyperparameter specs).

    Attributes:
        name: Optimizer name, callable, or dict spec. Defaults to "AdamW".
        module_path: Import path for the optimizer. Defaults to "torch.optim".
        lr: Learning rate (positive float, or dict for hyperparameter search).
        weight_decay: L2 regularization weight. Defaults to 0.0.
    """

    model_config = SettingsConfigDict(extra="allow", arbitrary_types_allowed=True)

    name: str | Callable[..., Any] | dict[str, Any] | None = Field(
        default="AdamW", description="Optimizer name"
    )
    module_path: str | None = Field(
        default="torch.optim", description="Module path to the optimizer"
    )
    lr: PositiveFloatHyperparameter = Field(
        default=1e-3, description="Learning rate", alias="learning_rate"
    )
    weight_decay: FloatHyperparameter = Field(default=0.0, description="Optional weight decay")


class SchedulerComponentSettings(ComponentSettings):
    """Settings for a learning rate scheduler component.

    Schedules learning rate adjustments during training. Supports both common
    scheduler names and custom implementations.

    Attributes:
        name: Scheduler name, callable, or dict spec. Defaults to None (no scheduler).
        module_path: Import path for the scheduler. Defaults to "torch.optim.lr_scheduler".
        monitor: Metric to monitor for plateau-based scheduling. Defaults to "val_loss".
        frequency: Update frequency (epoch or step). Defaults to 1.
    """

    model_config = SettingsConfigDict(extra="allow")

    name: str | Callable[..., Any] | dict[str, Any] | None = Field(
        default=None, description="Scheduler name"
    )
    module_path: str | None = Field(
        default="torch.optim.lr_scheduler", description="Module path to the scheduler"
    )
    monitor: str = Field(
        default="val_loss", description="Metric to monitor for learning rate adjustment"
    )
    frequency: int = Field(default=1, description="Update frequency (epochs or steps)")
