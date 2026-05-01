"""Typed settings for optimization stage configurations."""

from __future__ import annotations

from pydantic import Field

from .core.base_settings import BasicSettings
from .optimization_selector import ParameterSelectorSettings
from .optimization_trigger import TriggerSettings
from .optimizer_component import AdamWSettings, OptimizerSpec, SchedulerSpec


class OptimizationStageSettings(BasicSettings):
    """Settings for a single optimization stage.

    A stage combines one optimizer (which may be a ``ConcurrentOptimizerSettings``
    for parallel parameter groups), an optional scheduler, an optional parameter
    selector, and a trigger condition for advancing to the next stage.

    Attributes:
        optimizer: Optimizer configuration. Accepts any ``OptimizerSpec`` variant
            including ``ConcurrentOptimizerSettings`` for concurrent sub-optimizers.
        scheduler: Optional learning rate scheduler.
        selector: Optional parameter selector (None = all parameters).
        trigger: Trigger condition to advance to the next stage (None = never transitions).
    """

    optimizer: OptimizerSpec = Field(
        default_factory=AdamWSettings, description="Optimizer configuration"
    )
    scheduler: SchedulerSpec | None = Field(
        default=None, description="Optional learning rate scheduler"
    )
    selector: ParameterSelectorSettings | None = Field(
        default=None, description="Optional parameter selector (None = all parameters)"
    )
    trigger: TriggerSettings | None = Field(
        default=None, description="Trigger to advance to the next stage (None = never)"
    )
