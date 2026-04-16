"""Typed settings for optimization stage and concurrent group configurations."""

from __future__ import annotations

from pydantic import Field

from .core.base_settings import BasicSettings
from .optimization_selector import ParameterSelectorSettings
from .optimization_trigger import TriggerSettings
from .optimizer_component import OptimizerComponentSettings, SchedulerComponentSettings


class OptimizationStageSettings(BasicSettings):
    """Settings for a single optimization stage.

    A stage combines one optimizer, optional scheduler, parameter selector,
    and trigger condition for advancing to the next stage.

    Attributes:
        optimizer: Optimizer configuration.
        scheduler: Optional learning rate scheduler.
        selector: Optional parameter selector (None = all parameters).
        trigger: Trigger condition to advance to next stage (None = never transitions).
    """

    optimizer: OptimizerComponentSettings = Field(
        default_factory=OptimizerComponentSettings, description="Optimizer configuration"
    )
    scheduler: SchedulerComponentSettings | None = Field(
        default=None, description="Optional learning rate scheduler"
    )
    selector: ParameterSelectorSettings | None = Field(
        default=None, description="Optional parameter selector (None = all parameters)"
    )
    trigger: TriggerSettings = Field(
        default=None, description="Trigger condition to advance to next stage (None = never)"
    )


class ConcurrentOptimizationSettings(BasicSettings):
    """Settings for concurrent optimizers within a single stage.

    Multiple optimizers run simultaneously on disjoint parameter sets.

    Attributes:
        optimizers: Tuple of optimizer stages running concurrently.
        trigger: Trigger condition to advance beyond this concurrent group.
    """

    optimizers: tuple[OptimizationStageSettings, ...] = Field(
        default=(), description="Optimizers running concurrently"
    )
    trigger: TriggerSettings = Field(
        default=None, description="Trigger to advance beyond this concurrent group"
    )
