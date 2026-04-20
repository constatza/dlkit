"""Typed settings for optimization stage and concurrent group configurations."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from .core.base_settings import BasicSettings
from .optimization_selector import ParameterSelectorSettings
from .optimization_trigger import TriggerSpec
from .optimizer_component import AdamWSettings, OptimizerSpec, SchedulerSpec


class OptimizationStageSettings(BasicSettings):
    """Settings for a single optimization stage.

    A stage combines one optimizer, optional scheduler, parameter selector,
    and trigger condition for advancing to the next stage.

    Attributes:
        kind: Discriminator tag — always ``"stage"``.
        optimizer: Optimizer configuration. Accepts any ``OptimizerSpec`` variant
            (``AdamWSettings``, ``AdamSettings``, ``LBFGSSettings``, ``MuonSettings``).
            Pydantic dispatches deserialization via the ``name`` discriminator field.
        scheduler: Optional learning rate scheduler.
        selector: Optional parameter selector (None = all parameters).
        trigger: Trigger condition to advance to next stage (None = never transitions).
    """

    kind: Literal["stage"] = "stage"
    optimizer: OptimizerSpec = Field(
        default_factory=AdamWSettings, description="Optimizer configuration"
    )
    scheduler: SchedulerSpec | None = Field(
        default=None, description="Optional learning rate scheduler"
    )
    selector: ParameterSelectorSettings | None = Field(
        default=None, description="Optional parameter selector (None = all parameters)"
    )
    trigger: TriggerSpec | None = Field(
        default=None, description="Trigger condition to advance to next stage (None = never)"
    )


class ConcurrentOptimizationSettings(BasicSettings):
    """Settings for concurrent optimizers within a single stage.

    Multiple optimizers run simultaneously on disjoint parameter sets.

    Attributes:
        kind: Discriminator tag — always ``"concurrent"``.
        optimizers: Tuple of optimizer stages running concurrently.
        trigger: Trigger condition to advance beyond this concurrent group.
    """

    kind: Literal["concurrent"] = "concurrent"
    optimizers: tuple[OptimizationStageSettings, ...] = Field(
        default=(), description="Optimizers running concurrently"
    )
    trigger: TriggerSpec | None = Field(
        default=None, description="Trigger to advance beyond this concurrent group"
    )


StageSpec = Annotated[
    OptimizationStageSettings | ConcurrentOptimizationSettings,
    Field(discriminator="kind"),
]
"""Discriminated union of all stage variants.

Pydantic dispatches deserialization to the correct subclass via the ``kind``
discriminator field. Use ``tuple[StageSpec, ...]`` as the type for an ordered
sequence of stages or concurrent groups.
"""
