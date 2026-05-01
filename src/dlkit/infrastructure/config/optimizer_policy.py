"""Top-level optimization program configuration."""

from __future__ import annotations

from pydantic import Field

from .core.base_settings import BasicSettings
from .optimization_stage import OptimizationStageSettings
from .optimizer_component import AdamWSettings, OptimizerSpec, SchedulerSpec


class OptimizerPolicySettings(BasicSettings):
    """Top-level configuration for an optimization program.

    Supports single-optimizer training (``default_optimizer``), concurrent
    optimizers via ``ConcurrentOptimizerSettings`` as the ``default_optimizer``,
    and sequential multi-stage programs via ``stages``.

    When ``stages`` is empty, uses ``default_optimizer`` and ``default_scheduler``
    as a fallback for simple single-optimizer (or concurrent) training.

    Attributes:
        stages: Ordered sequential stages. Empty = use default_optimizer.
        default_optimizer: Fallback optimizer when stages is empty. Defaults to AdamW.
            Can be a ``ConcurrentOptimizerSettings`` for concurrent training without stages.
        default_scheduler: Fallback scheduler when stages is empty.
    """

    stages: tuple[OptimizationStageSettings, ...] = Field(
        default=(), description="Ordered optimization stages"
    )
    default_optimizer: OptimizerSpec = Field(
        default_factory=AdamWSettings,
        description="Fallback optimizer when stages is empty",
    )
    default_scheduler: SchedulerSpec | None = Field(
        default=None, description="Fallback scheduler when stages is empty"
    )
