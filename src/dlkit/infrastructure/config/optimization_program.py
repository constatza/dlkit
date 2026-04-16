"""Top-level optimization program configuration."""

from __future__ import annotations

from pydantic import Field

from .core.base_settings import BasicSettings
from .optimization_stage import ConcurrentOptimizationSettings, OptimizationStageSettings
from .optimizer_component import OptimizerComponentSettings, SchedulerComponentSettings


class OptimizationProgramSettings(BasicSettings):
    """Top-level configuration for an optimization program.

    Replaces the flat optimizer + scheduler approach from TrainingSettings.
    Supports multi-stage optimization with state transitions and concurrent optimizers.

    When stages is empty, uses default_optimizer and default_scheduler as a fallback
    for simple single-optimizer training.

    The intended usage pattern:
    - Simple training: Empty stages, use default_optimizer/default_scheduler
    - Multi-stage training: Populate stages, leave default_optimizer/scheduler unchanged

    Attributes:
        stages: Ordered stages (or concurrent groups). Empty = use default_optimizer.
        default_optimizer: Fallback optimizer when stages is empty.
        default_scheduler: Fallback scheduler when stages is empty.
    """

    stages: tuple[OptimizationStageSettings | ConcurrentOptimizationSettings, ...] = Field(
        default=(), description="Ordered optimization stages or concurrent groups"
    )
    default_optimizer: OptimizerComponentSettings = Field(
        default_factory=OptimizerComponentSettings,
        description="Fallback optimizer when stages is empty",
    )
    default_scheduler: SchedulerComponentSettings | None = Field(
        default=None, description="Fallback scheduler when stages is empty"
    )
