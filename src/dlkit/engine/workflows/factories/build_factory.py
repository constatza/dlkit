"""Build-factory dispatcher for runtime component construction."""

from __future__ import annotations

from dlkit.engine.adapters.lightning.factories import WrapperFactory
from dlkit.engine.adapters.lightning.model_detection import ModelType, detect_model_type
from dlkit.engine.training.components import RuntimeComponents
from dlkit.infrastructure.config.job_config import JobConfig
from dlkit.infrastructure.config.run_settings import apply_run_context
from dlkit.infrastructure.config.validators import validate_config_complete

from .build_strategy import (
    GraphBuildStrategy,
    IBuildStrategy,
    WorkflowSettings,
)
from .flexible_build_strategy import FlexibleBuildStrategy
from .generative_build_strategies import FlowMatchingBuildStrategy


class BuildFactory:
    """Select a build strategy and construct runtime components."""

    def __init__(self, strategies: list[IBuildStrategy] | None = None) -> None:
        self._strategies = strategies or [
            FlowMatchingBuildStrategy(),
            GraphBuildStrategy(),
            FlexibleBuildStrategy(),
        ]

    def _validate_settings(self, settings: WorkflowSettings) -> None:
        """Validate workflow completeness before building expensive components."""
        validate_config_complete(settings)

    def _build_with_context(
        self, strategy: IBuildStrategy, settings: WorkflowSettings
    ) -> RuntimeComponents:
        """Wrap strategy build in seed + precision context for the run.

        Args:
            strategy: The IBuildStrategy instance to use for building.
            settings: The workflow settings for component construction.

        Returns:
            Constructed RuntimeComponents with context applied.
        """
        with apply_run_context(settings.run, workers=True):
            return strategy.build(settings)

    def build_components(self, settings: WorkflowSettings) -> RuntimeComponents:
        """Build runtime components with the first matching strategy."""
        self._validate_settings(settings)
        for strategy in self._strategies:
            if strategy.can_handle(settings):
                return self._build_with_context(strategy, settings)
        raise ValueError(
            f"No build strategy matched settings of type {type(settings).__name__}. "
            "Ensure at least one strategy (e.g. FlexibleBuildStrategy) is registered."
        )


__all__ = [
    "BuildFactory",
    "JobConfig",
    "ModelType",
    "WrapperFactory",
    "WorkflowSettings",
    "detect_model_type",
]
