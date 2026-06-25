"""Build-factory dispatcher for runtime component construction."""

from __future__ import annotations

from dlkit.engine.adapters.lightning.factories import WrapperFactory
from dlkit.engine.training.components import RuntimeComponents
from dlkit.infrastructure.config.job_config import JobConfig
from dlkit.infrastructure.config.validators import validate_config_complete
from dlkit.infrastructure.precision.context import precision_override
from dlkit.infrastructure.precision.strategy import PrecisionStrategy

from .build_strategy import (
    GraphBuildStrategy,
    IBuildStrategy,
    WorkflowSettings,
)
from .flexible_build_strategy import FlexibleBuildStrategy
from .generative_build_strategies import FlowMatchingBuildStrategy
from .model_detection import ModelType, detect_model_type


def _get_seed(settings: WorkflowSettings) -> int:
    """Extract random seed from JobConfig.

    Args:
        settings: A JobConfig instance.

    Returns:
        Seed value (default 42 when not configured).
    """
    return settings.run.seed or 42


def _get_precision_strategy(settings: WorkflowSettings) -> PrecisionStrategy | None:
    """Extract precision strategy from JobConfig.

    Args:
        settings: A JobConfig instance.

    Returns:
        PrecisionStrategy instance, or None if not configured.
    """
    return settings.run.precision


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
        """Wrap strategy build in precision context after applying the run seed.

        Args:
            strategy: The IBuildStrategy instance to use for building.
            settings: The workflow settings for component construction.

        Returns:
            Constructed RuntimeComponents with context applied.
        """
        from pytorch_lightning import seed_everything

        seed_everything(_get_seed(settings), workers=True)
        precision_strategy = _get_precision_strategy(settings)
        if precision_strategy is None:
            return strategy.build(settings)
        with precision_override(precision_strategy):
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
