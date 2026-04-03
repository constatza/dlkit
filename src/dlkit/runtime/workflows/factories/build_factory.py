"""Build-factory dispatcher for runtime component construction."""

from __future__ import annotations

import contextlib

from dlkit.runtime.adapters.lightning.factories import WrapperFactory
from dlkit.runtime.execution.components import RuntimeComponents
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.validators import validate_config_complete
from dlkit.tools.config.workflow_configs import (
    InferenceWorkflowConfig,
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)
from dlkit.tools.io.path_context import get_current_path_context, path_override_context
from dlkit.tools.io.paths import coerce_root_dir_to_absolute
from dlkit.tools.precision.context import precision_override

from .build_strategy import (
    GraphBuildStrategy,
    IBuildStrategy,
    TimeSeriesBuildStrategy,
    WorkflowSettings,
)
from .flexible_build_strategy import FlexibleBuildStrategy
from .generative_build_strategies import FlowMatchingBuildStrategy
from .model_detection import ModelType, detect_model_type, requires_shape_spec


class BuildFactory:
    """Select a build strategy and construct runtime components."""

    def __init__(self, strategies: list[IBuildStrategy] | None = None) -> None:
        self._strategies = strategies or [
            FlowMatchingBuildStrategy(),
            GraphBuildStrategy(),
            TimeSeriesBuildStrategy(),
            FlexibleBuildStrategy(),
        ]

    def _validate_settings(self, settings: WorkflowSettings) -> None:
        """Validate workflow completeness before building expensive components."""
        if isinstance(
            settings, (TrainingWorkflowConfig, InferenceWorkflowConfig, OptimizationWorkflowConfig)
        ):
            validate_config_complete(settings)

    def _build_with_context(
        self, strategy: IBuildStrategy, settings: WorkflowSettings
    ) -> RuntimeComponents:
        """Wrap strategy build in precision and path context.

        Args:
            strategy: The IBuildStrategy instance to use for building.
            settings: The workflow settings for component construction.

        Returns:
            Constructed RuntimeComponents with context applied.
        """
        precision_strategy = settings.SESSION.get_precision_strategy()
        context = get_current_path_context()
        session_root_dir = coerce_root_dir_to_absolute(settings.SESSION.root_dir)
        needs_path_context = (not context or not context.root_dir) and session_root_dir

        precision_ctx = (
            precision_override(precision_strategy)
            if precision_strategy is not None
            else contextlib.nullcontext()
        )
        with precision_ctx:
            if needs_path_context:
                with path_override_context({"root_dir": session_root_dir}):
                    return strategy.build(settings)
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
    "GeneralSettings",
    "ModelType",
    "WrapperFactory",
    "WorkflowSettings",
    "detect_model_type",
    "requires_shape_spec",
]
