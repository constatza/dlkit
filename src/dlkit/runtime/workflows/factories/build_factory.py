"""Build-factory dispatcher for runtime component construction."""

from __future__ import annotations

from dlkit.runtime.adapters.lightning.factories import WrapperFactory
from dlkit.runtime.execution.components import RuntimeComponents
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.validators import validate_config_complete
from dlkit.tools.config.workflow_configs import (
    InferenceWorkflowConfig,
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)

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

    def build_components(self, settings: WorkflowSettings) -> RuntimeComponents:
        """Build runtime components with the first matching strategy."""
        self._validate_settings(settings)
        for strategy in self._strategies:
            if strategy.can_handle(settings):
                return strategy.build(settings)
        return FlexibleBuildStrategy().build(settings)


__all__ = [
    "BuildFactory",
    "GeneralSettings",
    "ModelType",
    "WrapperFactory",
    "WorkflowSettings",
    "detect_model_type",
    "requires_shape_spec",
]
