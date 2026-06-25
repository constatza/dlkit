"""Assembly value objects for dependency injection into Lightning wrappers."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from torch import nn

from .metrics_routing import MetricRoute
from .model_invoker import ModelOutputSpec

if TYPE_CHECKING:
    from dlkit.common.shapes import ShapeContext
    from dlkit.infrastructure.config import ModelComponentSettings, WrapperComponentSettings
    from dlkit.infrastructure.config.data_entries import DataEntry
    from dlkit.infrastructure.config.model_settings import ModelSettings
    from dlkit.infrastructure.config.optimizer_policy import OptimizerPolicySettings


def build_checkpoint_metadata(
    *,
    model_settings: ModelComponentSettings | ModelSettings,
    wrapper_settings: WrapperComponentSettings,
    entry_configs: tuple[DataEntry, ...],
    predict_target_key: str,
    output_spec: ModelOutputSpec,
    context: ShapeContext | None = None,
) -> WrapperCheckpointMetadata:
    """Build checkpoint metadata from wrapper configuration.

    Args:
        model_settings: Model configuration.
        wrapper_settings: Wrapper configuration.
        entry_configs: Data entry configurations in config-insertion order.
        predict_target_key: Name of target whose inverse transform applies at predict time.
        output_spec: Model output key specification.
        context: Optional ``ShapeContext`` used to build the model, persisted
            for inference-time reconstruction.

    Returns:
        Configured WrapperCheckpointMetadata ready for checkpoint persistence.
    """
    return WrapperCheckpointMetadata(
        model_settings=model_settings,
        wrapper_settings=wrapper_settings,
        entry_configs=entry_configs,
        predict_target_key=predict_target_key,
        output_spec=output_spec,
        context=context,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class WrapperCheckpointMetadata:
    """Value object carrying serialisation-only metadata for checkpoint persistence.

    Attributes:
        model_settings: Model configuration for checkpoint reconstruction.
        wrapper_settings: Wrapper configuration for checkpoint reconstruction.
        entry_configs: Data entry configurations in config order.
        predict_target_key: Name of target whose chain is inverted at predict time.
        output_spec: Model output key spec for checkpoint-driven invoker rebuild.
        context: Shape context used to build the model, or None when not applicable.
    """

    model_settings: ModelComponentSettings | ModelSettings
    wrapper_settings: WrapperComponentSettings
    entry_configs: tuple[DataEntry, ...]
    predict_target_key: str
    output_spec: ModelOutputSpec = dataclasses.field(default_factory=ModelOutputSpec)
    context: ShapeContext | None = None

    @property
    def input_shapes(self) -> Any:
        """Input shapes from context, or None."""
        return self.context.input_shapes if self.context is not None else None

    @property
    def output_shapes(self) -> Any:
        """Output shapes from context, or None."""
        return self.context.output_shapes if self.context is not None else None

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Derive ordered feature names from entry_configs.

        Returns:
            Tuple of feature name strings in insertion order.
        """
        from dlkit.infrastructure.config.data_entries import is_feature

        return tuple(
            e.name
            for e in self.entry_configs
            if is_feature(e) and e.model_input and e.name is not None
        )

    @property
    def forward_arg_map(self) -> dict[str, str]:
        """Derive forward-arg map from entry configs for checkpoint persistence.

        Returns:
            Dict mapping kwarg name to feature name.
        """
        from dlkit.infrastructure.config.data_entries import is_feature

        return {
            e.name: e.name
            for e in self.entry_configs
            if is_feature(e) and e.model_input and e.name is not None
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class WrapperComponents:
    """Pre-built wrapper components for dependency injection into a Lightning wrapper.

    Attributes:
        loss_fn: Instantiated loss function module.
        val_metric_routes: MetricRoute list for validation stage.
        test_metric_routes: MetricRoute list for test stage.
        optimizer_policy_settings: Optimization program configuration for building controller.
        feature_transforms: Pre-built ModuleList per feature entry name (empty → no transforms).
        target_transforms: Pre-built ModuleList per target entry name (empty → no transforms).
    """

    loss_fn: nn.Module
    val_metric_routes: list[MetricRoute]
    test_metric_routes: list[MetricRoute]
    optimizer_policy_settings: OptimizerPolicySettings
    feature_transforms: dict[str, nn.ModuleList]
    target_transforms: dict[str, nn.ModuleList]
