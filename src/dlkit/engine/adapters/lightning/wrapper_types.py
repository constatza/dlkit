"""Assembly value objects for dependency injection into Lightning wrappers."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import nn

from dlkit.domain.nn.contracts import ModelContractSpec

from .metrics_routing import MetricRoute
from .model_invoker import ModelOutputSpec

if TYPE_CHECKING:
    from dlkit.common.geometry import GeometrySpec
    from dlkit.infrastructure.config import ModelComponentSettings, WrapperComponentSettings
    from dlkit.infrastructure.config.data_entries import DataEntry
    from dlkit.infrastructure.config.optimizer_policy import OptimizerPolicySettings


def build_checkpoint_metadata(
    *,
    model_settings: ModelComponentSettings,
    wrapper_settings: WrapperComponentSettings,
    entry_configs: tuple[DataEntry, ...],
    predict_target_key: str,
    geometry: GeometrySpec | None,
    output_spec: ModelOutputSpec,
    contract: ModelContractSpec | None = None,
) -> WrapperCheckpointMetadata:
    """Build checkpoint metadata from wrapper configuration.

    Constructs the metadata value object that is serialized to checkpoint and used
    for inference-time checkpoint restoration.

    Args:
        model_settings: Model configuration.
        wrapper_settings: Wrapper configuration.
        entry_configs: Data entry configurations in config-insertion order.
        predict_target_key: Name of target whose inverse transform applies at predict time.
        geometry: GeometrySpec from dataset inference, or None.
        output_spec: Model output key specification.
        contract: Optional ModelContractSpec used to build the model, persisted for
            inference-time reconstruction.

    Returns:
        Configured WrapperCheckpointMetadata ready for checkpoint persistence.
    """
    return WrapperCheckpointMetadata(
        model_settings=model_settings,
        wrapper_settings=wrapper_settings,
        entry_configs=entry_configs,
        predict_target_key=predict_target_key,
        geometry=geometry,
        output_spec=output_spec,
        contract=contract,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class WrapperCheckpointMetadata:
    """Value object carrying serialisation-only metadata for checkpoint persistence.

    Keeps the wrapper __init__ clean by separating checkpoint metadata
    from the operational constructor arguments.

    Attributes:
        model_settings: Model configuration for checkpoint reconstruction.
        wrapper_settings: Wrapper configuration for checkpoint reconstruction.
        entry_configs: Data entry configurations in config order.
        predict_target_key: Name of target whose chain is inverted at predict time.
        geometry: GeometrySpec from dataset inference, or None.
        output_spec: Model output key spec for checkpoint-driven invoker rebuild.
        contract: ModelContractSpec used to build the model, or None when not
            applicable. Persisted in checkpoints for inference-time reconstruction.
    """

    model_settings: ModelComponentSettings
    wrapper_settings: WrapperComponentSettings
    entry_configs: tuple[DataEntry, ...]
    predict_target_key: str
    geometry: GeometrySpec | None = None
    output_spec: ModelOutputSpec = dataclasses.field(default_factory=ModelOutputSpec)
    contract: ModelContractSpec | None = None

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Derive ordered feature names from geometry fields or entry_configs.

        Returns:
            Tuple of feature name strings in insertion order.
        """
        if self.geometry is not None:
            return tuple(f.name for f in self.geometry.fields)
        from dlkit.infrastructure.config.data_entries import is_feature

        return tuple(
            e.name
            for e in self.entry_configs
            if is_feature(e) and e.model_input and e.name is not None
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class WrapperComponents:
    """Pre-built wrapper components for dependency injection into a Lightning wrapper.

    Constructed at the engine/factories boundary; consumed by core wrappers.
    All fields are pre-instantiated nn.Modules or callables — no FactoryProvider
    calls needed inside core after this object is created.

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
