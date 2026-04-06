"""Assembly value objects for dependency injection into Lightning wrappers."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch import nn

from .metrics_routing import MetricRoute
from .model_invoker import ModelOutputSpec


@dataclass(frozen=True, slots=True, kw_only=True)
class WrapperCheckpointMetadata:
    """Value object carrying serialisation-only metadata for checkpoint persistence.

    Keeps the wrapper __init__ clean by separating checkpoint metadata
    from the operational constructor arguments.

    Attributes:
        model_settings: Model configuration for checkpoint reconstruction.
        wrapper_settings: Wrapper configuration for checkpoint reconstruction.
        entry_configs: Data entry configurations in config order.
        feature_names: Ordered feature names for inference restore.
        predict_target_key: Name of target whose chain is inverted at predict time.
        shape_summary: Shape summary from dataset inference, or None.
        output_spec: Model output key spec for checkpoint-driven invoker rebuild.
    """

    model_settings: Any
    wrapper_settings: Any
    entry_configs: tuple[Any, ...]
    feature_names: tuple[str, ...]
    predict_target_key: str
    shape_summary: Any | None = None
    output_spec: ModelOutputSpec = dataclasses.field(default_factory=ModelOutputSpec)


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
        optimizer_factory: Callable accepting model parameters, returning Optimizer.
        scheduler_factory: Callable accepting optimizer, returning LRScheduler; or None.
        feature_transforms: Pre-built ModuleList per feature entry name (empty → no transforms).
        target_transforms: Pre-built ModuleList per target entry name (empty → no transforms).
    """

    loss_fn: nn.Module
    val_metric_routes: list[MetricRoute]
    test_metric_routes: list[MetricRoute]
    optimizer_factory: Callable[..., Any]
    scheduler_factory: Callable[..., Any] | None
    feature_transforms: dict[str, nn.ModuleList]
    target_transforms: dict[str, nn.ModuleList]
