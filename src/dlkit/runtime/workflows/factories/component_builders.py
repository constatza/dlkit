"""Pre-build wrapper components before injection into core wrappers.

All FactoryProvider calls for Lightning wrappers live here.
Core wrappers accept WrapperComponents and never call FactoryProvider directly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from torch import nn
from torch.nn import ModuleList

from dlkit.runtime.adapters.lightning.components import MetricRoute, WrapperComponents
from dlkit.tools.config.core.context import BuildContext
from dlkit.tools.config.core.factories import FactoryProvider
from dlkit.tools.config.data_entries import DataEntry, is_feature_entry, is_target_entry
from dlkit.tools.config.model_components import (
    LossComponentSettings,
    WrapperComponentSettings,
)

from .module_defaults import with_runtime_module_defaults


def build_loss_fn(loss_settings: LossComponentSettings) -> nn.Module:
    """Instantiate loss function from settings via FactoryProvider.

    Args:
        loss_settings: Loss configuration.

    Returns:
        Instantiated loss function module.
    """
    return FactoryProvider.create_component(
        with_runtime_module_defaults(loss_settings),
        BuildContext(mode="training"),
    )


def build_metric_routes(
    metric_specs: tuple,
    default_target_key: str,
) -> list[MetricRoute]:
    """Build MetricRoute list from metric settings.

    Replaces standard.py._make_routes logic.

    Args:
        metric_specs: Tuple of MetricComponentSettings.
        default_target_key: Target name used when spec.target_key is None.

    Returns:
        List of MetricRoute value objects.
    """
    routes = []
    for spec in metric_specs:
        metric = FactoryProvider.create_component(
            with_runtime_module_defaults(spec),
            BuildContext(mode="training"),
        )
        target_key_str = getattr(spec, "target_key", None)
        if target_key_str:
            target_name = target_key_str.split(".", 1)[1]
        else:
            target_name = default_target_key
        extra_inputs = getattr(spec, "extra_inputs", ()) or ()
        routes.append(
            MetricRoute(
                metric=metric,
                target_ns="targets",
                target_name=target_name,
                extra_inputs=tuple(extra_inputs),
            )
        )
    return routes


def build_transform_list(
    transform_seq: Any,
    shape_spec: Any = None,
    entry_name: str | None = None,
    validate_execution: bool = False,
) -> tuple[ModuleList, tuple[int, ...] | None]:
    """Instantiate transforms with analytical shape inference.

    Moved from chain.build_transforms. This is the only function allowed to
    call FactoryProvider for transforms.

    Args:
        transform_seq: Sequence of TransformSettings.
        shape_spec: Optional shape specification for pre-allocation.
        entry_name: Entry name for shape lookup.
        validate_execution: Whether to validate with dummy tensors.

    Returns:
        Tuple of (ModuleList, inferred_output_shape | None).
    """
    import torch

    current_shape = None
    if shape_spec and entry_name:
        current_shape = shape_spec.get_shape(entry_name)

    module_list = ModuleList()
    dummy_input = None
    if validate_execution and current_shape is not None:
        dummy_input = torch.zeros(current_shape)

    for transform_settings in transform_seq:
        context = BuildContext(mode="transform_chain")
        module = FactoryProvider.create_component(
            with_runtime_module_defaults(transform_settings),
            context,
        )

        if current_shape is not None and hasattr(module, "infer_output_shape"):
            current_shape = module.infer_output_shape(current_shape)

        if validate_execution and dummy_input is not None:
            dummy_input = module(dummy_input)
            if current_shape is not None:
                assert tuple(dummy_input.shape) == current_shape

        module_list.append(module)

    return module_list, current_shape


def make_optimizer_factory(optimizer_settings: Any) -> Callable[..., Any]:
    """Return a callable that builds an optimizer from model parameters.

    Args:
        optimizer_settings: Optimizer configuration.

    Returns:
        Callable accepting model parameters and returning an Optimizer instance.
    """

    def _factory(params: Iterable) -> Any:
        return FactoryProvider.create_component(
            optimizer_settings,
            BuildContext(mode="training", overrides={"params": params}),
        )

    return _factory


def make_scheduler_factory(scheduler_settings: Any) -> Callable[..., Any] | None:
    """Return a callable that builds a scheduler from an optimizer, or None.

    Args:
        scheduler_settings: Scheduler configuration, or None.

    Returns:
        Callable accepting an optimizer and returning a scheduler, or None if
        scheduler_settings is None.
    """
    if scheduler_settings is None:
        return None

    def _factory(optimizer: Any) -> Any:
        return FactoryProvider.create_component(
            scheduler_settings,
            BuildContext(mode="training", overrides={"optimizer": optimizer}),
        )

    return _factory


def build_wrapper_components(
    settings: WrapperComponentSettings,
    entry_configs: tuple[DataEntry, ...],
) -> WrapperComponents:
    """Build all pre-instantiated components for a standard Lightning wrapper.

    All FactoryProvider calls for wrappers are centralized here, allowing
    core wrappers to accept a pure value object (WrapperComponents) without
    knowing about factories or build contexts.

    Args:
        settings: Wrapper configuration (loss, metrics, optimizer, scheduler).
        entry_configs: Data entry configurations.

    Returns:
        WrapperComponents value object with all pre-built components.
    """
    feature_entries = [e for e in entry_configs if is_feature_entry(e)]
    target_entries = [e for e in entry_configs if is_target_entry(e)]
    all_target_keys = tuple(e.name for e in target_entries if e.name is not None)
    default_target_key = all_target_keys[0] if all_target_keys else ""

    # Build loss
    loss_fn = build_loss_fn(settings.loss_function)

    # Build metrics (separate instances for val and test)
    metric_specs = tuple(getattr(settings, "metrics", ()) or ())
    val_metric_routes = build_metric_routes(metric_specs, default_target_key)
    test_metric_routes = build_metric_routes(metric_specs, default_target_key)

    # Build optimizer / scheduler factories
    optimizer_factory = make_optimizer_factory(settings.optimizer)
    scheduler_factory = make_scheduler_factory(getattr(settings, "scheduler", None))

    # Build transform ModuleLists (empty ModuleList when no transforms configured)
    feature_transforms: dict[str, ModuleList] = {}
    for e in feature_entries:
        if e.name is None:
            continue
        t = getattr(e, "transforms", None)
        if t:
            module_list, _ = build_transform_list(t, entry_name=e.name)
        else:
            module_list = ModuleList()
        feature_transforms[e.name] = module_list

    target_transforms: dict[str, ModuleList] = {}
    for e in target_entries:
        if e.name is None:
            continue
        t = getattr(e, "transforms", None)
        if t:
            module_list, _ = build_transform_list(t, entry_name=e.name)
        else:
            module_list = ModuleList()
        target_transforms[e.name] = module_list

    return WrapperComponents(
        loss_fn=loss_fn,
        val_metric_routes=val_metric_routes,
        test_metric_routes=test_metric_routes,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
        feature_transforms=feature_transforms,
        target_transforms=target_transforms,
    )
