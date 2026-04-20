"""Pre-build wrapper components before injection into core wrappers.

All FactoryProvider calls for Lightning wrappers live here.
Core wrappers accept WrapperComponents and never call FactoryProvider directly.
"""

from __future__ import annotations

from torch import nn
from torch.nn import ModuleList

from dlkit.engine.adapters.lightning.metrics_routing import MetricRoute
from dlkit.engine.adapters.lightning.transform_builder import build_transform_list
from dlkit.engine.adapters.lightning.wrapper_types import WrapperComponents
from dlkit.infrastructure.config.core.context import BuildContext
from dlkit.infrastructure.config.core.factories import FactoryProvider
from dlkit.infrastructure.config.data_entries import DataEntry, is_feature_entry, is_target_entry
from dlkit.infrastructure.config.model_components import (
    LossComponentSettings,
    WrapperComponentSettings,
)
from dlkit.infrastructure.config.optimizer_policy import OptimizerPolicySettings

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

    # Build optimization program settings (deferred to wrapper for model access)
    optimizer_policy_settings = settings.optimizer_policy
    if optimizer_policy_settings is None:
        # Fallback: migrate legacy OptimizerSettings → typed OptimizerSpec subclass.
        from dlkit.infrastructure.config.optimizer_component import (
            AdamSettings,
            AdamWSettings,
            LBFGSSettings,
            MuonSettings,
            OptimizerSpec,
            SchedulerSpec,
        )

        raw = settings.optimizer
        name = str(raw.name or "AdamW")
        lr_raw = raw.lr
        lr = float(lr_raw) if isinstance(lr_raw, (int, float)) else 1e-3
        wd_raw = raw.weight_decay
        wd = float(wd_raw) if isinstance(wd_raw, (int, float)) else 0.0

        default_optimizer: OptimizerSpec
        match name.lower():
            case "adam":
                default_optimizer = AdamSettings(lr=lr, weight_decay=wd)
            case "lbfgs":
                default_optimizer = LBFGSSettings(lr=lr)
            case "muon":
                default_optimizer = MuonSettings(lr=lr)
            case _:
                default_optimizer = AdamWSettings(lr=lr, weight_decay=wd)

        # Migrate legacy SchedulerSettings to typed subclass.
        # The SchedulerSettings class from optimizer_settings.py is now legacy.
        # We convert it to ReduceLROnPlateauSettings (which matches SchedulerSettings defaults).
        from dlkit.infrastructure.config.optimizer_component import (
            ReduceLROnPlateauSettings,
        )

        raw_scheduler = settings.scheduler
        if raw_scheduler:
            # Legacy SchedulerSettings has factor, patience, min_lr — map to ReduceLROnPlateauSettings
            default_scheduler: SchedulerSpec | None = ReduceLROnPlateauSettings(
                factor=getattr(raw_scheduler, "factor", 0.5),
                patience=getattr(raw_scheduler, "patience", 1000),
                min_lr=getattr(raw_scheduler, "min_lr", 1e-8),
            )
        else:
            default_scheduler = None

        optimizer_policy_settings = OptimizerPolicySettings(
            default_optimizer=default_optimizer,
            default_scheduler=default_scheduler,
        )

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
        optimizer_policy_settings=optimizer_policy_settings,
        feature_transforms=feature_transforms,
        target_transforms=target_transforms,
    )
