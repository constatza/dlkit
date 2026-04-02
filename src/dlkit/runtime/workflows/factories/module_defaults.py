"""Runtime-owned defaults for settings-driven component resolution.

`tools.config` remains pure data and does not hardcode package ownership.
Runtime builders apply these defaults immediately before component construction.
"""

from __future__ import annotations

from typing import Any

from dlkit.tools.config.datamodule_settings import DataModuleSettings
from dlkit.tools.config.dataset_settings import DatasetSettings
from dlkit.tools.config.model_components import (
    LossComponentSettings,
    MetricComponentSettings,
    ModelComponentSettings,
    WrapperComponentSettings,
)
from dlkit.tools.config.transform_settings import TransformSettings

_MODEL_MODULE = "dlkit.domain.nn"
_DATASET_MODULE = "dlkit.runtime.data.datasets"
_DATAMODULE_MODULE = "dlkit.runtime.adapters.lightning.datamodules"
_WRAPPER_MODULE = "dlkit.runtime.adapters.lightning"
_LOSS_MODULE = "dlkit.domain.losses"
_TRANSFORM_MODULE = "dlkit.domain.transforms"
_METRIC_MODULE = "torchmetrics.regression"


def with_runtime_module_defaults[T: Any](settings: T) -> T:
    """Return settings with runtime-owned module-path defaults applied.

    The input object is returned unchanged when it already specifies a module path
    or when the settings type has no runtime-owned default.
    """

    module_path = getattr(settings, "module_path", None)
    if module_path:
        return settings

    resolved_module = _default_module_path(settings)
    if resolved_module is None:
        return settings

    model_copy = getattr(settings, "model_copy", None)
    if callable(model_copy):
        return model_copy(update={"module_path": resolved_module})
    return settings


def _default_module_path(settings: Any) -> str | None:
    if isinstance(settings, ModelComponentSettings):
        return _MODEL_MODULE
    if isinstance(settings, DatasetSettings):
        return _DATASET_MODULE
    if isinstance(settings, DataModuleSettings):
        return _DATAMODULE_MODULE
    if isinstance(settings, WrapperComponentSettings):
        return _WRAPPER_MODULE
    if isinstance(settings, LossComponentSettings):
        return _LOSS_MODULE
    if isinstance(settings, MetricComponentSettings):
        return _METRIC_MODULE
    if isinstance(settings, TransformSettings):
        return _TRANSFORM_MODULE
    return None
