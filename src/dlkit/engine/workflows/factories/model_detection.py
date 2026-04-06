"""Model-family detection helpers used during assembly."""

from __future__ import annotations

from typing import Any

from dlkit.domain.nn.detection import (  # noqa: F401
    ABCModelTypeDetector,
    IModelTypeDetector,
    ModelType,
    ModelTypeDetectionChain,
    detect_model_type,
    requires_shape_spec,
)


def should_skip_wrapper(model_settings: Any, dataset: Any) -> bool:
    """Return True if the model should be used directly without a Lightning wrapper.

    Skips wrapping when either:
    - The dataset is a ``ForecastingDataset`` (has its own Lightning lifecycle), or
    - The model class is already a ``LightningModule`` subclass.

    Args:
        model_settings: Component settings with ``name`` and ``module_path`` attributes.
        dataset: Constructed dataset instance.

    Returns:
        bool: True if wrapping should be skipped, False otherwise.
    """
    try:
        from dlkit.engine.data.datasets.timeseries import ForecastingDataset

        if isinstance(dataset, ForecastingDataset):
            return True
    except Exception:
        pass

    try:
        model_ref = getattr(model_settings, "name", None)
        model_cls = None
        if isinstance(model_ref, str):
            from dlkit.infrastructure.utils.general import import_object

            model_cls = import_object(model_ref, fallback_module=model_settings.module_path or "")
        elif isinstance(model_ref, type):
            model_cls = model_ref
        if isinstance(model_cls, type):
            from lightning.pytorch import LightningModule as BaseLightningModule

            if issubclass(model_cls, BaseLightningModule):
                return True
    except Exception:
        pass

    return False
