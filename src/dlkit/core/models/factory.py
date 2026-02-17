"""Pure factory function for model construction.

Replaces the _create_abc_model method in ProcessingLightningWrapper.
"""
from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch.nn as nn
    from dlkit.core.shape_specs.simple_inference import ShapeSummary


def build_model(
    model_cls: type[nn.Module],
    shape: ShapeSummary | None,
    kwargs: dict[str, Any],
) -> nn.Module:
    """Construct a model. Tries shape-aware args first, falls back to kwargs-only.

    For FFNN models: tries in_features/out_features.
    For CAE models: tries in_channels/in_length.
    For external models (no shape): passes kwargs only.

    Args:
        model_cls: Model class to instantiate.
        shape: Shape summary from dataset inference, or None for external models.
        kwargs: Additional keyword arguments from model settings.

    Returns:
        Constructed model instance.

    Raises:
        TypeError: If model cannot be constructed with any argument combination.
    """
    if shape is None:
        return model_cls(**kwargs)

    # Try FFNN convention (in_features / out_features)
    with suppress(TypeError):
        return model_cls(
            in_features=shape.in_features,
            out_features=shape.out_features,
            **kwargs,
        )

    # Try Conv/CAE convention (in_channels / in_length)
    with suppress(TypeError, IndexError):
        return model_cls(
            in_channels=shape.in_channels,
            in_length=shape.in_length,
            **kwargs,
        )

    # External model — pass kwargs only (no shape args)
    return model_cls(**kwargs)
