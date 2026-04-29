"""Explicit shape-aware model construction helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, Self, runtime_checkable

if TYPE_CHECKING:
    from torch import nn

    from dlkit.common.shapes import ShapeSummary


@runtime_checkable
class ShapeConsumer(Protocol):
    """Protocol for models that explicitly opt into shape-based construction."""

    @classmethod
    def from_shape(cls, shape: ShapeSummary, **kwargs: Any) -> Self:
        """Construct the model from a dataset-derived shape summary."""
        ...


type ModelFactory = type[nn.Module] | Callable[..., nn.Module]


def build_model(
    model_cls: ModelFactory, shape: ShapeSummary | None, kwargs: dict[str, Any]
) -> nn.Module:
    """Construct a model with explicit opt-in shape handling only.

    Models that need shape injection must implement ``from_shape()``.
    All others are constructed with the explicit kwargs only.
    """
    if shape is not None:
        from_shape = getattr(model_cls, "from_shape", None)
        if callable(from_shape):
            return from_shape(shape, **dict(kwargs))
    return model_cls(**kwargs)


__all__ = [
    "ModelFactory",
    "ShapeConsumer",
    "build_model",
]
