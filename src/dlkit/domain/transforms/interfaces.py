"""Compatibility re-exports for canonical transform Protocols.

The transform capability contracts live in :mod:`dlkit.domain.transforms.base`.
This module keeps a stable import path without maintaining duplicate ABC-based
interfaces.
"""

from .base import (
    FittableTransform,
    IncrementalFittableTransform,
    InvertibleTransform,
    ShapeAwareTransform,
    ShapeInferringTransform,
)

__all__ = [
    "FittableTransform",
    "IncrementalFittableTransform",
    "InvertibleTransform",
    "ShapeAwareTransform",
    "ShapeInferringTransform",
]
