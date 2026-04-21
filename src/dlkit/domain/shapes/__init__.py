"""Shape specification system for DLKit.

Key Components:
- Value Objects: Immutable data containers (ShapeEntry, ShapeData)
- Strategy Pattern: Pluggable validation, serialization, and alias resolution
- Composition: Lightweight ShapeSpec coordinator

Usage:
    from dlkit.domain.shapes import create_shape_spec, ModelFamily, ShapeSource

    shape_spec = create_shape_spec(
        shapes={"x": (784,), "y": (10,)},
        model_family=ModelFamily.DLKIT_NN,
        source=ShapeSource.TRAINING_DATASET
    )
"""

from .core import GraphShapeSpec, IShapeSpec, NullShapeSpec, ShapeSpec, create_shape_spec
from .strategies import ShapeAliasResolver, ShapeSerializer, ShapeValidator, ValidationResult
from .value_objects import ModelFamily, ShapeData, ShapeEntry, ShapeSource

__all__ = [
    # Core interfaces and implementations
    "IShapeSpec",
    "ShapeSpec",
    "GraphShapeSpec",
    "NullShapeSpec",
    "create_shape_spec",
    # Value objects
    "ShapeEntry",
    "ShapeData",
    "ModelFamily",
    "ShapeSource",
    # Strategies
    "ShapeValidator",
    "ShapeSerializer",
    "ShapeAliasResolver",
    "ValidationResult",
]
