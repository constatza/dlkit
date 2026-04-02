"""Shape specification public re-export surface."""

from .shape_specifications import (
    DimensionRangeSpecification,
    ModelFamilyCompatibilitySpecification,
    NonEmptyShapeSpecification,
    PositiveDimensionsSpecification,
    RequiredEntriesSpecification,
    UniqueEntryNamesSpecification,
)
from .specification_base import (
    AndSpecification,
    NotSpecification,
    OrSpecification,
    ShapeSpecification,
)
from .validation_engine import ShapeSpecificationBuilder, ShapeValidationEngine

__all__ = [
    "ShapeSpecification",
    "AndSpecification",
    "OrSpecification",
    "NotSpecification",
    "RequiredEntriesSpecification",
    "PositiveDimensionsSpecification",
    "NonEmptyShapeSpecification",
    "ModelFamilyCompatibilitySpecification",
    "DimensionRangeSpecification",
    "UniqueEntryNamesSpecification",
    "ShapeValidationEngine",
    "ShapeSpecificationBuilder",
]
