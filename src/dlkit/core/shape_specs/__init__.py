"""Modern shape specification system for DLKit.

This module provides a complete rewrite of DLKit's shape handling system
following SOLID principles and modern design patterns.

Key Components:
- Value Objects: Immutable data containers (ShapeEntry, ShapeData)
- Strategy Pattern: Pluggable validation, serialization, and alias resolution
- Composition: Lightweight ShapeSpec coordinator
- Registry Pattern: Extensible model family detection

Usage:
    from dlkit.core.shape_specs import create_shape_spec, ModelFamily, ShapeSource

    shape_spec = create_shape_spec(
        shapes={"x": (784,), "y": (10,)},
        model_family=ModelFamily.DLKIT_NN,
        source=ShapeSource.TRAINING_DATASET
    )
"""

from .core import IShapeSpec, ShapeSpec, GraphShapeSpec, NullShapeSpec, create_shape_spec
from .value_objects import ShapeEntry, ShapeData, ModelFamily, ShapeSource
from .strategies import ShapeValidator, ShapeSerializer, ShapeAliasResolver, ValidationResult
from .registry import ModelFamilyRegistry, ModelFamilyDetector, ModelFamilyRegistryFactory
from .factory import ShapeSystemFactory
from .specifications import (
    ShapeSpecification,
    ShapeValidationEngine,
    ShapeSpecificationBuilder,
    RequiredEntriesSpecification,
    PositiveDimensionsSpecification,
    ModelFamilyCompatibilitySpecification,
    DimensionRangeSpecification,
)
from .inference import (
    ShapeInferenceStrategy,
    ShapeInferenceChain,
    ShapeInferenceEngine,
    InferenceContext,
    CheckpointMetadataStrategy,
    DatasetSamplingStrategy,
)
from .serialization import (
    VersionedShapeSerializer,
    SerializationFormat,
    SerializationVersion,
    SerializedShape,
    ShapeFormatMigrator,
)
from .checkpoint_loader import CheckpointShapeLoader
from .simple_inference import ShapeSummary, infer_shapes_from_dataset

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
    # Registry system
    "ModelFamilyRegistry",
    "ModelFamilyDetector",
    "ModelFamilyRegistryFactory",
    # Factory system
    "ShapeSystemFactory",
    # Specification system
    "ShapeSpecification",
    "ShapeValidationEngine",
    "ShapeSpecificationBuilder",
    "RequiredEntriesSpecification",
    "PositiveDimensionsSpecification",
    "ModelFamilyCompatibilitySpecification",
    "DimensionRangeSpecification",
    # Inference system
    "ShapeInferenceStrategy",
    "ShapeInferenceChain",
    "ShapeInferenceEngine",
    "InferenceContext",
    "CheckpointMetadataStrategy",
    "DatasetSamplingStrategy",
    # Serialization system
    "VersionedShapeSerializer",
    "SerializationFormat",
    "SerializationVersion",
    "SerializedShape",
    "ShapeFormatMigrator",
    # Checkpoint utilities
    "CheckpointShapeLoader",
    # Simple inference system
    "ShapeSummary",
    "infer_shapes_from_dataset",
]
