"""Modern shape specification system for DLKit.

This module provides a complete rewrite of DLKit's shape handling system
following SOLID principles and modern design patterns.

Key Components:
- Value Objects: Immutable data containers (ShapeEntry, ShapeData)
- Strategy Pattern: Pluggable validation, serialization, and alias resolution
- Composition: Lightweight ShapeSpec coordinator
- Registry Pattern: Extensible model family detection

Usage:
    from dlkit.domain.shapes import create_shape_spec, ModelFamily, ShapeSource

    shape_spec = create_shape_spec(
        shapes={"x": (784,), "y": (10,)},
        model_family=ModelFamily.DLKIT_NN,
        source=ShapeSource.TRAINING_DATASET
    )
"""

from .core import GraphShapeSpec, IShapeSpec, NullShapeSpec, ShapeSpec, create_shape_spec
from .factory import ShapeSystemFactory
from .inference import (
    CheckpointMetadataStrategy,
    ConfigurationStrategy,
    DatasetSamplingStrategy,
    DefaultFallbackStrategy,
    GraphDatasetStrategy,
    InferenceContext,
    ShapeInferenceChain,
    ShapeInferenceEngine,
    ShapeInferenceStrategy,
)
from .registry import ModelFamilyDetector, ModelFamilyRegistry, ModelFamilyRegistryFactory
from .serialization import (
    SerializationFormat,
    SerializationMetadata,
    SerializationVersion,
    SerializedShape,
    ShapeFormatMigrator,
    ShapeFormatSerializer,
    VersionedShapeSerializer,
)
from .specifications import (
    DimensionRangeSpecification,
    ModelFamilyCompatibilitySpecification,
    NonEmptyShapeSpecification,
    PositiveDimensionsSpecification,
    RequiredEntriesSpecification,
    ShapeSpecification,
    ShapeSpecificationBuilder,
    ShapeValidationEngine,
    UniqueEntryNamesSpecification,
)
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
    "NonEmptyShapeSpecification",
    "ModelFamilyCompatibilitySpecification",
    "DimensionRangeSpecification",
    "UniqueEntryNamesSpecification",
    # Inference system
    "ShapeInferenceStrategy",
    "ShapeInferenceChain",
    "ShapeInferenceEngine",
    "InferenceContext",
    "CheckpointMetadataStrategy",
    "ConfigurationStrategy",
    "DatasetSamplingStrategy",
    "DefaultFallbackStrategy",
    "GraphDatasetStrategy",
    # Serialization system
    "ShapeFormatSerializer",
    "VersionedShapeSerializer",
    "SerializationFormat",
    "SerializationVersion",
    "SerializationMetadata",
    "SerializedShape",
    "ShapeFormatMigrator",
]
