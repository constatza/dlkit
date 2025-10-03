"""Modern factory for shape handling system.

This module provides the new factory implementation that uses the registry
pattern for model family detection and clean dependency injection.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .core import IShapeSpec, create_shape_spec
from .value_objects import ShapeData, ModelFamily, ShapeSource
from .registry import ModelFamilyRegistry, ModelFamilyRegistryFactory
from .strategies import ShapeValidator, ShapeSerializer, ShapeAliasResolver
from .performance import CachingShapeInferencer, BatchShapeProcessor, LRUShapeCache


class ShapeSystemFactory:
    """Factory for creating complete shape handling system components.

    This factory coordinates the creation of all shape system components
    with proper dependency injection and configuration.
    """

    def __init__(self,
                 model_registry: Optional[ModelFamilyRegistry] = None,
                 validator: Optional[ShapeValidator] = None,
                 serializer: Optional[ShapeSerializer] = None,
                 alias_resolver: Optional[ShapeAliasResolver] = None):
        """Initialize factory with optional dependencies.

        Args:
            model_registry: Registry for model family detection
            validator: Shape validation strategy
            serializer: Shape serialization strategy
            alias_resolver: Shape alias resolution strategy
        """
        self._model_registry = model_registry or ModelFamilyRegistryFactory.create_default_registry()
        self._validator = validator or ShapeValidator()
        self._serializer = serializer or ShapeSerializer()
        self._alias_resolver = alias_resolver or ShapeAliasResolver()

    def create_shape_spec_from_data(self,
                                   shapes: Dict[str, tuple[int, ...]] | None,
                                   model_settings: Any,
                                   source: ShapeSource = ShapeSource.DEFAULT_FALLBACK) -> IShapeSpec:
        """Create shape specification from raw shape data and model settings.

        Args:
            shapes: Dictionary of shapes or None
            model_settings: Model configuration settings for family detection
            source: Source of shape inference

        Returns:
            Appropriate IShapeSpec implementation
        """
        if shapes is None or len(shapes) == 0:
            model_family = self._model_registry.detect_family(model_settings)
            return create_shape_spec(None, model_family, source)

        model_family = self._model_registry.detect_family(model_settings)

        return create_shape_spec(
            shapes=shapes,
            model_family=model_family,
            source=source
        )

    def create_shape_spec_from_serialized(self, serialized_data: Dict[str, Any]) -> IShapeSpec:
        """Create shape specification from serialized data.

        Args:
            serialized_data: Dictionary containing serialized shape data

        Returns:
            Reconstructed IShapeSpec implementation

        Raises:
            ValueError: If serialized data is invalid
        """
        try:
            shape_data = self._serializer.deserialize(serialized_data)
            return create_shape_spec(
                shapes={name: entry.dimensions for name, entry in shape_data.entries.items()},
                model_family=shape_data.model_family,
                source=shape_data.source,
                default_input=shape_data.default_input,
                default_output=shape_data.default_output
            )
        except Exception as e:
            raise ValueError(f"Failed to deserialize shape data: {e}") from e

    def create_shape_spec_from_legacy(self, legacy_data: Dict[str, Any]) -> IShapeSpec | None:
        """Create shape specification from legacy checkpoint format.

        Args:
            legacy_data: Legacy shape_info from old checkpoints

        Returns:
            IShapeSpec implementation or None if conversion fails
        """
        shape_data = self._serializer.deserialize_legacy_format(legacy_data)
        if shape_data is None:
            return None

        return create_shape_spec(
            shapes={name: entry.dimensions for name, entry in shape_data.entries.items()},
            model_family=shape_data.model_family,
            source=shape_data.source
        )

    def get_model_registry(self) -> ModelFamilyRegistry:
        """Get the model family registry."""
        return self._model_registry

    def get_validator(self) -> ShapeValidator:
        """Get the shape validator."""
        return self._validator

    def get_serializer(self) -> ShapeSerializer:
        """Get the shape serializer."""
        return self._serializer

    def get_alias_resolver(self) -> ShapeAliasResolver:
        """Get the alias resolver."""
        return self._alias_resolver

    def get_batch_processor(self) -> BatchShapeProcessor:
        """Get batch processor for performance optimization."""
        return BatchShapeProcessor(
            validator=self._validator,
            serializer=self._serializer
        )

    def create_caching_inferencer(self, base_chain) -> CachingShapeInferencer:
        """Create caching inferencer wrapper.

        Args:
            base_chain: Base inference chain to wrap

        Returns:
            CachingShapeInferencer with performance optimizations
        """
        return CachingShapeInferencer(
            base_chain=base_chain,
            cache=LRUShapeCache(max_size=1000, ttl_seconds=3600)
        )

    @classmethod
    def create_production_system(cls) -> ShapeSystemFactory:
        """Create factory configured for production use.

        Returns:
            ShapeSystemFactory with production-ready configuration
        """
        return cls(
            model_registry=ModelFamilyRegistryFactory.create_default_registry(),
            validator=ShapeValidator(),
            serializer=ShapeSerializer(),
            alias_resolver=ShapeAliasResolver()
        )

    @classmethod
    def create_testing_system(cls) -> ShapeSystemFactory:
        """Create factory configured for testing.

        Returns:
            ShapeSystemFactory with minimal configuration for testing
        """
        return cls(
            model_registry=ModelFamilyRegistryFactory.create_minimal_registry(),
            validator=ShapeValidator(),
            serializer=ShapeSerializer(),
            alias_resolver=ShapeAliasResolver()
        )