"""Core shape specification implementation.

This module provides the new lightweight ShapeSpec implementation that uses
composition to coordinate different shape handling strategies.
"""

from __future__ import annotations

from typing import Any

from dlkit.common.shapes import ShapeSpecProtocol as IShapeSpec

from .strategies import ShapeAliasResolver, ShapeSerializer, ShapeValidator, ValidationResult
from .value_objects import ModelFamily, ShapeData, ShapeEntry, ShapeSource


class ShapeSpec(IShapeSpec):
    """Lightweight coordinator using composition of strategy objects.

    This implementation follows the Strategy pattern and Dependency Injection
    to separate concerns and make the system more testable and maintainable.

    Attributes:
        _data: Immutable shape data
        _validator: Strategy for validation logic
        _serializer: Strategy for serialization logic
        _alias_resolver: Strategy for alias resolution
    """

    def __init__(
        self,
        data: ShapeData,
        validator: ShapeValidator | None = None,
        serializer: ShapeSerializer | None = None,
        alias_resolver: ShapeAliasResolver | None = None,
    ):
        """Initialize ShapeSpec with data and strategies.

        Args:
            data: Immutable shape data
            validator: Optional validator strategy (creates default if None)
            serializer: Optional serializer strategy (creates default if None)
            alias_resolver: Optional alias resolver (creates default if None)
        """
        self._data = data
        self._validator = validator or ShapeValidator()
        self._serializer = serializer or ShapeSerializer()
        self._alias_resolver = alias_resolver or ShapeAliasResolver()

        # Cache resolved data for performance
        self._resolved_data: ShapeData | None = None

    def get_input_shape(self) -> tuple[int, ...] | None:
        """Get the primary input shape.

        Uses smart defaults: explicit default_input, then 'x', then first available.
        """
        resolved_data = self._get_resolved_data()
        default_input, _ = self._alias_resolver.resolve_smart_defaults(resolved_data)

        if default_input:
            return resolved_data.get_dimensions(default_input)
        return None

    def get_output_shape(self) -> tuple[int, ...] | None:
        """Get the primary output shape.

        Uses smart defaults: explicit default_output, then 'y', then second available.
        """
        resolved_data = self._get_resolved_data()
        _, default_output = self._alias_resolver.resolve_smart_defaults(resolved_data)

        if default_output:
            return resolved_data.get_dimensions(default_output)
        return None

    def get_shape(self, name: str) -> tuple[int, ...] | None:
        """Get shape for a specific entry name."""
        resolved_data = self._get_resolved_data()
        return resolved_data.get_dimensions(name)

    def has_shape(self, name: str) -> bool:
        """Check if shape exists for entry name."""
        resolved_data = self._get_resolved_data()
        return resolved_data.has_entry(name)

    def get_all_shapes(self) -> dict[str, tuple[int, ...]]:
        """Get all available shapes."""
        resolved_data = self._get_resolved_data()
        return {name: entry.dimensions for name, entry in resolved_data.entries.items()}

    def is_empty(self) -> bool:
        """Check if this shape spec contains no shape information."""
        return self._data.is_empty()

    def model_family(self) -> str:
        """Get the model family this shape spec is designed for."""
        return self._data.model_family.value

    def get_shape_data(self) -> ShapeData:
        """Get the underlying shape data."""
        return self._data

    def get_resolved_data(self) -> ShapeData:
        """Get the shape data with aliases resolved."""
        return self._get_resolved_data()

    def validate(self) -> ValidationResult:
        """Validate this shape specification.

        Returns:
            ValidationResult with any errors or warnings
        """
        return self._validator.validate_collection(self._data)

    def to_dict(self) -> dict[str, Any]:
        """Serialize ShapeSpec to dictionary for storage.

        Returns:
            Serializable dictionary representation
        """
        return self._serializer.serialize(self._data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShapeSpec:
        """Create ShapeSpec from dictionary representation.

        Args:
            data: Dictionary containing shape specification data

        Returns:
            ShapeSpec instance created from the dictionary

        Raises:
            ValueError: If dictionary format is invalid
        """
        from .factory import ShapeSystemFactory

        # Use factory to deserialize the data
        factory = ShapeSystemFactory.create_production_system()
        shape_data = factory.get_serializer().deserialize(data)

        return cls(
            data=shape_data,
            validator=factory.get_validator(),
            serializer=factory.get_serializer(),
            alias_resolver=factory.get_alias_resolver(),
        )

    def with_aliases(self) -> ShapeSpec:
        """Return new ShapeSpec with canonical x/y aliases added if missing.

        Returns:
            New ShapeSpec with canonical aliases
        """
        resolved_data = self._alias_resolver.resolve_aliases(self._data)
        return ShapeSpec(
            data=resolved_data,
            validator=self._validator,
            serializer=self._serializer,
            alias_resolver=self._alias_resolver,
        )

    def with_canonical_aliases(self) -> ShapeSpec:
        """Return ShapeSpec ensuring canonical aliases are present."""
        return self.with_aliases()

    def _get_resolved_data(self) -> ShapeData:
        """Get cached resolved data with aliases."""
        if self._resolved_data is None:
            self._resolved_data = self._alias_resolver.resolve_aliases(self._data)
        return self._resolved_data

    def __str__(self) -> str:
        """String representation for debugging."""
        entries = [f"{name}={entry.dimensions}" for name, entry in self._data.entries.items()]
        return f"ShapeSpec({', '.join(entries)}, family={self._data.model_family.value}, from={self._data.source.value})"

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return f"ShapeSpec(data={self._data!r})"


class GraphShapeSpec:
    """Shape specification for graph neural networks.

    Graph networks have special requirements:
    - Batch-free shape handling (graphs don't have explicit batch dimensions)
    - Node and edge feature shapes
    - Variable-sized graphs within batches
    """

    def __init__(self, data: ShapeData):
        """Initialize graph shape specification.

        Args:
            data: Shape data with graph-specific entries
        """
        if data.model_family != ModelFamily.GRAPH:
            raise ValueError(f"GraphShapeSpec requires GRAPH family, got {data.model_family}")
        self._data = data

    def get_shape_data(self) -> ShapeData:
        """Get the underlying shape data."""
        return self._data

    def get_input_shape(self) -> tuple[int, ...] | None:
        """Get primary input shape (node features)."""
        # For graphs, primary input is typically node features
        return self._data.get_dimensions("x") or self._data.get_dimensions("node_features")

    def get_output_shape(self) -> tuple[int, ...] | None:
        """Get primary output shape."""
        # For graphs, output is typically predictions per node or per graph
        return self._data.get_dimensions("y") or self._data.get_dimensions("out")

    def get_shape(self, name: str) -> tuple[int, ...] | None:
        """Get shape for specific entry name."""
        return self._data.get_dimensions(name)

    def has_shape(self, name: str) -> bool:
        """Check if shape exists for entry name."""
        return self._data.has_entry(name)

    def get_all_shapes(self) -> dict[str, tuple[int, ...]]:
        """Get all available shapes."""
        return {name: entry.dimensions for name, entry in self._data.entries.items()}

    def is_empty(self) -> bool:
        """Check if this shape spec contains no shape information."""
        return self._data.is_empty()

    def model_family(self) -> str:
        """Get the model family this shape spec is designed for."""
        return self._data.model_family.value

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph shape data via standard serializer."""
        serializer = ShapeSerializer()
        return serializer.serialize(self._data)

    def with_canonical_aliases(self) -> IShapeSpec:
        """Graph specs already expose canonical aliases when applicable."""
        return self

    def is_batch_free(self) -> bool:
        """Check if this spec represents batch-free shapes."""
        return True  # Graph shapes are typically batch-free

    def __str__(self) -> str:
        """String representation for debugging."""
        entries = [f"{name}={entry.dimensions}" for name, entry in self._data.entries.items()]
        return f"GraphShapeSpec({', '.join(entries)}, from={self._data.source.value})"


class NullShapeSpec(IShapeSpec):
    """Null shape specification for external models.

    External models (like PyTorch Forecasting models) don't need shape
    specifications, so this implementation provides safe defaults and
    indicates that no shape information is available or required.
    """

    def get_shape_data(self) -> ShapeData:
        """Get the underlying shape data (empty for null spec)."""
        from .value_objects import ModelFamily, ShapeData, ShapeSource

        return ShapeData(
            entries={}, model_family=ModelFamily.EXTERNAL, source=ShapeSource.DEFAULT_FALLBACK
        )

    def get_input_shape(self) -> tuple[int, ...] | None:
        """Return None - external models don't provide shape info."""
        return None

    def get_output_shape(self) -> tuple[int, ...] | None:
        """Return None - external models don't provide shape info."""
        return None

    def get_shape(self, name: str) -> tuple[int, ...] | None:
        """Return None - external models don't provide shape info."""
        return None

    def has_shape(self, name: str) -> bool:
        """Return False - external models don't provide shapes."""
        return False

    def get_all_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return empty dict - external models don't provide shapes."""
        return {}

    def is_empty(self) -> bool:
        """Return True - null spec is always empty."""
        return True

    def model_family(self) -> str:
        """Get the model family this shape spec is designed for."""
        return ModelFamily.EXTERNAL.value

    def to_dict(self) -> dict[str, Any]:
        """Serialize null spec to empty payload."""
        serializer = ShapeSerializer()
        return serializer.serialize(self.get_shape_data())

    def with_canonical_aliases(self) -> IShapeSpec:
        """Null specs are immutable; return self."""
        return self

    def __str__(self) -> str:
        """String representation for debugging."""
        return "NullShapeSpec(external_model)"


def create_shape_spec(
    shapes: dict[str, tuple[int, ...]] | None,
    model_family: ModelFamily = ModelFamily.DLKIT_NN,
    source: ShapeSource = ShapeSource.DEFAULT_FALLBACK,
    default_input: str | None = None,
    default_output: str | None = None,
) -> IShapeSpec:
    """Factory function to create appropriate shape specification.

    Args:
        shapes: Dictionary of shapes or None
        model_family: Model family identifier
        source: Source of shape inference
        default_input: Explicit default input key
        default_output: Explicit default output key

    Returns:
        Appropriate IShapeSpec implementation
    """
    if shapes is None or len(shapes) == 0:
        return NullShapeSpec()

    # Convert raw shapes to ShapeEntry objects
    entries = {name: ShapeEntry(name=name, dimensions=dims) for name, dims in shapes.items()}

    shape_data = ShapeData(
        entries=entries,
        model_family=model_family,
        source=source,
        default_input=default_input,
        default_output=default_output,
    )

    if model_family == ModelFamily.GRAPH:
        return GraphShapeSpec(shape_data)
    if model_family == ModelFamily.EXTERNAL:
        return NullShapeSpec()
    return ShapeSpec(shape_data)
