"""Shape migration logic — internal implementation.

This module contains migration logic and version detection for shape serialization.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .serialization_types import (
    SerializationMetadata,
    SerializationVersion,
    SerializedShape,
)
from .value_objects import ModelFamily, ShapeData, ShapeEntry, ShapeSource


class _V1LegacySerializer:
    """Internal serializer for V1 legacy shape_info format."""

    def serialize(self, shape_data: ShapeData) -> dict[str, Any]:
        """Serialize shape data using V1 legacy format.

        Note: This is primarily for testing and migration scenarios.

        Args:
            shape_data: Shape data to serialize

        Returns:
            Serialized dictionary in V1 legacy format
        """
        # Convert to legacy format
        if len(shape_data.entries) == 1:
            # Single entry - use direct format
            entry = next(iter(shape_data.entries.values()))
            return {"_type": "tuple", "data": list(entry.dimensions)}
        # Multiple entries - use dict format
        return {
            "_type": "dict",
            "data": {name: list(entry.dimensions) for name, entry in shape_data.entries.items()},
        }

    def deserialize(self, data: dict[str, Any]) -> ShapeData:
        """Deserialize V1 legacy format to shape data.

        Args:
            data: Serialized data in V1 legacy format

        Returns:
            ShapeData object
        """
        entries = {}
        shape_type = data.get("_type")
        shape_data_raw = data.get("data")

        if shape_data_raw is None:
            return ShapeData(
                entries=entries,
                model_family=ModelFamily.EXTERNAL,
                source=ShapeSource.LEGACY_CHECKPOINT,
            )

        if shape_type == "dict":
            for key, dims in shape_data_raw.items():
                entries[key] = ShapeEntry(name=key, dimensions=tuple(dims))
        elif shape_type in ("tuple", "torch.Size"):
            entries["x"] = ShapeEntry(name="x", dimensions=tuple(shape_data_raw))

        return ShapeData(
            entries=entries,
            model_family=ModelFamily.EXTERNAL,  # Cannot determine from legacy
            source=ShapeSource.LEGACY_CHECKPOINT,
        )


class _V2EnhancedSerializer:
    """Internal serializer for V2 enhanced metadata format."""

    def serialize(self, shape_data: ShapeData) -> dict[str, Any]:
        """Serialize shape data using V2 format.

        Args:
            shape_data: Shape data to serialize

        Returns:
            Serialized dictionary in V2 format
        """
        return {
            "entries": {name: list(entry.dimensions) for name, entry in shape_data.entries.items()},
            "model_family": shape_data.model_family.value,
            "source": shape_data.source.value,
            "default_input": shape_data.default_input,
            "default_output": shape_data.default_output,
        }

    def deserialize(self, data: dict[str, Any]) -> ShapeData:
        """Deserialize V2 format to shape data.

        Args:
            data: Serialized data in V2 format

        Returns:
            ShapeData object
        """
        entries = {
            name: ShapeEntry(name=name, dimensions=tuple(dims))
            for name, dims in data["entries"].items()
        }

        return ShapeData(
            entries=entries,
            model_family=ModelFamily(data["model_family"]),
            source=ShapeSource(data["source"]),
            default_input=data.get("default_input"),
            default_output=data.get("default_output"),
        )


class _V3ModernSerializer:
    """Internal serializer for V3 modern format (current)."""

    def serialize(self, shape_data: ShapeData) -> dict[str, Any]:
        """Serialize shape data using V3 format.

        Args:
            shape_data: Shape data to serialize

        Returns:
            Serialized dictionary in V3 format
        """
        return {
            "entries": {
                name: {"dimensions": list(entry.dimensions), "metadata": {"name": entry.name}}
                for name, entry in shape_data.entries.items()
            },
            "model_family": shape_data.model_family.value,
            "source": shape_data.source.value,
            "default_input": shape_data.default_input,
            "default_output": shape_data.default_output,
            "schema_version": "3.0",
        }

    def deserialize(self, data: dict[str, Any]) -> ShapeData:
        """Deserialize V3 format to shape data.

        Args:
            data: Serialized data in V3 format

        Returns:
            ShapeData object
        """
        entries = {}
        for name, entry_data in data["entries"].items():
            dimensions = tuple(entry_data["dimensions"])
            entries[name] = ShapeEntry(name=name, dimensions=dimensions)

        return ShapeData(
            entries=entries,
            model_family=ModelFamily(data["model_family"]),
            source=ShapeSource(data["source"]),
            default_input=data.get("default_input"),
            default_output=data.get("default_output"),
        )


class ShapeFormatMigrator:
    """Handles migration between different serialization formats."""

    def __init__(self):
        """Initialize migrator with available serializers."""
        self._serializers = {
            SerializationVersion.V1_LEGACY: _V1LegacySerializer(),
            SerializationVersion.V2_ENHANCED: _V2EnhancedSerializer(),
            SerializationVersion.V3_MODERN: _V3ModernSerializer(),
        }

    def migrate_to_current(self, serialized_shape: SerializedShape) -> SerializedShape:
        """Migrate serialized shape to current format.

        Args:
            serialized_shape: Shape in any supported format

        Returns:
            SerializedShape in current format (V3)
        """
        current_version = SerializationVersion.V3_MODERN

        if serialized_shape.metadata.version == current_version:
            # Already current version
            return serialized_shape

        # Deserialize using appropriate format
        source_serializer = self._serializers[serialized_shape.metadata.version]
        shape_data = source_serializer.deserialize(serialized_shape.data)

        # Serialize using current format
        target_serializer = self._serializers[current_version]
        new_data = target_serializer.serialize(shape_data)

        # Update metadata
        new_metadata = SerializationMetadata(
            version=current_version,
            format=serialized_shape.metadata.format,
            created_at=datetime.now().isoformat(),
            migration_history=serialized_shape.metadata.migration_history
            + (f"migrated_from_{serialized_shape.metadata.version.value}",),
        )

        return SerializedShape(metadata=new_metadata, data=new_data)

    def detect_version(self, data: dict[str, Any]) -> SerializationVersion:
        """Auto-detect serialization version from data.

        Args:
            data: Serialized data

        Returns:
            Detected SerializationVersion
        """
        # Check for explicit version in metadata
        if "metadata" in data and "version" in data["metadata"]:
            try:
                return SerializationVersion(data["metadata"]["version"])
            except ValueError:
                pass

        # Check for V3 schema version
        if "data" in data and "schema_version" in data["data"]:
            return SerializationVersion.V3_MODERN

        # Check for V2 enhanced format markers
        if "entries" in data and "model_family" in data and "source" in data:
            return SerializationVersion.V2_ENHANCED

        # Check for V1 legacy format markers
        if "_type" in data and "data" in data:
            return SerializationVersion.V1_LEGACY

        # Default to current version
        return SerializationVersion.V3_MODERN

    def can_migrate(
        self, from_version: SerializationVersion, to_version: SerializationVersion
    ) -> bool:
        """Check if migration is supported between versions.

        Args:
            from_version: Source version
            to_version: Target version

        Returns:
            True if migration is supported
        """
        return from_version in self._serializers and to_version in self._serializers
