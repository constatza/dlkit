"""Shape serialization with versioning and migration support.

This module consolidates all shape serialization logic: enums, metadata
containers, format-specific serializers, migration, and the versioned serializer.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from importlib import import_module
from typing import Any

from .value_objects import ModelFamily, ShapeData, ShapeEntry, ShapeSource

# ---------------------------------------------------------------------------
# Enums and metadata containers
# ---------------------------------------------------------------------------


class SerializationFormat(Enum):
    """Enumeration of supported serialization formats."""

    JSON = "json"
    MSGPACK = "msgpack"
    PICKLE = "pickle"


class SerializationVersion(Enum):
    """Enumeration of serialization format versions."""

    V1_LEGACY = "v1"  # Legacy shape_info format
    V2_ENHANCED = "v2"  # Enhanced metadata format
    V3_MODERN = "v3"  # New modular format (current)


@dataclass(frozen=True, slots=True, kw_only=True)
class SerializationMetadata:
    """Metadata for serialized shape specifications."""

    version: SerializationVersion
    format: SerializationFormat
    created_at: str
    checksum: str | None = None
    migration_history: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True, kw_only=True)
class SerializedShape:
    """Container for serialized shape data with metadata."""

    metadata: SerializationMetadata
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage.

        Returns:
            Dictionary representation of the serialized shape.
        """
        return {
            "metadata": {
                "version": self.metadata.version.value,
                "format": self.metadata.format.value,
                "created_at": self.metadata.created_at,
                "checksum": self.metadata.checksum,
                "migration_history": self.metadata.migration_history,
            },
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SerializedShape:
        """Create from dictionary.

        Args:
            data: Dictionary containing metadata and data keys.

        Returns:
            SerializedShape instance.
        """
        metadata_dict = data.get("metadata", {})
        metadata = SerializationMetadata(
            version=SerializationVersion(metadata_dict.get("version", "v3")),
            format=SerializationFormat(metadata_dict.get("format", "json")),
            created_at=metadata_dict.get("created_at", datetime.now().isoformat()),
            checksum=metadata_dict.get("checksum"),
            migration_history=tuple(metadata_dict.get("migration_history", [])),
        )
        return cls(metadata=metadata, data=data.get("data", {}))


# ---------------------------------------------------------------------------
# Format-specific serializers (private)
# ---------------------------------------------------------------------------


class _V1LegacySerializer:
    """Internal serializer for V1 legacy shape_info format."""

    def serialize(self, shape_data: ShapeData) -> dict[str, Any]:
        """Serialize shape data using V1 legacy format.

        Args:
            shape_data: Shape data to serialize.

        Returns:
            Serialized dictionary in V1 legacy format.
        """
        if len(shape_data.entries) == 1:
            entry = next(iter(shape_data.entries.values()))
            return {"_type": "tuple", "data": list(entry.dimensions)}
        return {
            "_type": "dict",
            "data": {name: list(entry.dimensions) for name, entry in shape_data.entries.items()},
        }

    def deserialize(self, data: dict[str, Any]) -> ShapeData:
        """Deserialize V1 legacy format to shape data.

        Args:
            data: Serialized data in V1 legacy format.

        Returns:
            ShapeData object.
        """
        entries: dict[str, ShapeEntry] = {}
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
            model_family=ModelFamily.EXTERNAL,
            source=ShapeSource.LEGACY_CHECKPOINT,
        )


class _V2EnhancedSerializer:
    """Internal serializer for V2 enhanced metadata format."""

    def serialize(self, shape_data: ShapeData) -> dict[str, Any]:
        """Serialize shape data using V2 format.

        Args:
            shape_data: Shape data to serialize.

        Returns:
            Serialized dictionary in V2 format.
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
            data: Serialized data in V2 format.

        Returns:
            ShapeData object.
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
            shape_data: Shape data to serialize.

        Returns:
            Serialized dictionary in V3 format.
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
            data: Serialized data in V3 format.

        Returns:
            ShapeData object.
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


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


class ShapeFormatMigrator:
    """Handles migration between different serialization formats.

    Supports V1 legacy, V2 enhanced, and V3 modern formats.
    """

    def __init__(self) -> None:
        """Initialize migrator with available serializers."""
        self._serializers: dict[SerializationVersion, Any] = {
            SerializationVersion.V1_LEGACY: _V1LegacySerializer(),
            SerializationVersion.V2_ENHANCED: _V2EnhancedSerializer(),
            SerializationVersion.V3_MODERN: _V3ModernSerializer(),
        }

    def migrate_to_current(self, serialized_shape: SerializedShape) -> SerializedShape:
        """Migrate serialized shape to current format.

        Args:
            serialized_shape: Shape in any supported format.

        Returns:
            SerializedShape in current (V3) format.
        """
        current_version = SerializationVersion.V3_MODERN

        if serialized_shape.metadata.version == current_version:
            return serialized_shape

        source_serializer = self._serializers[serialized_shape.metadata.version]
        shape_data = source_serializer.deserialize(serialized_shape.data)

        target_serializer = self._serializers[current_version]
        new_data = target_serializer.serialize(shape_data)

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
            data: Serialized data dictionary.

        Returns:
            Detected SerializationVersion.
        """
        if "metadata" in data and "version" in data["metadata"]:
            try:
                return SerializationVersion(data["metadata"]["version"])
            except ValueError:
                pass

        if "data" in data and "schema_version" in data["data"]:
            return SerializationVersion.V3_MODERN

        if "entries" in data and "model_family" in data and "source" in data:
            return SerializationVersion.V2_ENHANCED

        if "_type" in data and "data" in data:
            return SerializationVersion.V1_LEGACY

        return SerializationVersion.V3_MODERN

    def can_migrate(
        self, from_version: SerializationVersion, to_version: SerializationVersion
    ) -> bool:
        """Check if migration is supported between versions.

        Args:
            from_version: Source version.
            to_version: Target version.

        Returns:
            True if migration is supported.
        """
        return from_version in self._serializers and to_version in self._serializers


# ---------------------------------------------------------------------------
# Public versioned serializer
# ---------------------------------------------------------------------------


class ShapeFormatSerializer(ABC):
    """Abstract base class for format-specific serializers."""

    @abstractmethod
    def serialize(self, shape_data: ShapeData) -> dict[str, Any]:
        """Serialize shape data to dictionary.

        Args:
            shape_data: Shape data to serialize.

        Returns:
            Serialized dictionary.
        """
        ...

    @abstractmethod
    def deserialize(self, data: dict[str, Any]) -> ShapeData:
        """Deserialize dictionary to shape data.

        Args:
            data: Serialized data.

        Returns:
            ShapeData object.
        """
        ...

    @abstractmethod
    def get_version(self) -> SerializationVersion:
        """Get the version this serializer handles.

        Returns:
            SerializationVersion for this serializer.
        """
        ...


class VersionedShapeSerializer:
    """Main serializer with versioning and migration support."""

    def __init__(
        self,
        format: SerializationFormat = SerializationFormat.JSON,
        migrator: ShapeFormatMigrator | None = None,
    ) -> None:
        """Initialize versioned serializer.

        Args:
            format: Serialization format to use.
            migrator: Optional migrator (creates default if None).
        """
        self._format = format
        self._migrator = migrator or ShapeFormatMigrator()
        self._current_serializer = _V3ModernSerializer()

    def serialize(self, shape_data: ShapeData) -> SerializedShape:
        """Serialize shape data with current format and version.

        Args:
            shape_data: Shape data to serialize.

        Returns:
            SerializedShape with metadata and data.
        """
        data = self._current_serializer.serialize(shape_data)
        metadata = SerializationMetadata(
            version=SerializationVersion.V3_MODERN,
            format=self._format,
            created_at=datetime.now().isoformat(),
        )
        return SerializedShape(metadata=metadata, data=data)

    def deserialize(self, serialized: dict[str, Any] | SerializedShape) -> ShapeData:
        """Deserialize shape data with automatic version detection and migration.

        Args:
            serialized: Serialized data or SerializedShape object.

        Returns:
            ShapeData object.

        Raises:
            ValueError: If deserialization fails.
        """
        if isinstance(serialized, dict):
            if "metadata" in serialized and "data" in serialized:
                serialized_shape = SerializedShape.from_dict(serialized)
            else:
                version = self._migrator.detect_version(serialized)
                metadata = SerializationMetadata(
                    version=version,
                    format=self._format,
                    created_at=datetime.now().isoformat(),
                )
                serialized_shape = SerializedShape(metadata=metadata, data=serialized)
        else:
            serialized_shape = serialized

        current_shape = self._migrator.migrate_to_current(serialized_shape)
        return self._current_serializer.deserialize(current_shape.data)

    def serialize_to_string(self, shape_data: ShapeData) -> str:
        """Serialize shape data to string format.

        Args:
            shape_data: Shape data to serialize.

        Returns:
            Serialized string.

        Raises:
            ValueError: If the format does not support string serialization.
        """
        serialized = self.serialize(shape_data)

        if self._format == SerializationFormat.JSON:
            return json.dumps(serialized.to_dict(), indent=2)
        if self._format == SerializationFormat.MSGPACK:
            try:
                msgpack = import_module("msgpack")
                return msgpack.packb(serialized.to_dict()).decode("latin1")
            except ImportError:
                raise ValueError("msgpack library not available")
        raise ValueError(f"String serialization not supported for format: {self._format}")

    def deserialize_from_string(self, serialized_string: str) -> ShapeData:
        """Deserialize shape data from string format.

        Args:
            serialized_string: Serialized string.

        Returns:
            ShapeData object.

        Raises:
            ValueError: If the format does not support string deserialization.
        """
        if self._format == SerializationFormat.JSON:
            data = json.loads(serialized_string)
        elif self._format == SerializationFormat.MSGPACK:
            try:
                msgpack = import_module("msgpack")
                data = msgpack.unpackb(serialized_string.encode("latin1"), raw=False)
            except ImportError:
                raise ValueError("msgpack library not available")
        else:
            raise ValueError(f"String deserialization not supported for format: {self._format}")

        return self.deserialize(data)

    def get_supported_versions(self) -> list[SerializationVersion]:
        """Get list of supported serialization versions.

        Returns:
            List of supported SerializationVersion values.
        """
        return list(SerializationVersion)

    def get_current_version(self) -> SerializationVersion:
        """Get current serialization version.

        Returns:
            The current SerializationVersion.
        """
        return SerializationVersion.V3_MODERN
