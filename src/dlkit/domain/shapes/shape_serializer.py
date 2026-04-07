"""Shape serialization — internal implementation.

This module contains public serializer interfaces and the versioned serializer.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from importlib import import_module
from typing import Any

from .serialization_types import (
    SerializationFormat,
    SerializationMetadata,
    SerializationVersion,
    SerializedShape,
)
from .shape_migrator import ShapeFormatMigrator, _V3ModernSerializer
from .value_objects import ShapeData


class ShapeFormatSerializer(ABC):
    """Abstract base class for format-specific serializers."""

    @abstractmethod
    def serialize(self, shape_data: ShapeData) -> dict[str, Any]:
        """Serialize shape data to dictionary.

        Args:
            shape_data: Shape data to serialize

        Returns:
            Serialized dictionary
        """
        ...

    @abstractmethod
    def deserialize(self, data: dict[str, Any]) -> ShapeData:
        """Deserialize dictionary to shape data.

        Args:
            data: Serialized data

        Returns:
            ShapeData object
        """
        ...

    @abstractmethod
    def get_version(self) -> SerializationVersion:
        """Get the version this serializer handles."""
        ...


class VersionedShapeSerializer:
    """Main serializer with versioning and migration support."""

    def __init__(
        self,
        format: SerializationFormat = SerializationFormat.JSON,
        migrator: ShapeFormatMigrator | None = None,
    ):
        """Initialize versioned serializer.

        Args:
            format: Serialization format to use
            migrator: Optional migrator (creates default if None)
        """
        self._format = format
        self._migrator = migrator or ShapeFormatMigrator()
        self._current_serializer = _V3ModernSerializer()

    def serialize(self, shape_data: ShapeData) -> SerializedShape:
        """Serialize shape data with current format and version.

        Args:
            shape_data: Shape data to serialize

        Returns:
            SerializedShape with metadata and data
        """
        from datetime import datetime

        # Serialize using current format
        data = self._current_serializer.serialize(shape_data)

        # Create metadata
        metadata = SerializationMetadata(
            version=SerializationVersion.V3_MODERN,
            format=self._format,
            created_at=datetime.now().isoformat(),
        )

        return SerializedShape(metadata=metadata, data=data)

    def deserialize(self, serialized: dict[str, Any] | SerializedShape) -> ShapeData:
        """Deserialize shape data with automatic version detection and migration.

        Args:
            serialized: Serialized data or SerializedShape object

        Returns:
            ShapeData object

        Raises:
            ValueError: If deserialization fails
        """
        from datetime import datetime

        if isinstance(serialized, dict):
            # Convert dict to SerializedShape
            if "metadata" in serialized and "data" in serialized:
                serialized_shape = SerializedShape.from_dict(serialized)
            else:
                # Legacy format without metadata wrapper
                version = self._migrator.detect_version(serialized)
                metadata = SerializationMetadata(
                    version=version, format=self._format, created_at=datetime.now().isoformat()
                )
                serialized_shape = SerializedShape(metadata=metadata, data=serialized)
        else:
            serialized_shape = serialized

        # Migrate to current version if needed
        current_shape = self._migrator.migrate_to_current(serialized_shape)

        # Deserialize using current format
        return self._current_serializer.deserialize(current_shape.data)

    def serialize_to_string(self, shape_data: ShapeData) -> str:
        """Serialize shape data to string format.

        Args:
            shape_data: Shape data to serialize

        Returns:
            Serialized string
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
        else:
            raise ValueError(f"String serialization not supported for format: {self._format}")

    def deserialize_from_string(self, serialized_string: str) -> ShapeData:
        """Deserialize shape data from string format.

        Args:
            serialized_string: Serialized string

        Returns:
            ShapeData object
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
            List of supported versions
        """
        return list(SerializationVersion)

    def get_current_version(self) -> SerializationVersion:
        """Get current serialization version.

        Returns:
            Current version
        """
        return SerializationVersion.V3_MODERN
