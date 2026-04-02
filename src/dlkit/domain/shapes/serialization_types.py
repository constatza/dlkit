"""Serialization types — internal implementation.

This module contains enums, metadata, and container types for shape serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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
        """Convert to dictionary for storage."""
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
        """Create from dictionary."""
        from datetime import datetime

        metadata_dict = data.get("metadata", {})
        metadata = SerializationMetadata(
            version=SerializationVersion(metadata_dict.get("version", "v3")),
            format=SerializationFormat(metadata_dict.get("format", "json")),
            created_at=metadata_dict.get("created_at", datetime.now().isoformat()),
            checksum=metadata_dict.get("checksum"),
            migration_history=metadata_dict.get("migration_history", []),
        )
        return cls(metadata=metadata, data=data.get("data", {}))
