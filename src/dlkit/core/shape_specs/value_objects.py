"""Value objects for shape handling system.

This module contains pure data containers with minimal validation logic,
following the Value Object pattern for immutable shape data representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple


class ModelFamily(Enum):
    """Enumeration of supported model families."""
    DLKIT_NN = "dlkit_nn"
    GRAPH = "graph"
    EXTERNAL = "external"
    TIMESERIES = "timeseries"


class ShapeSource(Enum):
    """Enumeration of shape inference sources."""
    TRAINING_DATASET = "training_dataset"
    CHECKPOINT_METADATA = "checkpoint_metadata"
    LEGACY_CHECKPOINT = "legacy_checkpoint"
    CONFIGURATION = "configuration"
    DEFAULT_FALLBACK = "default_fallback"
    GRAPH_DATASET = "graph_dataset"
    ENTRY_CONFIGS = "entry_configs"


@dataclass(frozen=True, slots=True)
class ShapeEntry:
    """Single shape entry with basic validation.

    Represents a named shape with dimensions, enforcing basic invariants
    like positive dimensions and string naming.

    Attributes:
        name: Entry name identifier
        dimensions: Tuple of positive integer dimensions
    """
    name: str
    dimensions: Tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate shape entry invariants."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"Shape entry name must be non-empty string, got: {self.name}")

        if not isinstance(self.dimensions, tuple):
            raise ValueError(f"Dimensions must be tuple, got {type(self.dimensions)}: {self.dimensions}")

        if not self.dimensions:
            raise ValueError(f"Dimensions cannot be empty for entry '{self.name}'")

        for i, dim in enumerate(self.dimensions):
            if not isinstance(dim, int) or dim <= 0:
                raise ValueError(
                    f"Dimension {i} for entry '{self.name}' must be positive integer, "
                    f"got {type(dim)}: {dim}"
                )

    def __str__(self) -> str:
        """String representation for debugging."""
        return f"{self.name}={self.dimensions}"


@dataclass(frozen=True, slots=True)
class ShapeData:
    """Collection of validated shape entries with metadata.

    Represents a complete shape specification with all entries, model family
    information, and source tracking for debugging and validation.

    Attributes:
        entries: Dictionary mapping entry names to ShapeEntry objects
        model_family: Target model family for these shapes
        source: How these shapes were inferred/obtained
        default_input: Optional explicit default input key
        default_output: Optional explicit default output key
    """
    entries: Dict[str, ShapeEntry]
    model_family: ModelFamily
    source: ShapeSource
    default_input: str | None = None
    default_output: str | None = None

    def __post_init__(self) -> None:
        """Validate shape data collection invariants."""
        if not isinstance(self.entries, dict):
            raise ValueError(f"Entries must be dictionary, got {type(self.entries)}")

        if not isinstance(self.model_family, ModelFamily):
            raise ValueError(f"Model family must be ModelFamily enum, got {type(self.model_family)}")

        if not isinstance(self.source, ShapeSource):
            raise ValueError(f"Source must be ShapeSource enum, got {type(self.source)}")

        # Validate default keys exist if specified
        if self.default_input is not None and self.default_input not in self.entries:
            raise ValueError(f"Default input '{self.default_input}' not found in entries")

        if self.default_output is not None and self.default_output not in self.entries:
            raise ValueError(f"Default output '{self.default_output}' not found in entries")

        # Validate all entries are ShapeEntry objects
        for name, entry in self.entries.items():
            if not isinstance(entry, ShapeEntry):
                raise ValueError(f"Entry '{name}' must be ShapeEntry, got {type(entry)}")
            if entry.name != name:
                raise ValueError(f"Entry name mismatch: key '{name}' vs entry.name '{entry.name}'")

    def has_entry(self, name: str) -> bool:
        """Check if shape data contains entry with given name."""
        return name in self.entries

    def get_entry(self, name: str) -> ShapeEntry | None:
        """Get shape entry by name, returning None if not found."""
        return self.entries.get(name)

    def get_dimensions(self, name: str) -> Tuple[int, ...] | None:
        """Get dimensions for entry by name, returning None if not found."""
        entry = self.entries.get(name)
        return entry.dimensions if entry else None

    def entry_names(self) -> set[str]:
        """Get set of all entry names."""
        return set(self.entries.keys())

    def is_empty(self) -> bool:
        """Check if shape data contains no entries."""
        return len(self.entries) == 0

    def with_defaults(self, default_input: str | None, default_output: str | None) -> ShapeData:
        """Return new ShapeData with updated defaults."""
        return ShapeData(
            entries=self.entries,
            model_family=self.model_family,
            source=self.source,
            default_input=default_input,
            default_output=default_output
        )

    def __len__(self) -> int:
        """Number of shape entries."""
        return len(self.entries)

    def __str__(self) -> str:
        """String representation for debugging."""
        entries_str = ", ".join(str(entry) for entry in self.entries.values())
        return f"ShapeData({entries_str}, family={self.model_family.value}, source={self.source.value})"

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"ShapeData(entries={self.entries}, model_family={self.model_family}, "
            f"source={self.source}, default_input={self.default_input}, "
            f"default_output={self.default_output})"
        )