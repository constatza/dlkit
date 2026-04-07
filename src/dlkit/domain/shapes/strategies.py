"""Strategy classes for shape handling operations.

This module implements the Strategy pattern to separate different concerns
of shape handling: validation, serialization, and alias resolution.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any

from .value_objects import ShapeData, ShapeEntry


# TODO: Investigate whether shape-spec validation needs a dedicated result carrier.
# If this remains public, rename it to something domain-specific such as
# ShapeValidationReport; otherwise remove it with the broader validation surface.
@dataclass(frozen=True, slots=True, kw_only=True)
class ValidationResult:
    """Result of shape validation operation."""

    is_valid: bool
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "errors", tuple(self.errors))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def add_error(self, error: str) -> ValidationResult:
        """Add validation error."""
        return replace(self, is_valid=False, errors=self.errors + (error,))

    def add_warning(self, warning: str) -> ValidationResult:
        """Add validation warning."""
        return replace(self, warnings=self.warnings + (warning,))

    @classmethod
    def success(cls) -> ValidationResult:
        """Create successful validation result."""
        return cls(is_valid=True)

    @classmethod
    def failure(cls, errors: Iterable[str]) -> ValidationResult:
        """Create failed validation result."""
        return cls(is_valid=False, errors=tuple(errors))


class ShapeValidator:
    """Single responsibility: shape validation logic.

    This class now delegates to the specification-based validation engine
    for extensible and composable validation rules.
    """

    def __init__(self, validation_engine=None):
        """Initialize validator with optional validation engine.

        Args:
            validation_engine: Optional validation engine (creates default if None)
        """
        # Import here to avoid circular dependency
        from .specifications import ShapeValidationEngine

        self.validation_engine = validation_engine or ShapeValidationEngine()

    def validate_entry(self, entry: ShapeEntry) -> ValidationResult:
        """Validate single shape entry.

        Args:
            entry: Shape entry to validate

        Returns:
            ValidationResult with any errors found
        """
        result = ValidationResult.success()

        # Basic validation is handled by ShapeEntry.__post_init__
        # Additional business logic validation can go here

        return result

    def validate_collection(self, data: ShapeData) -> ValidationResult:
        """Validate complete shape data collection.

        Args:
            data: Shape data to validate

        Returns:
            ValidationResult with any errors found
        """
        # Delegate to specification-based validation engine
        return self.validation_engine.validate(data)

    def get_validation_engine(self):
        """Get the underlying validation engine."""
        return self.validation_engine


class ShapeSerializer:
    """Single responsibility: shape serialization and deserialization.

    This class delegates to the versioned serialization system.
    """

    def __init__(self, versioned_serializer=None):
        """Initialize serializer with optional versioned serializer.

        Args:
            versioned_serializer: Optional versioned serializer (creates default if None)
        """
        # Import here to avoid circular dependency
        from .serialization import VersionedShapeSerializer

        self._versioned_serializer = versioned_serializer or VersionedShapeSerializer()

    def serialize(self, data: ShapeData) -> dict[str, Any]:
        """Serialize ShapeData to dictionary.

        Args:
            data: Shape data to serialize

        Returns:
            Serializable dictionary representation
        """
        serialized_shape = self._versioned_serializer.serialize(data)
        return serialized_shape.to_dict()

    def deserialize(self, raw: dict[str, Any]) -> ShapeData:
        """Deserialize dictionary to ShapeData.

        Args:
            raw: Serialized shape data

        Returns:
            ShapeData object

        Raises:
            ValueError: If data is invalid or missing required fields
        """
        return self._versioned_serializer.deserialize(raw)

    def get_versioned_serializer(self):
        """Get the underlying versioned serializer."""
        return self._versioned_serializer


class ShapeAliasResolver:
    """Single responsibility: alias resolution for backward compatibility.

    Handles resolution of default input/output aliases (x/y pattern) and
    provides smart defaults for missing entries.
    """

    def resolve_aliases(self, data: ShapeData) -> ShapeData:
        """Resolve aliases and add smart defaults.

        Args:
            data: Original shape data

        Returns:
            New ShapeData with resolved aliases
        """
        if data.is_empty():
            return data

        new_entries = dict(data.entries)

        # Add x alias if missing and we have entries
        if "x" not in new_entries and data.entries:
            # Use explicit default_input, or first available entry
            if data.default_input and data.default_input in data.entries:
                source_entry = data.entries[data.default_input]
                new_entries["x"] = ShapeEntry(name="x", dimensions=source_entry.dimensions)
            else:
                # Use first available entry as x
                first_entry = next(iter(data.entries.values()))
                new_entries["x"] = ShapeEntry(name="x", dimensions=first_entry.dimensions)

        # Add y alias if missing
        if "y" not in new_entries:
            if data.default_output and data.default_output in data.entries:
                # Use explicit default_output
                source_entry = data.entries[data.default_output]
                new_entries["y"] = ShapeEntry(name="y", dimensions=source_entry.dimensions)
            elif len(data.entries) > 1:
                # Use second entry as y, or duplicate x if only one entry
                entries_list = list(data.entries.values())
                if len(entries_list) > 1:
                    second_entry = entries_list[1]
                    new_entries["y"] = ShapeEntry(name="y", dimensions=second_entry.dimensions)
                elif "x" in new_entries:
                    # Duplicate x as y for single-entry cases
                    new_entries["y"] = ShapeEntry(name="y", dimensions=new_entries["x"].dimensions)
            elif "x" in new_entries:
                # Duplicate x as y
                new_entries["y"] = ShapeEntry(name="y", dimensions=new_entries["x"].dimensions)

        return ShapeData(
            entries=new_entries,
            model_family=data.model_family,
            source=data.source,
            default_input=data.default_input,
            default_output=data.default_output,
        )

    def resolve_smart_defaults(self, data: ShapeData) -> tuple[str | None, str | None]:
        """Determine smart defaults for input and output shapes.

        Args:
            data: Shape data to analyze

        Returns:
            Tuple of (default_input, default_output) names
        """
        default_input = None
        default_output = None

        if data.is_empty():
            return default_input, default_output

        # Use explicit defaults if available and valid
        if data.default_input and data.has_entry(data.default_input):
            default_input = data.default_input
        elif data.has_entry("x"):
            default_input = "x"
        else:
            # Use first available entry
            default_input = next(iter(data.entries.keys()), None)

        if data.default_output and data.has_entry(data.default_output):
            default_output = data.default_output
        elif data.has_entry("y"):
            default_output = "y"
        elif len(data.entries) > 1:
            # Use second entry if available
            entries_list = list(data.entries.keys())
            default_output = entries_list[1] if len(entries_list) > 1 else None

        return default_input, default_output
