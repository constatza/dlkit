"""Domain-specific shape specifications — internal implementation.

This module contains concrete specification implementations for validating
shape data against specific constraints and requirements.
"""

from __future__ import annotations

from .specification_base import ShapeSpecification
from .strategies import ValidationResult
from .value_objects import ModelFamily, ShapeData


class RequiredEntriesSpecification(ShapeSpecification):
    """Specification that validates required shape entries exist."""

    def __init__(self, required_entries: set[str]):
        """Initialize with set of required entry names.

        Args:
            required_entries: Set of entry names that must be present
        """
        self._required_entries = required_entries

    def is_satisfied_by(self, shape_data: ShapeData) -> ValidationResult:
        """Check if all required entries are present.

        Args:
            shape_data: Shape data to validate

        Returns:
            ValidationResult indicating missing entries if any
        """
        result = ValidationResult.success()

        missing_entries = self._required_entries - shape_data.entry_names()
        if missing_entries:
            for entry in missing_entries:
                result = result.add_error(f"Required shape entry '{entry}' is missing")

        return result


class PositiveDimensionsSpecification(ShapeSpecification):
    """Specification that validates all dimensions are positive integers."""

    def is_satisfied_by(self, shape_data: ShapeData) -> ValidationResult:
        """Check if all dimensions are positive integers.

        Args:
            shape_data: Shape data to validate

        Returns:
            ValidationResult indicating any invalid dimensions
        """
        result = ValidationResult.success()

        for entry in shape_data.entries.values():
            for i, dim in enumerate(entry.dimensions):
                if not isinstance(dim, int) or dim <= 0:
                    result = result.add_error(
                        f"Dimension {i} of entry '{entry.name}' must be positive integer, "
                        f"got {type(dim).__name__}: {dim}"
                    )

        return result


class NonEmptyShapeSpecification(ShapeSpecification):
    """Specification that validates shape data is not empty."""

    def is_satisfied_by(self, shape_data: ShapeData) -> ValidationResult:
        """Check if shape data contains at least one entry.

        Args:
            shape_data: Shape data to validate

        Returns:
            ValidationResult indicating if shape data is empty
        """
        if shape_data.is_empty():
            return ValidationResult.failure(["Shape data cannot be empty"])
        return ValidationResult.success()


class ModelFamilyCompatibilitySpecification(ShapeSpecification):
    """Specification that validates shapes are compatible with model family."""

    def __init__(self, family: ModelFamily):
        """Initialize with target model family.

        Args:
            family: Model family to validate compatibility for
        """
        self._family = family

    def is_satisfied_by(self, shape_data: ShapeData) -> ValidationResult:
        """Check if shapes are compatible with model family.

        Args:
            shape_data: Shape data to validate

        Returns:
            ValidationResult with family-specific validation results
        """
        result = ValidationResult.success()

        if self._family == ModelFamily.GRAPH:
            # Graph models typically need 'x' for node features
            if not shape_data.has_entry("x"):
                result = result.add_warning(
                    "Graph models typically require 'x' entry for node features"
                )

            # Check for common graph entries
            graph_entries = {"x", "edge_index", "edge_attr", "y"}
            if not any(shape_data.has_entry(entry) for entry in graph_entries):
                result = result.add_warning(
                    "Graph models typically require graph-specific entries (x, edge_index, etc.)"
                )

        elif self._family == ModelFamily.DLKIT_NN:
            # Standard NN models benefit from x/y entries
            if not shape_data.has_entry("x"):
                result = result.add_warning(
                    "Standard NN models typically require 'x' entry for input"
                )
            if not shape_data.has_entry("y"):
                result = result.add_warning(
                    "Standard NN models typically require 'y' entry for output"
                )

        elif self._family == ModelFamily.EXTERNAL:
            # External models shouldn't have shapes
            if not shape_data.is_empty():
                result = result.add_warning(
                    "External models typically don't use shape specifications"
                )

        elif self._family == ModelFamily.TIMESERIES:
            # Time series models need sequence-like shapes
            if shape_data.has_entry("x"):
                x_dims = shape_data.get_dimensions("x")
                if x_dims and len(x_dims) < 2:
                    result = result.add_warning(
                        "Time series models typically require multi-dimensional input (sequence, features)"
                    )

        return result


class DimensionRangeSpecification(ShapeSpecification):
    """Specification that validates dimensions are within specified ranges."""

    def __init__(self, min_dimensions: int = 1, max_dimensions: int = 10):
        """Initialize with dimension range constraints.

        Args:
            min_dimensions: Minimum number of dimensions allowed
            max_dimensions: Maximum number of dimensions allowed
        """
        self._min_dimensions = min_dimensions
        self._max_dimensions = max_dimensions

    def is_satisfied_by(self, shape_data: ShapeData) -> ValidationResult:
        """Check if all shapes have dimensions within specified range.

        Args:
            shape_data: Shape data to validate

        Returns:
            ValidationResult indicating any out-of-range dimensions
        """
        result = ValidationResult.success()

        for entry in shape_data.entries.values():
            num_dims = len(entry.dimensions)
            if num_dims < self._min_dimensions:
                result = result.add_error(
                    f"Entry '{entry.name}' has {num_dims} dimensions, "
                    f"minimum required: {self._min_dimensions}"
                )
            elif num_dims > self._max_dimensions:
                result = result.add_error(
                    f"Entry '{entry.name}' has {num_dims} dimensions, "
                    f"maximum allowed: {self._max_dimensions}"
                )

        return result


class UniqueEntryNamesSpecification(ShapeSpecification):
    """Specification that validates all entry names are unique (case-insensitive)."""

    def is_satisfied_by(self, shape_data: ShapeData) -> ValidationResult:
        """Check if all entry names are unique when compared case-insensitively.

        Args:
            shape_data: Shape data to validate

        Returns:
            ValidationResult indicating any duplicate names
        """
        result = ValidationResult.success()

        # Check for case-insensitive duplicates
        seen_names = set()
        for entry_name in shape_data.entry_names():
            lower_name = entry_name.lower()
            if lower_name in seen_names:
                result = result.add_error(
                    f"Duplicate entry name (case-insensitive): '{entry_name}'"
                )
            seen_names.add(lower_name)

        return result
