"""Specification pattern for shape validation.

This module implements the Specification pattern to centralize and compose
validation rules for shape data, making the validation system highly extensible
and testable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Set, List

from .value_objects import ShapeData, ShapeEntry, ModelFamily
from .strategies import ValidationResult


class ShapeSpecification(ABC):
    """Base class for shape validation specifications.

    The Specification pattern allows complex validation rules to be composed
    and combined in flexible ways while keeping each rule focused and testable.
    """

    @abstractmethod
    def is_satisfied_by(self, shape_data: ShapeData) -> ValidationResult:
        """Check if the specification is satisfied by the given shape data.

        Args:
            shape_data: Shape data to validate

        Returns:
            ValidationResult indicating success/failure and any messages
        """
        ...

    def and_(self, other: ShapeSpecification) -> AndSpecification:
        """Combine this specification with another using logical AND.

        Args:
            other: Specification to combine with

        Returns:
            Combined specification that requires both to be satisfied
        """
        return AndSpecification(self, other)

    def or_(self, other: ShapeSpecification) -> OrSpecification:
        """Combine this specification with another using logical OR.

        Args:
            other: Specification to combine with

        Returns:
            Combined specification that requires either to be satisfied
        """
        return OrSpecification(self, other)

    def not_(self) -> NotSpecification:
        """Negate this specification.

        Returns:
            Specification that is satisfied when this one is not
        """
        return NotSpecification(self)


class AndSpecification(ShapeSpecification):
    """Specification that requires all sub-specifications to be satisfied."""

    def __init__(self, left: ShapeSpecification, right: ShapeSpecification):
        """Initialize with two specifications to combine.

        Args:
            left: First specification
            right: Second specification
        """
        self._left = left
        self._right = right

    def is_satisfied_by(self, shape_data: ShapeData) -> ValidationResult:
        """Check if both specifications are satisfied.

        Args:
            shape_data: Shape data to validate

        Returns:
            ValidationResult combining results from both specifications
        """
        left_result = self._left.is_satisfied_by(shape_data)
        right_result = self._right.is_satisfied_by(shape_data)

        combined_result = ValidationResult(
            is_valid=left_result.is_valid and right_result.is_valid,
            errors=left_result.errors + right_result.errors,
            warnings=left_result.warnings + right_result.warnings,
        )

        return combined_result


class OrSpecification(ShapeSpecification):
    """Specification that requires at least one sub-specification to be satisfied."""

    def __init__(self, left: ShapeSpecification, right: ShapeSpecification):
        """Initialize with two specifications to combine.

        Args:
            left: First specification
            right: Second specification
        """
        self._left = left
        self._right = right

    def is_satisfied_by(self, shape_data: ShapeData) -> ValidationResult:
        """Check if either specification is satisfied.

        Args:
            shape_data: Shape data to validate

        Returns:
            ValidationResult - success if either spec passes, combined errors if both fail
        """
        left_result = self._left.is_satisfied_by(shape_data)
        right_result = self._right.is_satisfied_by(shape_data)

        if left_result.is_valid or right_result.is_valid:
            # At least one passed - combine warnings but no errors
            return ValidationResult(
                is_valid=True, errors=(), warnings=left_result.warnings + right_result.warnings
            )
        else:
            # Both failed - combine all errors and warnings
            return ValidationResult(
                is_valid=False,
                errors=left_result.errors + right_result.errors,
                warnings=left_result.warnings + right_result.warnings,
            )


class NotSpecification(ShapeSpecification):
    """Specification that is satisfied when the wrapped specification is not."""

    def __init__(self, spec: ShapeSpecification):
        """Initialize with specification to negate.

        Args:
            spec: Specification to negate
        """
        self._spec = spec

    def is_satisfied_by(self, shape_data: ShapeData) -> ValidationResult:
        """Check if the wrapped specification is NOT satisfied.

        Args:
            shape_data: Shape data to validate

        Returns:
            ValidationResult - success if wrapped spec fails, failure if it passes
        """
        result = self._spec.is_satisfied_by(shape_data)

        if result.is_valid:
            return ValidationResult.failure([
                "Negated specification failed: wrapped spec was satisfied"
            ])
        else:
            return ValidationResult.success()


class RequiredEntriesSpecification(ShapeSpecification):
    """Specification that validates required shape entries exist."""

    def __init__(self, required_entries: Set[str]):
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


class ShapeValidationEngine:
    """Centralized validation engine using specification pattern.

    This engine maintains specifications for different model families and
    provides methods to validate shape data against appropriate specifications.
    """

    def __init__(self):
        """Initialize validation engine with default specifications."""
        self._family_specifications: dict[ModelFamily, ShapeSpecification] = {}
        self._register_default_specifications()

    def register_specification(self, family: ModelFamily, spec: ShapeSpecification) -> None:
        """Register validation specification for a model family.

        Args:
            family: Model family to register specification for
            spec: Validation specification to use
        """
        self._family_specifications[family] = spec

    def validate(self, shape_data: ShapeData) -> ValidationResult:
        """Validate shape data using appropriate family specification.

        Args:
            shape_data: Shape data to validate

        Returns:
            ValidationResult from family-specific validation
        """
        family = shape_data.model_family
        spec = self._family_specifications.get(family)

        if spec is None:
            # No specific validation for this family
            return self._validate_with_basic_spec(shape_data)

        return spec.is_satisfied_by(shape_data)

    def validate_with_spec(
        self, shape_data: ShapeData, spec: ShapeSpecification
    ) -> ValidationResult:
        """Validate shape data with custom specification.

        Args:
            shape_data: Shape data to validate
            spec: Custom specification to use

        Returns:
            ValidationResult from custom specification
        """
        return spec.is_satisfied_by(shape_data)

    def _validate_with_basic_spec(self, shape_data: ShapeData) -> ValidationResult:
        """Apply basic validation for unknown model families.

        Args:
            shape_data: Shape data to validate

        Returns:
            ValidationResult from basic validation
        """
        basic_spec = PositiveDimensionsSpecification().and_(UniqueEntryNamesSpecification())
        return basic_spec.is_satisfied_by(shape_data)

    def _register_default_specifications(self) -> None:
        """Register default validation specifications for built-in model families."""
        # Standard DLKit NN validation
        dlkit_spec = (
            PositiveDimensionsSpecification()
            .and_(UniqueEntryNamesSpecification())
            .and_(ModelFamilyCompatibilitySpecification(ModelFamily.DLKIT_NN))
            .and_(DimensionRangeSpecification(min_dimensions=1, max_dimensions=5))
        )
        self.register_specification(ModelFamily.DLKIT_NN, dlkit_spec)

        # Graph model validation
        graph_spec = (
            PositiveDimensionsSpecification()
            .and_(UniqueEntryNamesSpecification())
            .and_(ModelFamilyCompatibilitySpecification(ModelFamily.GRAPH))
            .and_(DimensionRangeSpecification(min_dimensions=1, max_dimensions=3))
        )
        self.register_specification(ModelFamily.GRAPH, graph_spec)

        # Time series validation
        timeseries_spec = (
            PositiveDimensionsSpecification()
            .and_(UniqueEntryNamesSpecification())
            .and_(ModelFamilyCompatibilitySpecification(ModelFamily.TIMESERIES))
            .and_(DimensionRangeSpecification(min_dimensions=2, max_dimensions=4))
        )
        self.register_specification(ModelFamily.TIMESERIES, timeseries_spec)

        # External model validation (minimal)
        external_spec = ModelFamilyCompatibilitySpecification(ModelFamily.EXTERNAL)
        self.register_specification(ModelFamily.EXTERNAL, external_spec)


class ShapeSpecificationBuilder:
    """Builder for composing complex shape specifications."""

    def __init__(self):
        """Initialize empty specification builder."""
        self._specifications: List[ShapeSpecification] = []

    def require_entries(self, entries: Set[str]) -> ShapeSpecificationBuilder:
        """Add required entries specification.

        Args:
            entries: Set of required entry names

        Returns:
            Self for method chaining
        """
        self._specifications.append(RequiredEntriesSpecification(entries))
        return self

    def positive_dimensions(self) -> ShapeSpecificationBuilder:
        """Add positive dimensions specification.

        Returns:
            Self for method chaining
        """
        self._specifications.append(PositiveDimensionsSpecification())
        return self

    def non_empty(self) -> ShapeSpecificationBuilder:
        """Add non-empty specification.

        Returns:
            Self for method chaining
        """
        self._specifications.append(NonEmptyShapeSpecification())
        return self

    def compatible_with_family(self, family: ModelFamily) -> ShapeSpecificationBuilder:
        """Add model family compatibility specification.

        Args:
            family: Model family to validate compatibility for

        Returns:
            Self for method chaining
        """
        self._specifications.append(ModelFamilyCompatibilitySpecification(family))
        return self

    def dimension_range(self, min_dims: int, max_dims: int) -> ShapeSpecificationBuilder:
        """Add dimension range specification.

        Args:
            min_dims: Minimum dimensions allowed
            max_dims: Maximum dimensions allowed

        Returns:
            Self for method chaining
        """
        self._specifications.append(DimensionRangeSpecification(min_dims, max_dims))
        return self

    def unique_names(self) -> ShapeSpecificationBuilder:
        """Add unique entry names specification.

        Returns:
            Self for method chaining
        """
        self._specifications.append(UniqueEntryNamesSpecification())
        return self

    def build(self) -> ShapeSpecification:
        """Build the composite specification.

        Returns:
            Composite specification combining all added specifications

        Raises:
            ValueError: If no specifications were added
        """
        if not self._specifications:
            raise ValueError("Cannot build specification with no rules")

        # Combine all specifications with AND
        result = self._specifications[0]
        for spec in self._specifications[1:]:
            result = result.and_(spec)

        return result
