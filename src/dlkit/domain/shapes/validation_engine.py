"""Shape validation engine — internal implementation.

This module contains the validation engine and builder for specifications.
"""

from __future__ import annotations

from .shape_specifications import (
    DimensionRangeSpecification,
    ModelFamilyCompatibilitySpecification,
    NonEmptyShapeSpecification,
    PositiveDimensionsSpecification,
    RequiredEntriesSpecification,
    UniqueEntryNamesSpecification,
)
from .specification_base import ShapeSpecification
from .strategies import ValidationResult
from .value_objects import ModelFamily, ShapeData


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
        self._specifications: list[ShapeSpecification] = []

    def require_entries(self, entries: set[str]) -> ShapeSpecificationBuilder:
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
