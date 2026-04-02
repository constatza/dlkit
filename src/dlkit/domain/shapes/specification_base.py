"""Base specification classes — internal implementation.

This module contains the Specification pattern ABC and composite specifications.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .strategies import ValidationResult
from .value_objects import ShapeData


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
            return ValidationResult.failure(
                ["Negated specification failed: wrapped spec was satisfied"]
            )
        return ValidationResult.success()
