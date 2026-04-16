"""Parameter selectors for filtering parameter subsets."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from dlkit.domain.nn.parameter_roles import ParameterRole

from .inventory import ParameterDescriptor


@runtime_checkable
class IParameterSelector(Protocol):
    """Protocol for objects that can test whether a parameter is selected.

    Any object implementing this protocol can be used to filter parameters
    by arbitrary predicates.
    """

    def is_satisfied_by(self, descriptor: ParameterDescriptor) -> bool:
        """Test whether a parameter satisfies this selector's criterion.

        Args:
            descriptor: The parameter descriptor to test.

        Returns:
            True if the parameter satisfies the selection criterion.
        """
        ...


class RoleSelector:
    """Selector that matches parameters by their semantic role.

    Attributes:
        _role: The target role to match.
    """

    def __init__(self, role: ParameterRole) -> None:
        """Initialize the role selector.

        Args:
            role: The parameter role to match.
        """
        self._role = role

    def is_satisfied_by(self, descriptor: ParameterDescriptor) -> bool:
        """Test if the parameter has the target role.

        Args:
            descriptor: The parameter descriptor to test.

        Returns:
            True if descriptor.role == self._role.
        """
        return descriptor.role == self._role


class ModulePathSelector:
    """Selector that matches parameters by module path prefix.

    Attributes:
        _prefix: The module path prefix to match.
    """

    def __init__(self, prefix: str) -> None:
        """Initialize the module path selector.

        Args:
            prefix: The module path prefix. Parameters whose module_path
                starts with this prefix will match.
        """
        self._prefix = prefix

    def is_satisfied_by(self, descriptor: ParameterDescriptor) -> bool:
        """Test if the parameter's module path has the target prefix.

        Args:
            descriptor: The parameter descriptor to test.

        Returns:
            True if descriptor.module_path.startswith(self._prefix).
        """
        return descriptor.module_path.startswith(self._prefix)


class IntersectionSelector:
    """Selector that combines multiple selectors with AND logic.

    A parameter is selected only if ALL child selectors select it.

    Attributes:
        _selectors: The child selectors to combine.
    """

    def __init__(self, *selectors: IParameterSelector) -> None:
        """Initialize the intersection selector.

        Args:
            *selectors: Variable number of selectors to combine with AND.
        """
        self._selectors = selectors

    def is_satisfied_by(self, descriptor: ParameterDescriptor) -> bool:
        """Test if the parameter satisfies ALL selectors.

        Args:
            descriptor: The parameter descriptor to test.

        Returns:
            True if all selectors return True.
        """
        return all(sel.is_satisfied_by(descriptor) for sel in self._selectors)


class UnionSelector:
    """Selector that combines multiple selectors with OR logic.

    A parameter is selected if ANY child selector selects it.

    Attributes:
        _selectors: The child selectors to combine.
    """

    def __init__(self, *selectors: IParameterSelector) -> None:
        """Initialize the union selector.

        Args:
            *selectors: Variable number of selectors to combine with OR.
        """
        self._selectors = selectors

    def is_satisfied_by(self, descriptor: ParameterDescriptor) -> bool:
        """Test if the parameter satisfies ANY selector.

        Args:
            descriptor: The parameter descriptor to test.

        Returns:
            True if any selector returns True.
        """
        return any(sel.is_satisfied_by(descriptor) for sel in self._selectors)


class DifferenceSelector:
    """Selector that subtracts one selector from another.

    A parameter is selected if it satisfies the include selector
    AND does NOT satisfy the exclude selector.

    Attributes:
        _include: The selector to include parameters from.
        _exclude: The selector to exclude parameters with.
    """

    def __init__(self, include: IParameterSelector, exclude: IParameterSelector) -> None:
        """Initialize the difference selector.

        Args:
            include: Selector for parameters to include.
            exclude: Selector for parameters to exclude.
        """
        self._include = include
        self._exclude = exclude

    def is_satisfied_by(self, descriptor: ParameterDescriptor) -> bool:
        """Test if the parameter is in include but not in exclude.

        Args:
            descriptor: The parameter descriptor to test.

        Returns:
            True if include matches and exclude does not match.
        """
        return self._include.is_satisfied_by(descriptor) and not self._exclude.is_satisfied_by(
            descriptor
        )


class MuonEligibleSelector:
    """Selector for parameters eligible for the Muon optimizer.

    Muon requires:
    - 2D weight matrices (ndim == 2)
    - Hidden layer weights (role == HIDDEN)

    Parameters matching both criteria are eligible for Muon.
    """

    def is_satisfied_by(self, descriptor: ParameterDescriptor) -> bool:
        """Test if the parameter is eligible for Muon.

        Args:
            descriptor: The parameter descriptor to test.

        Returns:
            True if ndim == 2 and role == HIDDEN.
        """
        return descriptor.ndim == 2 and descriptor.role == ParameterRole.HIDDEN


class NonMuonSelector:
    """Selector for parameters NOT eligible for Muon.

    This is the complement of MuonEligibleSelector.
    Selects all parameters that should NOT be used with Muon.
    """

    def __init__(self) -> None:
        """Initialize the non-muon selector."""
        self._muon_eligible = MuonEligibleSelector()

    def is_satisfied_by(self, descriptor: ParameterDescriptor) -> bool:
        """Test if the parameter is NOT eligible for Muon.

        Args:
            descriptor: The parameter descriptor to test.

        Returns:
            True if the parameter is not Muon-eligible.
        """
        return not self._muon_eligible.is_satisfied_by(descriptor)
