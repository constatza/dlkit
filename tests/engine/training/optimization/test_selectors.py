"""Tests for parameter selector implementations."""

from __future__ import annotations

from dlkit.domain.nn.parameter_roles import ParameterRole
from dlkit.engine.training.optimization.inventory import ParameterDescriptor
from dlkit.engine.training.optimization.selectors import (
    DifferenceSelector,
    IntersectionSelector,
    ModulePathSelector,
    MuonEligibleSelector,
    NonMuonSelector,
    RoleSelector,
    UnionSelector,
)


class TestRoleSelector:
    """Tests for RoleSelector."""

    def test_role_selector_matches_correct_role(
        self, hidden_2d_descriptor: ParameterDescriptor
    ) -> None:
        """RoleSelector(HIDDEN) matches HIDDEN descriptor.

        Args:
            hidden_2d_descriptor: 2-D descriptor with HIDDEN role.
        """
        selector = RoleSelector(ParameterRole.HIDDEN)
        assert selector.is_satisfied_by(hidden_2d_descriptor)

    def test_role_selector_rejects_wrong_role(
        self, bias_1d_descriptor: ParameterDescriptor
    ) -> None:
        """RoleSelector(HIDDEN) rejects BIAS descriptor.

        Args:
            bias_1d_descriptor: 1-D descriptor with BIAS role.
        """
        selector = RoleSelector(ParameterRole.HIDDEN)
        assert not selector.is_satisfied_by(bias_1d_descriptor)


class TestModulePathSelector:
    """Tests for ModulePathSelector."""

    def test_module_path_selector_matches_prefix(
        self, hidden_2d_descriptor: ParameterDescriptor
    ) -> None:
        """ModulePathSelector matches when module_path starts with prefix.

        Args:
            hidden_2d_descriptor: Descriptor with module_path="layer".
        """
        selector = ModulePathSelector("lay")
        assert selector.is_satisfied_by(hidden_2d_descriptor)

    def test_module_path_selector_rejects_wrong_prefix(
        self, hidden_2d_descriptor: ParameterDescriptor
    ) -> None:
        """ModulePathSelector rejects when module_path doesn't start with prefix.

        Args:
            hidden_2d_descriptor: Descriptor with module_path="layer".
        """
        selector = ModulePathSelector("encoder")
        assert not selector.is_satisfied_by(hidden_2d_descriptor)


class TestIntersectionSelector:
    """Tests for IntersectionSelector."""

    def test_intersection_selector_both_pass(
        self, hidden_2d_descriptor: ParameterDescriptor
    ) -> None:
        """IntersectionSelector matches when both child selectors match.

        Args:
            hidden_2d_descriptor: 2-D HIDDEN descriptor.
        """
        selector = IntersectionSelector(
            RoleSelector(ParameterRole.HIDDEN),
            ModulePathSelector("layer"),
        )
        assert selector.is_satisfied_by(hidden_2d_descriptor)

    def test_intersection_selector_one_fails(
        self, hidden_2d_descriptor: ParameterDescriptor
    ) -> None:
        """IntersectionSelector rejects when any child selector fails.

        Args:
            hidden_2d_descriptor: 2-D HIDDEN descriptor.
        """
        selector = IntersectionSelector(
            RoleSelector(ParameterRole.HIDDEN),
            ModulePathSelector("encoder"),  # Doesn't match
        )
        assert not selector.is_satisfied_by(hidden_2d_descriptor)


class TestUnionSelector:
    """Tests for UnionSelector."""

    def test_union_selector_either_passes(self, hidden_2d_descriptor: ParameterDescriptor) -> None:
        """UnionSelector matches when any child selector matches.

        Args:
            hidden_2d_descriptor: 2-D HIDDEN descriptor.
        """
        selector = UnionSelector(
            RoleSelector(ParameterRole.BIAS),  # Doesn't match
            RoleSelector(ParameterRole.HIDDEN),  # Matches
        )
        assert selector.is_satisfied_by(hidden_2d_descriptor)

    def test_union_selector_none_passes(self, hidden_2d_descriptor: ParameterDescriptor) -> None:
        """UnionSelector rejects when no child selector matches.

        Args:
            hidden_2d_descriptor: 2-D HIDDEN descriptor.
        """
        selector = UnionSelector(
            RoleSelector(ParameterRole.BIAS),
            RoleSelector(ParameterRole.OUTPUT),
        )
        assert not selector.is_satisfied_by(hidden_2d_descriptor)


class TestDifferenceSelector:
    """Tests for DifferenceSelector."""

    def test_difference_selector_include_minus_exclude(
        self, hidden_2d_descriptor: ParameterDescriptor
    ) -> None:
        """DifferenceSelector matches when include passes and exclude fails.

        Args:
            hidden_2d_descriptor: 2-D HIDDEN descriptor.
        """
        selector = DifferenceSelector(
            include=RoleSelector(ParameterRole.HIDDEN),
            exclude=ModulePathSelector("encoder"),
        )
        assert selector.is_satisfied_by(hidden_2d_descriptor)

    def test_difference_selector_excluded_param(
        self, hidden_2d_descriptor: ParameterDescriptor
    ) -> None:
        """DifferenceSelector rejects when exclude also matches.

        Args:
            hidden_2d_descriptor: 2-D HIDDEN descriptor with module_path="layer".
        """
        selector = DifferenceSelector(
            include=RoleSelector(ParameterRole.HIDDEN),
            exclude=ModulePathSelector("layer"),
        )
        assert not selector.is_satisfied_by(hidden_2d_descriptor)


class TestMuonEligibleSelector:
    """Tests for MuonEligibleSelector."""

    def test_muon_eligible_requires_2d_hidden(
        self, hidden_2d_descriptor: ParameterDescriptor
    ) -> None:
        """MuonEligibleSelector matches 2-D HIDDEN parameters.

        Args:
            hidden_2d_descriptor: 2-D HIDDEN descriptor.
        """
        selector = MuonEligibleSelector()
        assert selector.is_satisfied_by(hidden_2d_descriptor)

    def test_muon_eligible_rejects_1d(self, bias_1d_descriptor: ParameterDescriptor) -> None:
        """MuonEligibleSelector rejects 1-D parameters.

        Args:
            bias_1d_descriptor: 1-D BIAS descriptor.
        """
        selector = MuonEligibleSelector()
        assert not selector.is_satisfied_by(bias_1d_descriptor)

    def test_muon_eligible_rejects_unknown_role(
        self, unknown_descriptor: ParameterDescriptor
    ) -> None:
        """MuonEligibleSelector rejects parameters with UNKNOWN role.

        Args:
            unknown_descriptor: 2-D UNKNOWN descriptor.
        """
        selector = MuonEligibleSelector()
        assert not selector.is_satisfied_by(unknown_descriptor)


class TestNonMuonSelector:
    """Tests for NonMuonSelector."""

    def test_non_muon_selector_rejects_muon_eligible(
        self, hidden_2d_descriptor: ParameterDescriptor
    ) -> None:
        """NonMuonSelector rejects Muon-eligible parameters.

        Args:
            hidden_2d_descriptor: 2-D HIDDEN descriptor (Muon-eligible).
        """
        selector = NonMuonSelector()
        assert not selector.is_satisfied_by(hidden_2d_descriptor)

    def test_non_muon_selector_accepts_1d(self, bias_1d_descriptor: ParameterDescriptor) -> None:
        """NonMuonSelector accepts 1-D parameters.

        Args:
            bias_1d_descriptor: 1-D BIAS descriptor.
        """
        selector = NonMuonSelector()
        assert selector.is_satisfied_by(bias_1d_descriptor)

    def test_non_muon_selector_accepts_unknown_role(
        self, unknown_descriptor: ParameterDescriptor
    ) -> None:
        """NonMuonSelector accepts UNKNOWN role parameters.

        Args:
            unknown_descriptor: 2-D UNKNOWN descriptor.
        """
        selector = NonMuonSelector()
        assert selector.is_satisfied_by(unknown_descriptor)
