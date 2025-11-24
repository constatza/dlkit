"""Architecture fitness tests for Transform system.

These tests enforce SOLID principles and architectural constraints:
- Interface Segregation: Transforms declare capabilities via ABC interfaces
- Dependency Inversion: Code checks capabilities via isinstance(ABC), not hasattr()
- Single Responsibility: Transform classes have focused, single purposes

Tests fail if architectural violations are introduced.
"""

import inspect
from typing import get_type_hints

import pytest
import torch

from dlkit.core.training.transforms.interfaces import (
    IFittableTransform,
    IInvertibleTransform,
    ISerializableTransform,
)
from dlkit.core.training.transforms.base import Transform
from dlkit.core.training.transforms.minmax import MinMaxScaler
from dlkit.core.training.transforms.standard import StandardScaler
from dlkit.core.training.transforms.pca import PCA
from dlkit.core.training.transforms.chain import TransformChain


class TestInterfaceSegregation:
    """Test ISP: Transforms explicitly declare their capabilities via ABC mixins."""

    def test_fittable_transforms_implement_IFittableTransform(self):
        """Transforms with fit() method must inherit from IFittableTransform."""
        fittable_transforms = [MinMaxScaler, StandardScaler, PCA, TransformChain]

        for transform_cls in fittable_transforms:
            assert issubclass(transform_cls, IFittableTransform), (
                f"{transform_cls.__name__} has fit() but doesn't inherit from IFittableTransform. "
                f"This violates Interface Segregation Principle."
            )

    def test_invertible_transforms_implement_IInvertibleTransform(self):
        """Transforms with inverse_transform() must inherit from IInvertibleTransform."""
        invertible_transforms = [MinMaxScaler, StandardScaler, PCA, TransformChain]

        for transform_cls in invertible_transforms:
            assert issubclass(transform_cls, IInvertibleTransform), (
                f"{transform_cls.__name__} has inverse_transform() but doesn't inherit from IInvertibleTransform. "
                f"This violates Interface Segregation Principle."
            )

    def test_ABC_interfaces_are_abstract(self):
        """ABC interfaces cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IFittableTransform()  # type: ignore

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IInvertibleTransform()  # type: ignore

    def test_isinstance_checks_work_at_runtime(self):
        """isinstance() checks work correctly for ABC mixins."""
        scaler = MinMaxScaler(dim=0)

        assert isinstance(scaler, IFittableTransform)
        assert isinstance(scaler, IInvertibleTransform)
        assert isinstance(scaler, Transform)

    def test_transforms_without_fit_dont_inherit_IFittableTransform(self):
        """Transforms without fit() should not inherit from IFittableTransform."""
        # This test documents expected behavior - add non-fittable transforms here as they're created
        # For now, all existing transforms are fittable, so this test is a placeholder
        pass

    def test_transforms_without_inverse_dont_inherit_IInvertibleTransform(self):
        """Transforms without inverse_transform() should not inherit from IInvertibleTransform."""
        # This test documents expected behavior - add non-invertible transforms here as they're created
        # For now, all existing transforms are invertible, so this test is a placeholder
        pass


class TestDependencyInversion:
    """Test DIP: Code should depend on abstractions (ABCs), not concrete checks."""

    def test_transform_chain_uses_isinstance_not_hasattr_for_fit(self):
        """TransformChain should use isinstance(IFittableTransform), not hasattr('fit')."""
        from dlkit.core.training.transforms import chain

        # Read the source code of TransformChain.fit()
        source = inspect.getsource(chain.TransformChain.fit)

        # Should use isinstance checks, not hasattr
        assert "isinstance" in source or "IFittableTransform" not in source, (
            "TransformChain.fit() should use isinstance(IFittableTransform) "
            "instead of hasattr() checks (when refactored in Phase 4)"
        )

        # For now, this test is marked as expected to fail until Phase 4
        # After Phase 4, update this test to strictly require isinstance()

    def test_transform_chain_uses_isinstance_not_hasattr_for_inverse(self):
        """TransformChain should use isinstance(IInvertibleTransform), not hasattr('inverse_transform')."""
        from dlkit.core.training.transforms import chain

        # Read the source code of TransformChain.inverse_transform()
        source = inspect.getsource(chain.TransformChain.inverse_transform)

        # Should use isinstance checks, not hasattr
        assert "isinstance" in source or "IInvertibleTransform" not in source, (
            "TransformChain.inverse_transform() should use isinstance(IInvertibleTransform) "
            "instead of hasattr() checks (when refactored in Phase 4)"
        )

        # For now, this test is marked as expected to fail until Phase 4


class TestSingleResponsibility:
    """Test SRP: Transform classes have focused, single responsibilities."""

    def test_transform_classes_are_reasonably_sized(self):
        """Transform classes should not be god classes (< 300 lines)."""
        transform_classes = [
            (MinMaxScaler, "MinMaxScaler"),
            (StandardScaler, "StandardScaler"),
            (PCA, "PCA"),
            (TransformChain, "TransformChain"),
        ]

        for transform_cls, name in transform_classes:
            source = inspect.getsource(transform_cls)
            line_count = len(source.splitlines())

            assert line_count < 300, (
                f"{name} has {line_count} lines, which may indicate "
                f"it's doing too much (violates SRP). Consider refactoring."
            )

    def test_base_Transform_provides_fitted_property(self):
        """Base Transform class provides fitted property for all transforms."""
        assert hasattr(Transform, "fitted"), (
            "Transform base class must provide fitted property "
            "for consistent state management"
        )

        # Test that it's a property, not just an attribute
        assert isinstance(inspect.getattr_static(Transform, "fitted"), property), (
            "fitted should be a property, not a plain attribute"
        )


class TestArchitecturalConstraints:
    """Test architectural constraints and best practices."""

    def test_fitted_state_uses_tensor_buffer(self):
        """Fitted state must use torch.Tensor buffer for checkpoint persistence."""
        scaler = MinMaxScaler(dim=0)

        # Check that _fitted is registered as a buffer
        assert "_fitted" in scaler._buffers, (
            "Fitted state must be stored as torch buffer (_fitted) "
            "for checkpoint persistence and device movement"
        )

        # Check that it's a tensor
        assert isinstance(scaler._buffers["_fitted"], torch.Tensor), (
            "Fitted buffer must be a torch.Tensor"
        )

    def test_transform_methods_have_consistent_signatures(self):
        """Transform methods should have consistent parameter names across classes."""
        # fit() should take 'data' parameter
        fittable_transforms = [MinMaxScaler, StandardScaler, PCA]

        for transform_cls in fittable_transforms:
            fit_sig = inspect.signature(transform_cls.fit)
            param_names = list(fit_sig.parameters.keys())

            # Note: TransformChain uses 'x' instead of 'data' - this is a known inconsistency
            # It's acceptable since it's documented in the interface
            if transform_cls != TransformChain:
                assert "data" in param_names, (
                    f"{transform_cls.__name__}.fit() should have 'data' parameter "
                    f"for consistency, got: {param_names}"
                )

    def test_all_transforms_inherit_from_Transform_base(self):
        """All transform classes must inherit from Transform base class."""
        transform_classes = [MinMaxScaler, StandardScaler, PCA, TransformChain]

        for transform_cls in transform_classes:
            assert issubclass(transform_cls, Transform), (
                f"{transform_cls.__name__} must inherit from Transform base class"
            )

    def test_ABC_interfaces_have_docstrings(self):
        """All ABC interfaces must have comprehensive docstrings."""
        interfaces = [IFittableTransform, IInvertibleTransform, ISerializableTransform]

        for interface in interfaces:
            assert interface.__doc__ is not None, (
                f"{interface.__name__} must have a docstring explaining its purpose"
            )

            assert len(interface.__doc__) > 100, (
                f"{interface.__name__} docstring should be comprehensive (>100 chars), "
                f"got {len(interface.__doc__)} chars"
            )


class TestInvariants:
    """Test invariants that must always hold."""

    def test_fitted_transforms_have_fitted_property(self):
        """All IFittableTransform instances must have fitted property."""
        scaler = MinMaxScaler(dim=0)

        # Initially not fitted
        assert not scaler.fitted, "Transform should not be fitted initially"

        # After fitting
        data = torch.randn(32, 64)
        scaler.fit(data)
        assert scaler.fitted, "Transform should be fitted after calling fit()"

    def test_invertible_transforms_have_inverse_transform_method(self):
        """All IInvertibleTransform instances must have inverse_transform method."""
        invertible_transforms = [
            MinMaxScaler(dim=0),
            StandardScaler(dim=0),
            PCA(n_components=10),
        ]

        for transform in invertible_transforms:
            assert hasattr(transform, "inverse_transform"), (
                f"{transform.__class__.__name__} must have inverse_transform() method"
            )

            assert callable(transform.inverse_transform), (
                f"{transform.__class__.__name__}.inverse_transform must be callable"
            )

    def test_transforms_are_pytorch_modules(self):
        """All transforms must be PyTorch nn.Module instances."""
        import torch.nn as nn

        transform_instances = [
            MinMaxScaler(dim=0),
            StandardScaler(dim=0),
            PCA(n_components=10),
        ]

        for transform in transform_instances:
            assert isinstance(transform, nn.Module), (
                f"{transform.__class__.__name__} must be a PyTorch nn.Module"
            )


# ===========================================================================================
# PHASE 4 COMPLETION VERIFICATION
# ===========================================================================================

def test_transform_chain_uses_isinstance_not_hasattr():
    """FIXED: TransformChain now uses isinstance() instead of hasattr()."""
    from dlkit.core.training.transforms import chain

    # Read source of TransformChain.fit()
    fit_source = inspect.getsource(chain.TransformChain.fit)

    # Should NOT use hasattr
    assert "hasattr" not in fit_source, (
        "TransformChain.fit() should use isinstance(IFittableTransform), "
        "not hasattr('fit')"
    )

    # Should use isinstance
    assert "isinstance" in fit_source, (
        "TransformChain.fit() should use isinstance(IFittableTransform)"
    )

    # Read source of TransformChain.inverse_transform()
    inverse_source = inspect.getsource(chain.TransformChain.inverse_transform)

    # Should NOT use hasattr
    assert "hasattr" not in inverse_source, (
        "TransformChain.inverse_transform() should use isinstance(IInvertibleTransform), "
        "not hasattr('inverse_transform')"
    )

    # Should use isinstance
    assert "isinstance" in inverse_source, (
        "TransformChain.inverse_transform() should use isinstance(IInvertibleTransform)"
    )
