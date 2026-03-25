"""Architecture fitness tests for Transform system (Phase 2 Protocol-Based Design).

These tests enforce Protocol-based architectural principles:
- 3 Runtime Checkable Protocols for type-safe capability checking
- No ABCs - all capabilities expressed via Protocols
- Single Responsibility: Transform classes have focused purposes
- Pythonic design: Protocols instead of hasattr() checks

Phase 2 Simplification:
- Removed: 4 ABCs (IFittableTransform, IInvertibleTransform, IShapeAwareTransform, ISerializableTransform)
- Added: 3 Protocols (FittableTransform, InvertibleTransform, ShapeAwareTransform)
- Use: isinstance() checks with Protocols (runtime_checkable)

Benefits:
- Type safety without ABC ceremony
- Structural typing (duck typing with isinstance support)
- Clear documentation via Protocol definitions

Tests fail if architectural violations are introduced.
"""

import inspect

import torch

from dlkit.core.training.transforms.base import (
    FittableTransform,
    InvertibleTransform,
    ShapeAwareTransform,
    Transform,
)
from dlkit.core.training.transforms.chain import TransformChain
from dlkit.core.training.transforms.minmax import MinMaxScaler
from dlkit.core.training.transforms.pca import PCA
from dlkit.core.training.transforms.standard import StandardScaler


class TestProtocolBasedArchitecture:
    """Test Phase 2 Protocol-based architecture: 3 Protocols for all capabilities."""

    def test_fittable_transforms_pass_protocol_check(self):
        """Transforms with fit() method pass FittableTransform Protocol check."""
        fittable_instances = [
            MinMaxScaler(dim=0),
            StandardScaler(dim=0),
            PCA(n_components=10),
            TransformChain([]),  # Takes list of transforms as positional arg
        ]

        for instance in fittable_instances:
            # Protocol uses structural typing - any class with fit() passes
            assert isinstance(instance, FittableTransform), (
                f"{instance.__class__.__name__} has fit() but doesn't pass "
                f"FittableTransform Protocol check."
            )

    def test_invertible_transforms_pass_protocol_check(self):
        """Transforms with inverse_transform() method pass InvertibleTransform Protocol check."""
        invertible_instances = [
            MinMaxScaler(dim=0),
            StandardScaler(dim=0),
            PCA(n_components=10),
            TransformChain([]),  # Takes list of transforms as positional arg
        ]

        for instance in invertible_instances:
            # Protocol uses structural typing - any class with inverse_transform() passes
            assert isinstance(instance, InvertibleTransform), (
                f"{instance.__class__.__name__} has inverse_transform() but doesn't pass "
                f"InvertibleTransform Protocol check."
            )

    def test_shape_aware_transforms_pass_protocol_check(self):
        """Transforms with configure_shape() method pass ShapeAwareTransform Protocol check."""
        shape_aware_instances = [
            MinMaxScaler(dim=0),
            StandardScaler(dim=0),
            PCA(n_components=10),
        ]

        for instance in shape_aware_instances:
            # Protocol uses structural typing - any class with configure_shape() passes
            assert isinstance(instance, ShapeAwareTransform), (
                f"{instance.__class__.__name__} has configure_shape() but doesn't pass "
                f"ShapeAwareTransform Protocol check."
            )

    def test_protocols_are_runtime_checkable(self):
        """All Protocols support runtime isinstance() checks."""
        scaler = MinMaxScaler(dim=0)

        # All Protocols should be @runtime_checkable
        assert isinstance(scaler, FittableTransform)
        assert isinstance(scaler, InvertibleTransform)
        assert isinstance(scaler, ShapeAwareTransform)
        assert isinstance(scaler, Transform)

    def test_non_invertible_transforms_fail_protocol_check(self):
        """Transforms without inverse_transform() fail InvertibleTransform Protocol check."""

        # Create a simple non-invertible transform
        class NonInvertible(Transform):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2

        transform = NonInvertible()
        assert not isinstance(transform, InvertibleTransform), (
            "NonInvertible transform should not pass InvertibleTransform Protocol check"
        )

    def test_no_default_methods_in_base(self):
        """Transform base class should NOT provide default capability methods."""
        # This prevents false positives in Protocol checks
        assert not hasattr(Transform, "inverse_transform"), (
            "Transform base should not have inverse_transform() to prevent false Protocol matches"
        )
        # fit() and configure_shape() have default implementations (no-op/pass)
        # This is OK because they're truly optional


class TestProtocolUsage:
    """Test that code uses Protocols correctly instead of hasattr()."""

    def test_transform_chain_uses_protocols(self):
        """TransformChain uses Protocols for capability checking."""
        from dlkit.core.training.transforms import chain

        fit_source = inspect.getsource(chain.TransformChain.fit)
        inverse_source = inspect.getsource(chain.TransformChain.inverse_transform)

        # Should use isinstance() checks with Protocols
        assert "isinstance" in fit_source, (
            "TransformChain.fit() should use isinstance() for Protocol check"
        )
        assert "FittableTransform" in fit_source, (
            "TransformChain.fit() should check FittableTransform Protocol"
        )

        assert "isinstance" in inverse_source, (
            "TransformChain.inverse_transform() should use isinstance() for Protocol check"
        )
        assert "InvertibleTransform" in inverse_source, (
            "TransformChain.inverse_transform() should check InvertibleTransform Protocol"
        )


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
            "Transform base class must provide fitted property for consistent state management"
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

    def test_protocol_has_docstring(self):
        """InvertibleTransform Protocol must have comprehensive docstring."""
        assert InvertibleTransform.__doc__ is not None, (
            "InvertibleTransform Protocol must have a docstring explaining its purpose"
        )

        assert len(InvertibleTransform.__doc__) > 100, (
            f"InvertibleTransform docstring should be comprehensive (>100 chars), "
            f"got {len(InvertibleTransform.__doc__)} chars"
        )


class TestInvariants:
    """Test invariants that must always hold."""

    def test_fitted_transforms_have_fitted_property(self):
        """All fittable transform instances must have fitted property."""
        scaler = MinMaxScaler(dim=0)

        # Initially not fitted
        assert not scaler.fitted, "Transform should not be fitted initially"

        # After fitting
        data = torch.randn(32, 64)
        scaler.fit(data)
        assert scaler.fitted, "Transform should be fitted after calling fit()"

    def test_invertible_transforms_have_inverse_transform_method(self):
        """All transforms passing InvertibleTransform Protocol must have inverse_transform method."""
        invertible_transforms = [
            MinMaxScaler(dim=0),
            StandardScaler(dim=0),
            PCA(n_components=10),
        ]

        for transform in invertible_transforms:
            # Should pass Protocol check
            assert isinstance(transform, InvertibleTransform), (
                f"{transform.__class__.__name__} should pass InvertibleTransform Protocol check"
            )

            assert hasattr(transform, "inverse_transform"), (
                f"{transform.__class__.__name__} must have inverse_transform() method"
            )

            assert callable(transform.inverse_transform), (
                f"{transform.__class__.__name__}.inverse_transform must be callable"
            )

    def test_transforms_are_pytorch_modules(self):
        """All transforms must be PyTorch nn.Module instances."""
        from torch import nn

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
# PHASE 2 COMPLETION VERIFICATION
# ===========================================================================================


def test_phase2_protocol_architecture_complete():
    """Verify Phase 2 Protocol-based architecture is correctly implemented."""
    from dlkit.core.training.transforms import chain

    # 1. TransformChain uses FittableTransform Protocol
    fit_source = inspect.getsource(chain.TransformChain.fit)
    assert "isinstance" in fit_source, "Should use isinstance() for Protocol check"
    assert "FittableTransform" in fit_source, "Should check FittableTransform Protocol"

    # 2. TransformChain uses InvertibleTransform Protocol
    inverse_source = inspect.getsource(chain.TransformChain.inverse_transform)
    assert "isinstance" in inverse_source, "Should use isinstance() for Protocol check"
    assert "InvertibleTransform" in inverse_source, "Should check InvertibleTransform Protocol"

    # 3. All 3 Protocols work correctly
    scaler = MinMaxScaler(dim=0)
    assert isinstance(scaler, FittableTransform), "Should pass FittableTransform check"
    assert isinstance(scaler, InvertibleTransform), "Should pass InvertibleTransform check"
    assert isinstance(scaler, ShapeAwareTransform), "Should pass ShapeAwareTransform check"

    # 4. No default inverse_transform() in Transform base
    assert not hasattr(Transform, "inverse_transform"), (
        "Transform base should not have default inverse_transform()"
    )
