"""Architecture tests for context manager lifecycle compliance.

These tests verify that the experiment tracker architecture follows SOLID
principles and properly implements context manager protocols.
"""

from contextlib import AbstractContextManager

from dlkit.runtime.workflows.optimization.domain import IExperimentTracker
from dlkit.runtime.workflows.optimization.infrastructure import (
    MLflowTrackingAdapter,
    NullTrackingAdapter,
)


class TestContextManagerCompliance:
    """Verify all trackers implement AbstractContextManager protocol."""

    def test_experiment_tracker_protocol_requires_context_manager(self):
        """IExperimentTracker protocol requires AbstractContextManager."""
        # Verify protocol inheritance
        assert issubclass(IExperimentTracker, AbstractContextManager), (
            "IExperimentTracker must inherit from AbstractContextManager to ensure "
            "all implementations provide proper context management"
        )

    def test_mlflow_adapter_is_context_manager(self):
        """MLflowTrackingAdapter implements AbstractContextManager."""
        assert issubclass(MLflowTrackingAdapter, AbstractContextManager), (
            "MLflowTrackingAdapter must implement AbstractContextManager for "
            "proper resource lifecycle management"
        )

    def test_null_adapter_is_context_manager(self):
        """NullTrackingAdapter implements AbstractContextManager."""
        assert issubclass(NullTrackingAdapter, AbstractContextManager), (
            "NullTrackingAdapter must implement AbstractContextManager to provide "
            "uniform interface with MLflowTrackingAdapter (Null Object Pattern)"
        )

    def test_mlflow_adapter_has_enter_exit(self):
        """MLflowTrackingAdapter implements __enter__ and __exit__ methods."""
        assert hasattr(MLflowTrackingAdapter, "__enter__"), (
            "MLflowTrackingAdapter must implement __enter__ for context management"
        )
        assert hasattr(MLflowTrackingAdapter, "__exit__"), (
            "MLflowTrackingAdapter must implement __exit__ for resource cleanup"
        )

    def test_null_adapter_has_enter_exit(self):
        """NullTrackingAdapter implements __enter__ and __exit__ methods."""
        assert hasattr(NullTrackingAdapter, "__enter__"), (
            "NullTrackingAdapter must implement __enter__ for uniform interface"
        )
        assert hasattr(NullTrackingAdapter, "__exit__"), (
            "NullTrackingAdapter must implement __exit__ for uniform interface"
        )


class TestNullObjectPattern:
    """Verify NullTrackingAdapter follows Null Object Pattern correctly."""

    def test_null_adapter_context_manager_is_no_op(self):
        """NullTrackingAdapter context management is a no-op."""
        adapter = NullTrackingAdapter()

        # Should not raise any errors
        with adapter:
            pass

        # Should be idempotent - can enter multiple times
        with adapter:
            pass

    def test_null_adapter_returns_self_from_enter(self):
        """NullTrackingAdapter.__enter__ returns self."""
        adapter = NullTrackingAdapter()
        result = adapter.__enter__()
        assert result is adapter, "__enter__ should return self for context manager protocol"

    def test_null_adapter_exit_returns_false(self):
        """NullTrackingAdapter.__exit__ returns False (doesn't suppress exceptions)."""
        adapter = NullTrackingAdapter()
        adapter.__enter__()
        result = adapter.__exit__(None, None, None)
        assert result is False, "__exit__ should return False to propagate exceptions"


class TestSingleResponsibility:
    """Verify context lifecycle has single owner (Service layer)."""

    def test_optimization_strategy_does_not_manage_context(self):
        """OptimizationStrategy does not contain context management logic."""
        import inspect

        from dlkit.runtime.workflows.optimization.strategy import OptimizationStrategy

        source = inspect.getsource(OptimizationStrategy.execute_optimization)

        # Should not have conditional context entry
        assert "hasattr" not in source or "__enter__" not in source, (
            "Strategy should not check for context manager support at runtime. "
            "Context management is the responsibility of the service layer."
        )

        # Should not enter contexts
        assert "with orchestrator._experiment_tracker" not in source, (
            "Strategy should not manage experiment tracker context. "
            "This violates Single Responsibility Principle - context lifecycle "
            "is the responsibility of the service layer."
        )

    def test_optimization_service_manages_context(self):
        """OptimizationService is the single owner of tracker context lifecycle."""
        import inspect

        from dlkit.interfaces.api.services.optimization_service import OptimizationService

        source = inspect.getsource(OptimizationService.execute_optimization)

        # Service should enter context
        assert "with experiment_tracker" in source, (
            "Service must enter experiment tracker context to ensure proper "
            "resource lifecycle management. Service layer owns context lifecycle."
        )


class TestDependencyInversion:
    """Verify factory returns uninitialized trackers (service manages lifecycle)."""

    def test_factory_does_not_enter_context(self):
        """OptimizationServiceFactory returns uninitialized trackers."""
        import inspect

        from dlkit.runtime.workflows.optimization.factory import OptimizationServiceFactory

        # Check _create_experiment_tracker method
        source = inspect.getsource(OptimizationServiceFactory._create_experiment_tracker)

        # Factory should NOT enter contexts
        assert "with" not in source or "__enter__" not in source, (
            "Factory should create uninitialized tracker instances. "
            "Context entry is the responsibility of the service layer, not the factory. "
            "This follows Dependency Inversion Principle - factory creates, service manages."
        )


class TestInterfaceSegregation:
    """Verify IExperimentTracker protocol is properly segregated."""

    def test_protocol_defines_context_manager_contract(self):
        """IExperimentTracker explicitly declares context manager requirement."""

        # Get the protocol's docstring
        doc = IExperimentTracker.__doc__ or ""

        # Should document context manager requirement
        assert any(
            keyword in doc.lower() for keyword in ["context", "enter", "exit", "resource"]
        ), (
            "IExperimentTracker protocol must document its context manager requirement "
            "to make the contract explicit for implementers"
        )

    def test_protocol_methods_documented(self):
        """All IExperimentTracker protocol methods are documented."""
        import inspect

        # Check that abstract methods have docstrings
        for name, method in inspect.getmembers(IExperimentTracker, inspect.isfunction):
            if name.startswith("_"):
                continue
            assert method.__doc__ is not None, (
                f"Protocol method {name} must be documented to clarify contract"
            )
