"""Architecture tests for optimization lifecycle ownership contracts."""

from __future__ import annotations

import inspect
from contextlib import AbstractContextManager

from dlkit.engine.workflows.optimization.infrastructure import (
    MLflowTrackingAdapter,
    NullTrackingAdapter,
)
from dlkit.engine.workflows.optimization.value_objects import IExperimentTracker, IStudyRepository


def _load_backend_session_protocol():
    from dlkit.engine.workflows.optimization.value_objects import IOptimizationBackendSession

    return IOptimizationBackendSession


class TestExperimentTrackerContract:
    """Verify experiment trackers keep their explicit context-manager contract."""

    def test_experiment_tracker_protocol_requires_context_manager(self):
        """IExperimentTracker protocol requires AbstractContextManager."""
        assert issubclass(IExperimentTracker, AbstractContextManager), (
            "IExperimentTracker must inherit from AbstractContextManager to keep "
            "tracking setup and cleanup explicit."
        )

    def test_tracking_adapters_are_context_managers(self):
        """Concrete tracking adapters implement the tracker context contract."""
        assert issubclass(MLflowTrackingAdapter, AbstractContextManager)
        assert issubclass(NullTrackingAdapter, AbstractContextManager)
        assert hasattr(MLflowTrackingAdapter, "__enter__")
        assert hasattr(MLflowTrackingAdapter, "__exit__")
        assert hasattr(NullTrackingAdapter, "__enter__")
        assert hasattr(NullTrackingAdapter, "__exit__")


class TestOptimizationBackendSessionContract:
    """Verify optimization backends expose an explicit context-manager seam."""

    def test_backend_session_protocol_requires_context_manager(self):
        """IOptimizationBackendSession must be a context manager abstraction."""
        backend_session = _load_backend_session_protocol()

        assert issubclass(backend_session, AbstractContextManager), (
            "IOptimizationBackendSession must inherit from AbstractContextManager so "
            "orchestrators can own backend resource lifecycle explicitly."
        )

    def test_backend_session_docstring_makes_lifecycle_explicit(self):
        """The backend session contract documents setup and cleanup ownership."""
        backend_session = _load_backend_session_protocol()
        doc = backend_session.__doc__ or ""

        assert any(
            keyword in doc.lower()
            for keyword in ("context", "session", "resource", "lifecycle", "enter", "exit")
        ), (
            "IOptimizationBackendSession should document that it owns backend setup "
            "and cleanup responsibilities."
        )

    def test_backend_session_public_methods_are_documented(self):
        """All backend session protocol methods should explain the contract."""
        backend_session = _load_backend_session_protocol()

        for name, method in inspect.getmembers(backend_session, inspect.isfunction):
            if name.startswith("_"):
                continue
            assert method.__doc__, (
                f"IOptimizationBackendSession.{name} must be documented so the "
                "backend boundary stays explicit for implementers."
            )


class TestRepositoryAbstraction:
    """Verify the study repository stays backend-agnostic."""

    def test_study_repository_has_no_backend_specific_methods(self):
        """IStudyRepository should not leak Optuna or other backend-specific APIs."""
        method_names = {
            name
            for name, member in inspect.getmembers(IStudyRepository, inspect.isfunction)
            if not name.startswith("_")
        }

        assert "get_optuna_study" not in method_names, (
            "IStudyRepository must stay backend-agnostic. Backend session "
            "operations belong on IOptimizationBackendSession instead."
        )
        assert not any("optuna" in name.lower() for name in method_names), (
            "IStudyRepository should not expose backend-branded methods."
        )
