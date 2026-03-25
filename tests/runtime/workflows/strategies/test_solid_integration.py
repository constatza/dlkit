"""Integration tests verifying SOLID principles work together."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest

from dlkit.interfaces.api.domain import TrainingResult

# OptimizationDecorator removed - use clean architecture tests instead
from dlkit.runtime.workflows.factories.build_factory import BuildComponents
from dlkit.runtime.workflows.strategies.core import ITrainingExecutor, VanillaExecutor
from dlkit.runtime.workflows.strategies.factory import ExecutionStrategyFactory
from dlkit.runtime.workflows.strategies.tracking import TrackingDecorator
from dlkit.tools.config.general_settings import GeneralSettings
from dlkit.tools.config.mlflow_settings import MLflowSettings
from dlkit.tools.config.optuna_settings import OptunaSettings


@pytest.fixture
def build_components():
    """Create realistic BuildComponents for integration testing."""

    @dataclass(frozen=True, slots=True)
    class TestModel:
        pass

    class TestTrainer:
        def __init__(self):
            self.callbacks = []
            self.logged_metrics = {"val_loss": 0.45, "accuracy": 0.92}
            self.called = {"fit": 0, "predict": 0, "test": 0}

        def fit(self, model, datamodule=None, **kwargs):
            """Match PyTorch Lightning Trainer.fit() signature."""
            self.called["fit"] += 1

        def predict(self, model, datamodule=None, **kwargs):
            """Match PyTorch Lightning Trainer.predict() signature."""
            self.called["predict"] += 1

        def test(self, model, datamodule=None, **kwargs):
            """Match PyTorch Lightning Trainer.test() signature."""
            self.called["test"] += 1

    return BuildComponents(
        model=cast("Any", TestModel()),
        datamodule=Mock(),
        trainer=cast("Any", TestTrainer()),
        shape_spec=None,
        meta={},
    )


def test_single_responsibility_principle_integration(build_components, monkeypatch):
    """Test that each component has a single, focused responsibility."""
    # Ensure no local MLflow server is detected and no env var set
    from dlkit.runtime.workflows.strategies.tracking import uri_resolver

    monkeypatch.setattr(uri_resolver, "local_host_alive", lambda: False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    factory = ExecutionStrategyFactory()

    # Factory creates TrackingDecorator with NullTracker when no features enabled
    vanilla_settings = GeneralSettings()
    vanilla_executor = factory.create_executor(vanilla_settings)

    # Factory now always returns TrackingDecorator (null object pattern)
    assert isinstance(vanilla_executor, TrackingDecorator)
    assert isinstance(vanilla_executor._executor, VanillaExecutor)

    result = vanilla_executor.execute(build_components, vanilla_settings)
    assert isinstance(result, TrainingResult)
    assert result.metrics["val_loss"] == 0.45


def test_open_closed_principle_integration(build_components):
    """Test that new behaviors can be added without modifying existing code."""
    factory = ExecutionStrategyFactory()

    # Can add MLflow tracking without modifying VanillaExecutor
    mlflow_settings = GeneralSettings(
        MLFLOW=MLflowSettings(
            experiment_name="integration_test",
        )
    )

    with patch(
        "dlkit.runtime.workflows.strategies.tracking.mlflow_tracker.MLflowTracker"
    ) as mock_tracker_class:
        mock_tracker = Mock()
        mock_tracker_class.return_value = mock_tracker

        # Mock the tracker context manager
        mock_tracker.__enter__ = Mock(return_value=mock_tracker)
        mock_tracker.__exit__ = Mock(return_value=None)

        # Mock the create_run context manager
        mock_run_context = Mock()
        mock_tracker.create_run.return_value.__enter__ = Mock(return_value=mock_run_context)
        mock_tracker.create_run.return_value.__exit__ = Mock(return_value=None)
        mock_tracker.configure = Mock()
        mock_tracker.get_tracking_uri = Mock(return_value=None)
        mock_tracker.is_local = Mock(return_value=False)

        executor = factory.create_executor(mlflow_settings)

        assert isinstance(executor, TrackingDecorator)
        result = executor.execute(build_components, mlflow_settings)

        # Core functionality preserved
        assert isinstance(result, TrainingResult)
        assert build_components.trainer.called["fit"] == 1


def test_liskov_substitution_principle_integration(build_components):
    """Test that all executors can be substituted for the base interface."""
    factory = ExecutionStrategyFactory()

    # Create different executor configurations
    vanilla_settings = GeneralSettings()
    mlflow_settings = GeneralSettings(MLFLOW=MLflowSettings())
    optuna_settings = GeneralSettings(OPTUNA=OptunaSettings(enabled=True, n_trials=2))
    both_settings = GeneralSettings(
        MLFLOW=MLflowSettings(),
        OPTUNA=OptunaSettings(enabled=True, n_trials=2),
    )

    executors = [
        factory.create_executor(vanilla_settings),
        factory.create_executor(mlflow_settings),
        factory.create_executor(optuna_settings),
        factory.create_executor(both_settings),
    ]

    # All should be substitutable for ITrainingExecutor
    for executor in executors:
        assert isinstance(executor, ITrainingExecutor)
        # All should have the same interface
        assert hasattr(executor, "execute")
        assert callable(executor.execute)


def test_interface_segregation_principle_integration():
    """Test that interfaces are focused and not bloated."""
    from dlkit.runtime.workflows.strategies.core.interfaces import ITrainingExecutor
    from dlkit.runtime.workflows.strategies.optuna.interfaces import IHyperparameterOptimizer
    from dlkit.runtime.workflows.strategies.tracking.interfaces import IExperimentTracker

    # ITrainingExecutor should only have execute method
    training_methods = [method for method in dir(ITrainingExecutor) if not method.startswith("_")]
    assert training_methods == ["execute"]

    # IExperimentTracker should only have tracking methods
    tracker_methods = [method for method in dir(IExperimentTracker) if not method.startswith("_")]
    assert "create_run" in tracker_methods
    assert "log_settings" in tracker_methods

    # IHyperparameterOptimizer should only have optimization methods
    optimizer_methods = [
        method for method in dir(IHyperparameterOptimizer) if not method.startswith("_")
    ]
    assert "optimize" in optimizer_methods
    assert "create_sampled_settings" in optimizer_methods


def test_dependency_inversion_principle_integration(build_components):
    """Test that high-level modules depend on abstractions, not concretions."""
    # TrackingDecorator depends on IExperimentTracker abstraction
    # Create mock tracker without spec to allow magic method mocking
    from unittest.mock import MagicMock

    mock_tracker = MagicMock()
    # Explicitly set context manager support
    mock_tracker.__enter__.return_value = mock_tracker
    mock_tracker.__exit__.return_value = None
    # Mock setup_mlflow_config to return tracking URI and status placeholder
    mock_tracker.setup_mlflow_config.return_value = (None, None)
    mock_run_context = MagicMock()
    mock_tracker.create_run.return_value.__enter__.return_value = mock_run_context
    mock_tracker.create_run.return_value.__exit__.return_value = None

    vanilla_executor = VanillaExecutor()
    tracking_decorator = TrackingDecorator(vanilla_executor, mock_tracker)

    # Should be able to inject any implementation
    settings = GeneralSettings(MLFLOW=MLflowSettings())

    result = tracking_decorator.execute(build_components, settings)
    assert isinstance(result, TrainingResult)

    # Mock tracker should have been used
    mock_tracker.create_run.assert_called()


def test_pure_solid_architecture_integration(build_components):
    """Test that pure SOLID architecture works without backward compatibility."""
    from dlkit.runtime.workflows.strategies import ExecutionStrategyFactory, VanillaExecutor

    settings = GeneralSettings()

    # Pure SOLID interface
    vanilla_executor = VanillaExecutor()
    result = vanilla_executor.execute(build_components, settings)  # New method name

    assert isinstance(result, TrainingResult)
    assert build_components.trainer.called["fit"] == 1

    # Factory-created executor should also work
    factory = ExecutionStrategyFactory()
    factory_executor = factory.create_executor(settings)
    result2 = factory_executor.execute(build_components, settings)
    assert isinstance(result2, TrainingResult)


def test_error_handling_integration(build_components):
    """Test that error handling works correctly across the architecture."""
    from dlkit.interfaces.api.domain import WorkflowError

    # Create failing trainer
    class FailingTrainer:
        def fit(self, *args, **kwargs):
            raise RuntimeError("Training failed")

    failing_components = BuildComponents(
        model=cast("Any", Mock()),
        datamodule=Mock(),
        trainer=cast("Any", FailingTrainer()),
        shape_spec=None,
        meta={},
    )

    factory = ExecutionStrategyFactory()
    executor = factory.create_executor(GeneralSettings())

    # Error should be properly wrapped
    with pytest.raises(WorkflowError) as exc_info:
        executor.execute(failing_components, GeneralSettings())

    assert "Vanilla execution failed" in str(exc_info.value.message)
    assert "Training failed" in str(exc_info.value.message)


def test_end_to_end_solid_workflow():
    """Test complete end-to-end workflow demonstrating all SOLID principles."""
    # This test demonstrates how all principles work together
    factory = ExecutionStrategyFactory()

    # Settings with all features enabled
    full_settings = GeneralSettings(
        MLFLOW=MLflowSettings(),
        OPTUNA=OptunaSettings(enabled=True, n_trials=2),
    )

    # Factory's create_executor only creates TrackingDecorator
    executor = factory.create_executor(full_settings)

    # create_executor returns TrackingDecorator regardless of Optuna settings
    # OptimizationDecorator is handled by create_optimization_strategy
    assert isinstance(executor, TrackingDecorator)  # Outer
    assert isinstance(executor._executor, VanillaExecutor)  # Core

    # Each component has single responsibility (SRP)
    # All components can be substituted (LSP)
    # Interfaces are focused (ISP)
    # Dependencies are inverted (DIP)

    assert True
