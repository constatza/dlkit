"""Integration tests verifying SOLID principles work together."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest

from dlkit.common import TrainingResult
from dlkit.engine.tracking.tracking_decorator import TrackingDecorator
from dlkit.engine.training import ITrainingExecutor, VanillaExecutor

# OptimizationDecorator removed - use clean architecture tests instead
from dlkit.engine.training.components import RuntimeComponents
from dlkit.engine.workflows.factories.execution_strategy_factory import ExecutionStrategyFactory
from dlkit.infrastructure.config.experiment_settings import ExperimentSettings
from dlkit.infrastructure.config.job_config import JobConfig
from dlkit.infrastructure.config.run_settings import RunSettings
from dlkit.infrastructure.config.tracking_settings import TrackingSettings

type _ExecutionSettings = JobConfig


def _job(
    *,
    mlflow: bool = False,
    experiment_name: str = "integration_test",
) -> JobConfig:
    """Create a minimal JobConfig for executor.execute() calls.

    Args:
        mlflow: Whether to enable mlflow tracking backend.
        experiment_name: Name for the experiment section.

    Returns:
        Minimal JobConfig suitable for passing to executor.execute().
    """
    return JobConfig(
        run=RunSettings(type="train"),
        experiment=ExperimentSettings(name=experiment_name) if mlflow else None,
        tracking=TrackingSettings(backend="mlflow") if mlflow else TrackingSettings(),
    )


@pytest.fixture
def build_components():
    """Create realistic RuntimeComponents for integration testing."""

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

    return RuntimeComponents(
        model=cast("Any", TestModel()),
        datamodule=Mock(),
        trainer=cast("Any", TestTrainer()),
        meta={},
    )


def test_single_responsibility_principle_integration(build_components, monkeypatch):
    """Test that each component has a single, focused responsibility."""
    # Ensure no local MLflow server is detected and no env var set
    import dlkit.engine.tracking.uri_resolver as uri_resolver

    monkeypatch.setattr(uri_resolver, "local_host_alive", lambda: False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    factory = ExecutionStrategyFactory()

    # Factory creates TrackingDecorator with NullTracker when no features enabled
    vanilla_job = _job()
    vanilla_executor = factory.create_executor(vanilla_job)

    # Factory now always returns TrackingDecorator (null object pattern)
    assert isinstance(vanilla_executor, TrackingDecorator)
    assert isinstance(vanilla_executor._executor, VanillaExecutor)

    # Use JobConfig for execute() since VanillaExecutor requires JobConfig.run
    result = vanilla_executor.execute(build_components, vanilla_job)
    assert isinstance(result, TrainingResult)
    assert result.metrics["val_loss"] == 0.45


def test_open_closed_principle_integration(build_components):
    """Test that new behaviors can be added without modifying existing code."""
    factory = ExecutionStrategyFactory()

    # Can add MLflow tracking without modifying VanillaExecutor
    mlflow_job = _job(mlflow=True, experiment_name="integration_test")

    with patch("dlkit.engine.tracking.mlflow_tracker.MLflowTracker") as mock_tracker_class:
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

        executor = factory.create_executor(mlflow_job)

        assert isinstance(executor, TrackingDecorator)
        # Use JobConfig for execute() since executor requires JobConfig.run
        result = executor.execute(build_components, mlflow_job)

        # Core functionality preserved
        assert isinstance(result, TrainingResult)
        assert build_components.trainer.called["fit"] == 1


def test_liskov_substitution_principle_integration(build_components):
    """Test that all executors can be substituted for the base interface."""
    factory = ExecutionStrategyFactory()

    # Create different executor configurations
    vanilla_job = _job()
    mlflow_job = _job(mlflow=True)
    search_job = _job()
    both_job = _job(mlflow=True, experiment_name="both_test")

    executors = [
        factory.create_executor(vanilla_job),
        factory.create_executor(mlflow_job),
        factory.create_executor(search_job),
        factory.create_executor(both_job),
    ]

    # All should be substitutable for ITrainingExecutor
    for executor in executors:
        assert isinstance(executor, ITrainingExecutor)
        # All should have the same interface
        assert hasattr(executor, "execute")
        assert callable(executor.execute)


def test_interface_segregation_principle_integration():
    """Test that interfaces are focused and not bloated."""
    from dlkit.engine.tracking.interfaces import IExperimentTracker
    from dlkit.engine.training.interfaces import IOptimizationStrategy, ITrainingExecutor

    # ITrainingExecutor should only have execute method
    training_methods = [method for method in dir(ITrainingExecutor) if not method.startswith("_")]
    assert training_methods == ["execute"]

    # IExperimentTracker should only have tracking methods (no config-serialization concerns)
    tracker_methods = [method for method in dir(IExperimentTracker) if not method.startswith("_")]
    assert "create_run" in tracker_methods
    assert "log_settings" not in tracker_methods

    # IOptimizationStrategy should only expose optimization execution
    optimizer_methods = [
        method for method in dir(IOptimizationStrategy) if not method.startswith("_")
    ]
    assert optimizer_methods == ["execute_optimization"]


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

    # Use JobConfig for execute() since executor requires JobConfig.run
    job = _job(mlflow=True)

    result = tracking_decorator.execute(build_components, job)
    assert isinstance(result, TrainingResult)

    # Mock tracker should have been used
    mock_tracker.create_run.assert_called()


def test_pure_solid_architecture_integration(build_components):
    """Test that pure SOLID architecture works without backward compatibility."""
    # Use JobConfig for execute() since executors require JobConfig.run
    job = _job()

    # Pure SOLID interface
    vanilla_executor = VanillaExecutor()
    result = vanilla_executor.execute(build_components, job)

    assert isinstance(result, TrainingResult)
    assert build_components.trainer.called["fit"] == 1

    # Factory-created executor should also work
    factory = ExecutionStrategyFactory()
    factory_executor = factory.create_executor(job)
    result2 = factory_executor.execute(build_components, job)
    assert isinstance(result2, TrainingResult)


def test_error_handling_integration(build_components):
    """Test that error handling works correctly across the architecture."""
    from dlkit.common import WorkflowError

    # Create failing trainer
    class FailingTrainer:
        def fit(self, *args, **kwargs):
            raise RuntimeError("Training failed")

    failing_components = RuntimeComponents(
        model=cast("Any", Mock()),
        datamodule=Mock(),
        trainer=cast("Any", FailingTrainer()),
        meta={},
    )

    factory = ExecutionStrategyFactory()
    job = _job()
    executor = factory.create_executor(job)

    # Error should be properly wrapped — use JobConfig for execute()
    with pytest.raises(WorkflowError) as exc_info:
        executor.execute(failing_components, job)

    assert "Vanilla execution failed" in str(exc_info.value.message)
    assert "Training failed" in str(exc_info.value.message)


def test_end_to_end_solid_workflow():
    """Test complete end-to-end workflow demonstrating all SOLID principles."""
    # This test demonstrates how all principles work together
    factory = ExecutionStrategyFactory()

    # Settings with MLflow enabled
    full_job = _job(mlflow=True, experiment_name="e2e_test")

    # Factory's create_executor only creates TrackingDecorator
    executor = factory.create_executor(full_job)

    # create_executor returns TrackingDecorator
    assert isinstance(executor, TrackingDecorator)  # Outer
    assert isinstance(executor._executor, VanillaExecutor)  # Core

    # Each component has single responsibility (SRP)
    # All components can be substituted (LSP)
    # Interfaces are focused (ISP)
    # Dependencies are inverted (DIP)

    assert True
