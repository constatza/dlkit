"""Tests for ExecutionStrategyFactory following SOLID principles."""

from __future__ import annotations

import pytest

from dlkit.engine.tracking.tracking_decorator import TrackingDecorator
from dlkit.engine.training import ITrainingExecutor, VanillaExecutor
from dlkit.engine.workflows.factories.execution_strategy_factory import ExecutionStrategyFactory
from dlkit.infrastructure.config.experiment_settings import ExperimentSettings
from dlkit.infrastructure.config.job_config import JobConfig, SearchJobConfig
from dlkit.infrastructure.config.run_settings import RunSettings
from dlkit.infrastructure.config.tracking_settings import TrackingSettings


def _training_job(
    *,
    mlflow: bool = False,
    experiment_name: str = "test",
) -> JobConfig:
    """Create a minimal JobConfig for factory tests.

    Args:
        mlflow: Whether to enable mlflow tracking backend.
        experiment_name: Name for the experiment section.

    Returns:
        Minimal JobConfig suitable for passing to ExecutionStrategyFactory.
    """
    return JobConfig(
        run=RunSettings(type="train"),
        experiment=ExperimentSettings(name=experiment_name) if mlflow else None,
        tracking=TrackingSettings(backend="mlflow") if mlflow else TrackingSettings(),
    )


@pytest.fixture
def factory():
    """Create factory instance for testing."""
    return ExecutionStrategyFactory()


def test_factory_creates_vanilla_executor_by_default(factory, monkeypatch):
    """Test that factory creates tracking decorator with null tracker when no features are enabled."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(
        "dlkit.engine.workflows.factories.execution_strategy_factory.ExecutionStrategyFactory._has_mlflow_config_or_env",
        lambda self, s: False,
    )
    settings = _training_job()  # No mlflow backend, no env var

    executor = factory.create_executor(settings)

    # Factory now always returns TrackingDecorator (null object pattern)
    assert isinstance(executor, TrackingDecorator)
    assert isinstance(executor._executor, VanillaExecutor)
    # Verify it uses NullTracker when MLflow is disabled (no section AND no env var)
    from dlkit.engine.tracking.interfaces import NullTracker

    assert isinstance(executor._tracker, NullTracker)


def test_factory_creates_tracking_decorator_for_mlflow(factory):
    """Test that factory creates tracking decorator when MLflow is enabled."""
    settings = _training_job(mlflow=True, experiment_name="test")

    executor = factory.create_executor(settings)

    assert isinstance(executor, TrackingDecorator)
    # Verify it wraps a VanillaExecutor
    assert isinstance(executor._executor, VanillaExecutor)


def test_factory_creates_optimization_decorator_for_optuna(factory, monkeypatch):
    """Test that factory creates tracking decorator even when only a plain job is given."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(
        "dlkit.engine.tracking.uri_resolver.local_host_alive",
        lambda: False,
    )
    settings = _training_job()  # No mlflow backend

    executor = factory.create_executor(settings)

    # create_executor always returns TrackingDecorator, not OptimizationDecorator
    assert isinstance(executor, TrackingDecorator)
    # Verify it wraps a VanillaExecutor
    assert isinstance(executor._executor, VanillaExecutor)
    # Should use NullTracker since MLflow is not enabled
    from dlkit.engine.tracking.interfaces import NullTracker

    assert isinstance(executor._tracker, NullTracker)


def test_factory_creates_composed_executor_for_both_features(factory):
    """Test that factory creates tracking decorator when mlflow is enabled."""
    settings = _training_job(mlflow=True, experiment_name="test")

    executor = factory.create_executor(settings)

    # create_executor only creates TrackingDecorator wrapping VanillaExecutor
    assert isinstance(executor, TrackingDecorator)
    assert isinstance(executor._executor, VanillaExecutor)

    # Should have MLflow tracker since MLflow is enabled
    from dlkit.engine.tracking.mlflow_tracker import MLflowTracker

    assert isinstance(executor._tracker, MLflowTracker)


def test_factory_follows_open_closed_principle(factory):
    """Test that factory can be extended without modification (OCP)."""
    # Original factory should work with existing settings
    vanilla_settings = _training_job()
    mlflow_settings = _training_job(mlflow=True)
    search_settings = _training_job()  # Plain job without mlflow

    vanilla_executor = factory.create_executor(vanilla_settings)
    mlflow_executor = factory.create_executor(mlflow_settings)
    search_executor = factory.create_executor(search_settings)

    # All should implement the same interface
    assert isinstance(vanilla_executor, ITrainingExecutor)
    assert isinstance(mlflow_executor, ITrainingExecutor)
    assert isinstance(search_executor, ITrainingExecutor)


def test_factory_mlflow_detection_logic(factory):
    """Test MLflow feature detection logic."""
    # MLflow disabled case: no backend set
    no_mlflow = _training_job()
    assert not factory._has_mlflow_config_or_env(no_mlflow)

    # MLflow enabled case: backend == "mlflow"
    with_mlflow = _training_job(mlflow=True)
    assert factory._has_mlflow_config_or_env(with_mlflow)


def test_factory_activates_mlflow_when_local_probe_is_mocked_true(factory, monkeypatch):
    """Test MLflow activation from a mocked localhost probe without real server I/O."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(
        "dlkit.engine.tracking.uri_resolver.local_host_alive",
        lambda: True,
    )
    settings = _training_job()  # No mlflow section, no env var

    executor = factory.create_executor(settings)

    assert isinstance(executor, TrackingDecorator)
    from dlkit.engine.tracking.mlflow_tracker import MLflowTracker

    assert isinstance(executor._tracker, MLflowTracker)


def test_factory_optimization_workflow_detection(factory):
    """Test optimization workflow detection using new SearchJobConfig."""
    # Non-search configs are not detected as optimization
    plain_job = _training_job()
    assert not factory._is_optimization_workflow(plain_job)

    # SearchJobConfig with missing required fields raises ValidationError at construction
    with pytest.raises(Exception):
        SearchJobConfig.model_validate({"run": {"type": "search"}})


def test_factory_direct_usage():
    """Test that factory can be used directly."""
    from dlkit.engine.workflows.factories.execution_strategy_factory import (
        ExecutionStrategyFactory,
    )

    settings = _training_job(mlflow=True)

    factory = ExecutionStrategyFactory()
    executor = factory.create_executor(settings)

    assert isinstance(executor, TrackingDecorator)
    assert isinstance(executor, ITrainingExecutor)


def test_factory_dependency_injection_ready():
    """Test that factory can be easily dependency-injected."""
    # Factory should be stateless and easily mockable
    factory1 = ExecutionStrategyFactory()
    factory2 = ExecutionStrategyFactory()

    settings = _training_job()

    executor1 = factory1.create_executor(settings)
    executor2 = factory2.create_executor(settings)

    # Should create equivalent executors
    assert type(executor1) is type(executor2)
    assert isinstance(executor1, TrackingDecorator)
    assert isinstance(executor2, TrackingDecorator)


def test_factory_handles_partial_configurations():
    """Test that factory handles partial/incomplete configurations gracefully."""
    # MLflow with minimal config
    minimal_mlflow = _training_job(mlflow=True)
    executor = ExecutionStrategyFactory().create_executor(minimal_mlflow)
    assert isinstance(executor, TrackingDecorator)

    # Plain job (no mlflow)
    plain_job = _training_job()
    executor = ExecutionStrategyFactory().create_executor(plain_job)
    # create_executor always returns TrackingDecorator
    assert isinstance(executor, TrackingDecorator)
