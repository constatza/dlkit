"""Tests for ExecutionStrategyFactory following SOLID principles."""

from __future__ import annotations

import pytest

from dlkit.engine.tracking.tracking_decorator import TrackingDecorator
from dlkit.engine.training import ITrainingExecutor, VanillaExecutor
from dlkit.engine.workflows.factories.execution_strategy_factory import ExecutionStrategyFactory

# OptimizationDecorator removed - tests updated for clean architecture
from dlkit.infrastructure.config.general_settings import GeneralSettings
from dlkit.infrastructure.config.mlflow_settings import MLflowSettings
from dlkit.infrastructure.config.optuna_settings import OptunaSettings


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
    settings = GeneralSettings()  # No MLFLOW or OPTUNA

    executor = factory.create_executor(settings)

    # Factory now always returns TrackingDecorator (null object pattern)
    assert isinstance(executor, TrackingDecorator)
    assert isinstance(executor._executor, VanillaExecutor)
    # Verify it uses NullTracker when MLflow is disabled (no section AND no env var)
    from dlkit.engine.tracking.interfaces import NullTracker

    assert isinstance(executor._tracker, NullTracker)


def test_factory_creates_tracking_decorator_for_mlflow(factory):
    """Test that factory creates tracking decorator when MLflow is enabled."""
    settings = GeneralSettings(
        MLFLOW=MLflowSettings(
            experiment_name="test",
        )
    )

    executor = factory.create_executor(settings)

    assert isinstance(executor, TrackingDecorator)
    # Verify it wraps a VanillaExecutor
    assert isinstance(executor._executor, VanillaExecutor)


def test_factory_creates_optimization_decorator_for_optuna(factory, monkeypatch):
    """Test that factory creates tracking decorator even when only Optuna is enabled."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(
        "dlkit.engine.tracking.uri_resolver.local_host_alive",
        lambda: False,
    )
    settings = GeneralSettings(
        OPTUNA=OptunaSettings(enabled=True, n_trials=10, direction="minimize")
    )

    executor = factory.create_executor(settings)

    # create_executor always returns TrackingDecorator, not OptimizationDecorator
    # OptimizationDecorator is created by create_optimization_strategy
    assert isinstance(executor, TrackingDecorator)
    # Verify it wraps a VanillaExecutor
    assert isinstance(executor._executor, VanillaExecutor)
    # Should use NullTracker since MLflow is not enabled (no section AND no env var)
    from dlkit.engine.tracking.interfaces import NullTracker

    assert isinstance(executor._tracker, NullTracker)


def test_factory_creates_composed_executor_for_both_features(factory):
    """Test that factory creates tracking decorator even when both features are enabled."""
    settings = GeneralSettings(
        MLFLOW=MLflowSettings(
            experiment_name="test",
        ),
        OPTUNA=OptunaSettings(enabled=True, n_trials=5, direction="minimize"),
    )

    executor = factory.create_executor(settings)

    # create_executor only creates TrackingDecorator wrapping VanillaExecutor
    # OptimizationDecorator is handled by create_optimization_strategy
    assert isinstance(executor, TrackingDecorator)
    assert isinstance(executor._executor, VanillaExecutor)

    # Should have MLflow tracker since MLflow is enabled
    from dlkit.engine.tracking.mlflow_tracker import MLflowTracker

    assert isinstance(executor._tracker, MLflowTracker)


def test_factory_follows_open_closed_principle(factory):
    """Test that factory can be extended without modification (OCP)."""
    # Original factory should work with existing settings
    vanilla_settings = GeneralSettings()
    mlflow_settings = GeneralSettings(MLFLOW=MLflowSettings())
    optuna_settings = GeneralSettings(OPTUNA=OptunaSettings(enabled=True, n_trials=3))

    vanilla_executor = factory.create_executor(vanilla_settings)
    mlflow_executor = factory.create_executor(mlflow_settings)
    optuna_executor = factory.create_executor(optuna_settings)

    # All should implement the same interface
    assert isinstance(vanilla_executor, ITrainingExecutor)
    assert isinstance(mlflow_executor, ITrainingExecutor)
    assert isinstance(optuna_executor, ITrainingExecutor)


def test_factory_mlflow_detection_logic(factory):
    """Test MLflow feature detection logic."""
    # MLflow disabled cases
    assert not factory._has_mlflow_config(GeneralSettings())
    assert not factory._has_mlflow_config(GeneralSettings(MLFLOW=None))

    # MLflow enabled case
    assert factory._has_mlflow_config(GeneralSettings(MLFLOW=MLflowSettings()))


def test_factory_activates_mlflow_when_local_probe_is_mocked_true(factory, monkeypatch):
    """Test MLflow activation from a mocked localhost probe without real server I/O."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(
        "dlkit.engine.tracking.uri_resolver.local_host_alive",
        lambda: True,
    )
    settings = GeneralSettings()  # No MLFLOW section, no env var

    executor = factory.create_executor(settings)

    assert isinstance(executor, TrackingDecorator)
    from dlkit.engine.tracking.mlflow_tracker import MLflowTracker

    assert isinstance(executor._tracker, MLflowTracker)


def test_factory_optimization_workflow_detection(factory):
    """Test optimization workflow detection logic."""
    from dlkit.infrastructure.config.workflow_configs import (
        OptimizationWorkflowConfig,
    )

    # General settings and training settings are not optimization
    assert not factory._is_optimization_workflow(GeneralSettings())

    # Optimization workflow should be detected using isinstance check
    # We test the logic with a mock object that satisfies isinstance check
    class MockOptimizationConfig(OptimizationWorkflowConfig):
        """Mock config for testing."""

        def __init__(self):
            # Bypass pydantic validation
            pass

    mock_opt = MockOptimizationConfig()
    assert factory._is_optimization_workflow(mock_opt)
    assert isinstance(mock_opt, OptimizationWorkflowConfig)


def test_factory_direct_usage():
    """Test that factory can be used directly."""
    from dlkit.engine.workflows.factories.execution_strategy_factory import (
        ExecutionStrategyFactory,
    )

    settings = GeneralSettings(MLFLOW=MLflowSettings())

    factory = ExecutionStrategyFactory()
    executor = factory.create_executor(settings)

    assert isinstance(executor, TrackingDecorator)
    assert isinstance(executor, ITrainingExecutor)


def test_factory_dependency_injection_ready():
    """Test that factory can be easily dependency-injected."""
    # Factory should be stateless and easily mockable
    factory1 = ExecutionStrategyFactory()
    factory2 = ExecutionStrategyFactory()

    settings = GeneralSettings()

    executor1 = factory1.create_executor(settings)
    executor2 = factory2.create_executor(settings)

    # Should create equivalent executors
    assert type(executor1) is type(executor2)
    assert isinstance(executor1, TrackingDecorator)
    assert isinstance(executor2, TrackingDecorator)


def test_factory_handles_partial_configurations():
    """Test that factory handles partial/incomplete configurations gracefully."""
    # MLflow with minimal config
    minimal_mlflow = GeneralSettings(
        MLFLOW=MLflowSettings()  # Minimal flat config
    )
    executor = ExecutionStrategyFactory().create_executor(minimal_mlflow)
    assert isinstance(executor, TrackingDecorator)

    # Optuna with minimal config
    minimal_optuna = GeneralSettings(
        OPTUNA=OptunaSettings(enabled=True)  # No n_trials specified
    )
    executor = ExecutionStrategyFactory().create_executor(minimal_optuna)
    # create_executor always returns TrackingDecorator regardless of Optuna settings
    assert isinstance(executor, TrackingDecorator)
