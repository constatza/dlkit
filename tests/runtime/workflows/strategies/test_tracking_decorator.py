"""Tests for TrackingDecorator following SOLID principles."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock
from typing import Any

import pytest

from dlkit.interfaces.api.domain import WorkflowError, TrainingResult
from dlkit.runtime.workflows.strategies.core import VanillaExecutor
from dlkit.runtime.workflows.strategies.tracking import (
    TrackingDecorator,
    IExperimentTracker,
    IRunContext,
)
from dlkit.runtime.workflows.factories.build_factory import BuildComponents
from dlkit.tools.config.general_settings import GeneralSettings
from dlkit.tools.config.mlflow_settings import MLflowSettings, MLflowClientSettings


class MockRunContext(IRunContext):
    """Mock implementation of IRunContext for testing."""

    def __init__(self):
        self._run_id = "mock-run-id-12345"
        self.logged_metrics = {}
        self.logged_params = {}
        self.logged_artifacts = []
        self.tags = {}

    @property
    def run_id(self) -> str:
        """Get the run ID for this tracking run."""
        return self._run_id

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.logged_metrics.update(metrics)

    def log_params(self, params: dict[str, Any]) -> None:
        self.logged_params.update(params)

    def log_artifact(self, artifact_path: Path, artifact_dir: str = "") -> None:
        self.logged_artifacts.append((artifact_path, artifact_dir))

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value


class MockExperimentTracker(IExperimentTracker):
    """Mock implementation of IExperimentTracker for testing."""

    def __init__(self):
        self.run_context = MockRunContext()
        self.created_runs = []
        self.logged_settings = []

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        return False

    @contextmanager
    def create_run(
        self,
        experiment_name: str | None = None,
        run_name: str | None = None,
        nested: bool = False,
    ):
        self.created_runs.append({
            "experiment_name": experiment_name,
            "run_name": run_name,
            "nested": nested,
        })
        yield self.run_context

    def log_settings(self, settings: GeneralSettings, run_context: IRunContext) -> None:
        self.logged_settings.append(settings)

    def log_model_parameters(
        self, model: Any, run_context: IRunContext, settings: GeneralSettings
    ) -> None:
        """Mock model parameter logging."""
        pass


@pytest.fixture
def mock_executor():
    """Create a mock executor for testing."""
    executor = Mock(spec=VanillaExecutor)
    executor.execute.return_value = TrainingResult(
        model_state=None,
        metrics={"val_loss": 0.5, "accuracy": 0.95, "status": "ok"},
        artifacts={},
        duration_seconds=120.0,
    )
    return executor


@pytest.fixture
def mock_tracker():
    """Create a mock tracker for testing."""
    return MockExperimentTracker()


@pytest.fixture
def mlflow_settings():
    """Create MLflow settings for testing."""
    return GeneralSettings(
        MLFLOW=MLflowSettings(
            enabled=True,
            client=MLflowClientSettings(
                tracking_uri="http://localhost:5000",
                experiment_name="test_experiment",
                run_name="test_run",
            ),
        )
    )


@pytest.fixture
def build_components():
    """Create BuildComponents for testing."""

    @dataclass(frozen=True)
    class DummyModel:
        def ensure_precision_applied(self):
            pass

    trainer = Mock()
    trainer.callbacks = []
    trainer.callback_metrics = {"val_loss": 0.5}
    trainer.logged_metrics = {"train_loss": 0.3, "lr-Adam": 0.001}
    trainer.progress_bar_metrics = {}
    # Make fit, predict, test do nothing
    trainer.fit.return_value = None
    trainer.predict.return_value = None
    trainer.test.return_value = None

    datamodule = Mock()

    return BuildComponents(
        model=DummyModel(), datamodule=datamodule, trainer=trainer, shape_spec=None, meta={}
    )


def test_tracking_decorator_single_responsibility(
    mock_executor, mock_tracker, mlflow_settings, build_components
):
    """Test that TrackingDecorator has single responsibility: adding tracking to execution."""
    decorator = TrackingDecorator(mock_executor, mock_tracker)

    result = decorator.execute(build_components, mlflow_settings)

    # Verify tracking was performed
    assert len(mock_tracker.created_runs) == 1
    # Experiment name priority: SESSION.name → MLFLOW.client.experiment_name → "DLKit"
    # Default SESSION.name is "session", so that takes priority
    assert mock_tracker.created_runs[0]["experiment_name"] == "session"
    assert mock_tracker.created_runs[0]["run_name"] == "test_run"

    # Verify settings were logged
    assert len(mock_tracker.logged_settings) == 1

    # Verify underlying executor was called
    mock_executor.execute.assert_called_once_with(build_components, mlflow_settings)

    # Verify result enrichment
    assert isinstance(result, TrainingResult)
    assert mock_tracker.run_context.logged_metrics == {"val_loss": 0.5, "accuracy": 0.95}
    assert mock_tracker.run_context.logged_params["metric_status"] == "ok"


def test_tracking_decorator_composition_pattern(mock_tracker, mlflow_settings, build_components):
    """Test that decorator properly composes with base executor (OCP compliance)."""
    base_executor = VanillaExecutor()
    decorator = TrackingDecorator(base_executor, mock_tracker)

    # Should be able to compose without modifying base executor
    result = decorator.execute(build_components, mlflow_settings)

    assert isinstance(result, TrainingResult)
    assert len(mock_tracker.created_runs) == 1
    assert mock_tracker.run_context.logged_metrics["val_loss"] == 0.5


def test_tracking_decorator_mlflow_disabled_error(mock_executor, build_components):
    """Test that decorator works gracefully when MLflow is not configured."""
    settings_no_mlflow = GeneralSettings()  # No MLFLOW section

    # Use NullTracker for the proper "disabled MLflow" scenario
    from dlkit.runtime.workflows.strategies.tracking.interfaces import NullTracker

    decorator = TrackingDecorator(mock_executor, NullTracker())

    # Should delegate to executor without error
    decorator.execute(build_components, settings_no_mlflow)

    # Should have called the underlying executor
    mock_executor.execute.assert_called_once_with(build_components, settings_no_mlflow)


def test_tracking_decorator_server_metadata_tagging(
    mock_executor, mock_tracker, mlflow_settings, build_components
):
    """Test that server metadata is properly tagged."""

    # Mock tracker with server info capabilities
    class MockMLflowTracker(MockExperimentTracker):
        def setup_mlflow_config(self, mlflow_config):
            return "http://localhost:5000", {"running": True, "response_time": 0.1}

    tracker = MockMLflowTracker()
    decorator = TrackingDecorator(mock_executor, tracker)

    result = decorator.execute(build_components, mlflow_settings)

    # Verify server tags were set
    assert tracker.run_context.tags["mlflow_server_url"] == "http://localhost:5000"
    assert tracker.run_context.tags["mlflow_server_running"] == "True"
    assert tracker.run_context.tags["mlflow_server_response_time"] == "0.1"
    assert isinstance(result, TrainingResult)


def test_tracking_decorator_dependency_inversion(mock_executor, mlflow_settings, build_components):
    """Test that decorator depends on abstractions, not implementations (DIP)."""
    # Can inject any tracker implementation
    tracker1 = MockExperimentTracker()
    tracker2 = MockExperimentTracker()

    decorator1 = TrackingDecorator(mock_executor, tracker1)
    decorator2 = TrackingDecorator(mock_executor, tracker2)

    # Both should work with same interface
    result1 = decorator1.execute(build_components, mlflow_settings)
    result2 = decorator2.execute(build_components, mlflow_settings)

    assert isinstance(result1, TrainingResult)
    assert isinstance(result2, TrainingResult)

    # Different tracker instances should be used
    assert len(tracker1.created_runs) == 1
    assert len(tracker2.created_runs) == 1


def test_tracking_decorator_exception_handling(
    mock_executor, mock_tracker, mlflow_settings, build_components
):
    """Test that tracking exceptions are properly wrapped."""
    mock_executor.execute.side_effect = RuntimeError("Training failed")

    decorator = TrackingDecorator(mock_executor, mock_tracker)

    with pytest.raises(WorkflowError) as exc_info:
        decorator.execute(build_components, mlflow_settings)

    assert "Training with tracking failed" in str(exc_info.value.message)
    assert "Training failed" in str(exc_info.value.message)
    assert exc_info.value.context["stage"] == "tracking"


def test_tracking_decorator_result_enrichment(
    mock_executor, mock_tracker, mlflow_settings, build_components
):
    """Test that results are enriched with tracking metadata."""

    # Mock tracker to simulate MLflow enrichment
    class EnrichingTracker(MockExperimentTracker):
        def setup_mlflow_config(self, mlflow_config):
            return "http://localhost:5000", {"running": True, "response_time": 0.05}

    tracker = EnrichingTracker()
    decorator = TrackingDecorator(mock_executor, tracker)

    result = decorator.execute(build_components, mlflow_settings)

    # Original metrics should be preserved
    assert result.metrics["val_loss"] == 0.5
    assert result.metrics["accuracy"] == 0.95

    # Should not have added enrichment in our mock (that's handled by real MLflowTracker)
    assert isinstance(result, TrainingResult)
