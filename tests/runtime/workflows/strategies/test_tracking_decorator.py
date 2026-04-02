"""Tests for TrackingDecorator following SOLID principles."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import pytest

from dlkit.runtime.execution import VanillaExecutor
from dlkit.runtime.execution.components import RuntimeComponents
from dlkit.runtime.tracking.interfaces import IExperimentTracker, IRunContext
from dlkit.runtime.tracking.tracking_decorator import TrackingDecorator
from dlkit.shared import TrainingResult, WorkflowError
from dlkit.tools.config.general_settings import GeneralSettings
from dlkit.tools.config.mlflow_settings import MLflowSettings
from dlkit.tools.config.workflow_configs import OptimizationWorkflowConfig, TrainingWorkflowConfig

type _WorkflowSettings = GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig


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

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.logged_metrics.update(metrics)

    def log_params(self, params: dict[str, Any]) -> None:
        self.logged_params.update(params)

    def log_artifact_content(self, content: str | bytes, artifact_file: str) -> None:
        pass

    def log_artifact(self, artifact_path: Path, artifact_dir: str = "") -> None:
        self.logged_artifacts.append((artifact_path, artifact_dir))

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value

    def log_dataset(
        self, dataset: Any, context: str | None = None, tags: dict[str, str] | None = None
    ) -> None:
        """Mock dataset logging."""
        if not hasattr(self, "logged_datasets"):
            self.logged_datasets = []
        self.logged_datasets.append({"dataset": dataset, "context": context, "tags": tags})

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        *,
        registered_model_name: str | None = None,
        signature: Any | None = None,
        input_example: Any | None = None,
    ) -> str | None:
        if not hasattr(self, "logged_models"):
            self.logged_models = []
        self.logged_models.append(
            {
                "artifact_path": artifact_path,
                "registered_model_name": registered_model_name,
            }
        )
        return f"runs:/{self._run_id}/{artifact_path}"

    def get_latest_model_version(
        self,
        model_name: str,
        *,
        run_id: str | None = None,
        artifact_path: str | None = None,
    ) -> int | None:
        return 1

    def set_model_alias(self, model_name: str, alias: str, version: int) -> None:
        if not hasattr(self, "model_aliases"):
            self.model_aliases = []
        self.model_aliases.append((model_name, alias, version))

    def set_model_version_tag(
        self,
        model_name: str,
        version: int,
        key: str,
        value: str,
    ) -> None:
        if not hasattr(self, "model_version_tags"):
            self.model_version_tags = []
        self.model_version_tags.append((model_name, version, key, value))


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
        tags: dict[str, str] | None = None,
    ):
        self.created_runs.append(
            {
                "experiment_name": experiment_name,
                "run_name": run_name,
                "nested": nested,
                "tags": tags,
            }
        )
        yield self.run_context

    def log_settings(self, settings: _WorkflowSettings, run_context: IRunContext) -> None:
        self.logged_settings.append(settings)

    def log_model_parameters(
        self, model: Any, run_context: IRunContext, settings: _WorkflowSettings
    ) -> None:
        """Mock model parameter logging."""

    def get_tracking_uri(self) -> str | None:
        return None

    def is_local(self) -> bool:
        return False


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
            experiment_name="test_experiment",
            run_name="test_run",
        )
    )


@pytest.fixture
def build_components():
    """Create RuntimeComponents for testing."""

    @dataclass(frozen=True, slots=True)
    class DummyModel:
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

    return RuntimeComponents(
        model=cast("Any", DummyModel()),
        datamodule=datamodule,
        trainer=trainer,
        shape_spec=None,
        meta={},
    )


def test_tracking_decorator_single_responsibility(
    mock_executor, mock_tracker, mlflow_settings, build_components
):
    """Test that TrackingDecorator has single responsibility: adding tracking to execution."""
    decorator = TrackingDecorator(mock_executor, mock_tracker)

    result = decorator.execute(build_components, mlflow_settings)

    # Verify tracking was performed
    assert len(mock_tracker.created_runs) == 1
    # Experiment name priority: MLFLOW.experiment_name → SESSION.name → "DLKit"
    # Explicit MLFLOW.experiment_name takes priority over SESSION.name
    assert mock_tracker.created_runs[0]["experiment_name"] == "test_experiment"
    assert mock_tracker.created_runs[0]["run_name"] == "test_run"

    # Verify settings were logged
    assert len(mock_tracker.logged_settings) == 1

    # Verify underlying executor was called
    mock_executor.execute.assert_called_once_with(build_components, mlflow_settings)

    # Verify result enrichment
    assert isinstance(result, TrainingResult)
    # Only non-stage-specific metrics should be logged (val_loss is filtered out)
    assert mock_tracker.run_context.logged_metrics == {"accuracy": 0.95}
    assert mock_tracker.run_context.logged_params["metric_status"] == "ok"


def test_tracking_decorator_composition_pattern(mock_tracker, mlflow_settings, build_components):
    """Test that decorator properly composes with base executor (OCP compliance)."""
    base_executor = VanillaExecutor()
    decorator = TrackingDecorator(base_executor, mock_tracker)

    # Should be able to compose without modifying base executor
    result = decorator.execute(build_components, mlflow_settings)

    assert isinstance(result, TrainingResult)
    assert len(mock_tracker.created_runs) == 1
    # val_loss is filtered out as it's a stage-specific metric already logged by MLflowEpochLogger
    assert "val_loss" not in mock_tracker.run_context.logged_metrics


def test_tracking_decorator_mlflow_disabled_error(mock_executor, build_components):
    """Test that decorator works gracefully when MLflow is not configured."""
    settings_no_mlflow = GeneralSettings()  # No MLFLOW section

    # Use NullTracker for the proper "disabled MLflow" scenario
    from dlkit.runtime.tracking.interfaces import NullTracker

    decorator = TrackingDecorator(mock_executor, NullTracker())

    # Should delegate to executor without error
    decorator.execute(build_components, settings_no_mlflow)

    # Should have called the underlying executor
    mock_executor.execute.assert_called_once_with(build_components, settings_no_mlflow)


def test_tracking_decorator_tracking_uri_tagging(
    mock_executor, mock_tracker, mlflow_settings, build_components
):
    """Test that tracking URI metadata is tagged."""

    # Mock tracker with server info capabilities
    class MockMLflowTracker(MockExperimentTracker):
        def configure(self, mlflow_config: Any, *, root_dir: Any = None) -> None:
            pass

        def get_tracking_uri(self) -> str:
            return "http://localhost:5000"

    tracker = MockMLflowTracker()
    decorator = TrackingDecorator(mock_executor, tracker)

    result = decorator.execute(build_components, mlflow_settings)

    assert tracker.run_context.tags["mlflow_tracking_uri"] == "http://localhost:5000"
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

    tracker = MockExperimentTracker()
    decorator = TrackingDecorator(mock_executor, tracker)

    result = decorator.execute(build_components, mlflow_settings)

    # Original metrics should be preserved
    assert result.metrics["val_loss"] == 0.5
    assert result.metrics["accuracy"] == 0.95

    # Should not have added enrichment in our mock (that's handled by real MLflowTracker)
    assert isinstance(result, TrainingResult)
