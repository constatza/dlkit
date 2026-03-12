"""Tests for VanillaExecutor following SOLID principles."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from dlkit.interfaces.api.domain import WorkflowError, TrainingResult
from dlkit.runtime.workflows.strategies.core import VanillaExecutor
from dlkit.runtime.workflows.factories.build_factory import BuildComponents
from dlkit.tools.config.general_settings import GeneralSettings
from dlkit.tools.config.training_settings import TrainingSettings
from dlkit.tools.config.session_settings import SessionSettings


@pytest.fixture
def trainer_stub():
    """Create a trainer stub for testing."""

    class TrainerStub:
        def __init__(self):
            self.called = {"fit": 0, "predict": 0, "test": 0}
            self.logged_metrics = {"val_loss": 1.23, "acc": 0.9, "tensor_val": "non_numeric"}
            self.callbacks = []

        def fit(self, *args, **kwargs):
            self.called["fit"] += 1

        def predict(self, *args, **kwargs):
            self.called["predict"] += 1

        def test(self, *args, **kwargs):
            self.called["test"] += 1

    return TrainerStub()


@pytest.fixture
def build_components(trainer_stub):
    """Create BuildComponents for testing."""

    @dataclass(frozen=True, slots=True)
    class DummyModel:
        pass

    return BuildComponents(
        model=DummyModel(), datamodule=object(), trainer=trainer_stub, shape_spec=None, meta={}
    )


@pytest.fixture
def settings():
    """Create basic settings for testing."""
    return GeneralSettings(SESSION=SessionSettings(inference=False), TRAINING=TrainingSettings())


def test_vanilla_executor_single_responsibility(build_components, settings, trainer_stub):
    """Test that VanillaExecutor has single responsibility: pure training execution."""
    executor = VanillaExecutor()

    result = executor.execute(build_components, settings)

    # Verify training execution
    assert trainer_stub.called["fit"] == 1
    assert trainer_stub.called["predict"] == 1
    assert trainer_stub.called["test"] == 1

    # Verify result structure
    assert isinstance(result, TrainingResult)
    assert isinstance(result.metrics, dict)
    assert isinstance(result.artifacts, dict)
    assert result.duration_seconds == 0.0


def test_vanilla_executor_metric_extraction(build_components, settings, trainer_stub):
    """Test metric extraction from trainer."""
    executor = VanillaExecutor()

    result = executor.execute(build_components, settings)

    # Numeric metrics should be converted to float
    assert result.metrics["val_loss"] == 1.23
    assert result.metrics["acc"] == 0.9

    # Non-numeric values should be preserved as-is
    assert result.metrics["tensor_val"] == "non_numeric"


def test_vanilla_executor_no_trainer_error(settings):
    """Test that executor raises WorkflowError when trainer is None."""
    components = BuildComponents(
        model=object(),
        datamodule=object(),
        trainer=None,  # No trainer
        shape_spec=None,
        meta={},
    )

    executor = VanillaExecutor()

    with pytest.raises(WorkflowError) as exc_info:
        executor.execute(components, settings)

    assert "Trainer is required for training" in str(exc_info.value)
    assert exc_info.value.context["stage"] == "execute"


def test_vanilla_executor_trainer_exception_handling(settings):
    """Test that trainer exceptions are properly wrapped."""

    class FailingTrainer:
        def fit(self, *args, **kwargs):
            raise RuntimeError("Training failed")

    components = BuildComponents(
        model=object(), datamodule=object(), trainer=FailingTrainer(), shape_spec=None, meta={}
    )

    executor = VanillaExecutor()

    with pytest.raises(WorkflowError) as exc_info:
        executor.execute(components, settings)

    assert "Vanilla execution failed" in str(exc_info.value.message)
    assert "Training failed" in str(exc_info.value.message)
    assert exc_info.value.context["stage"] == "execute"


def test_vanilla_executor_post_training_exception_handling(build_components, settings):
    """Test that predict/test exceptions don't fail the entire execution."""

    class PartiallyFailingTrainer:
        def __init__(self):
            self.called = {"fit": 0, "predict": 0, "test": 0}
            self.logged_metrics = {"val_loss": 1.5}
            self.callbacks = []

        def fit(self, *args, **kwargs):
            self.called["fit"] += 1

        def predict(self, *args, **kwargs):
            self.called["predict"] += 1
            raise RuntimeError("Predict failed")

        def test(self, *args, **kwargs):
            self.called["test"] += 1
            raise RuntimeError("Test failed")

    failing_trainer = PartiallyFailingTrainer()
    components = BuildComponents(
        model=object(), datamodule=object(), trainer=failing_trainer, shape_spec=None, meta={}
    )

    executor = VanillaExecutor()

    # Should not raise exception despite predict/test failures
    result = executor.execute(components, settings)

    assert isinstance(result, TrainingResult)
    assert failing_trainer.called["fit"] == 1
    assert failing_trainer.called["predict"] == 1  # Called but failed
    assert failing_trainer.called["test"] == 1  # Called but failed
