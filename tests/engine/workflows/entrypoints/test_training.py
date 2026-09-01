"""Tests for the train() entrypoint's shared ``EntrypointContext.run()`` plumbing.

``train()`` is one of five entrypoints (``fit``, ``training``, ``optimization``,
``convergence``, ``evaluate``) that route their workflow execution through
``EntrypointContext.run()``: a raw exception from the orchestrator is wrapped
as ``WorkflowError``, while any ``DLKitError`` raised by the workflow itself
(``WorkflowError`` included) propagates unchanged. These tests stub the
orchestrator instead of running a real training pipeline, mirroring
``test_fit.py``'s pattern.
"""

from __future__ import annotations

import pytest

from dlkit.common import TrainingResult
from dlkit.common.errors import TrackingError, WorkflowError
from dlkit.engine.workflows.entrypoints import training as training_module
from dlkit.infrastructure.config.job_config import TrainingJobConfig


class _StubOrchestrator:
    """Stands in for Orchestrator: returns a canned result instead of running
    a real build+execute pipeline, which is unrelated to what this change
    touches."""

    result: TrainingResult | None = None
    exception: Exception | None = None

    def execute_training(self, settings: TrainingJobConfig, hooks: object = None) -> TrainingResult:
        del settings, hooks
        if _StubOrchestrator.exception is not None:
            raise _StubOrchestrator.exception
        assert _StubOrchestrator.result is not None
        return _StubOrchestrator.result


@pytest.fixture(autouse=True)
def stub_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training_module, "Orchestrator", _StubOrchestrator)
    _StubOrchestrator.result = None
    _StubOrchestrator.exception = None


@pytest.fixture
def training_settings() -> TrainingJobConfig:
    return TrainingJobConfig.model_validate(
        {
            "run": {"type": "train", "seed": 0},
            "model": {"class": "Dummy", "module_path": "dlkit.domain.nn"},
            "data": {"batch_size": 4, "num_workers": 0},
            "training": {
                "loss": "mse",
                "trainer": {"fast_dev_run": True, "accelerator": "cpu"},
                "optimizer": {"name": "AdamW", "lr": 1e-3},
            },
        }
    )


def test_train_returns_result_built_from_orchestrator_execution(
    training_settings: TrainingJobConfig,
) -> None:
    _StubOrchestrator.result = TrainingResult(
        model_state=None, metrics={"loss": 0.1}, artifacts={}, duration_seconds=1.5
    )

    result = training_module.train(training_settings)

    assert result.metrics == {"loss": 0.1}
    # `train()` prefers `context.elapsed()`'s wall-clock reading over the
    # stub's `duration_seconds` whenever it's positive, which it always is
    # here — so this only asserts a duration was actually populated.
    assert result.duration_seconds > 0


def test_train_wraps_unexpected_exception_as_workflow_error(
    training_settings: TrainingJobConfig,
) -> None:
    _StubOrchestrator.exception = RuntimeError("boom")

    with pytest.raises(WorkflowError, match="Training execution failed"):
        training_module.train(training_settings)


def test_train_reraises_workflow_error_unchanged(training_settings: TrainingJobConfig) -> None:
    original = WorkflowError("orchestrator-specific failure", {"stage": "build"})
    _StubOrchestrator.exception = original

    with pytest.raises(WorkflowError) as exc_info:
        training_module.train(training_settings)

    assert exc_info.value is original


def test_train_reraises_any_dlkit_error_unchanged(training_settings: TrainingJobConfig) -> None:
    """Not just WorkflowError -- any DLKitError raised by the workflow (e.g. a
    TrackingError from a failed MLflow connectivity check) must survive
    EntrypointContext.run()'s wrapping boundary unflattened, since the CLI's
    per-type suggestion dispatch depends on seeing the real type.
    """
    original = TrackingError("MLflow tracking backend unreachable")
    _StubOrchestrator.exception = original

    with pytest.raises(TrackingError) as exc_info:
        training_module.train(training_settings)

    assert exc_info.value is original
