"""Tests for the converge() entrypoint's shared ``EntrypointContext.run()`` plumbing.

``converge()`` is one of five entrypoints (``fit``, ``training``,
``optimization``, ``convergence``, ``evaluate``) that route their workflow
execution through ``EntrypointContext.run()``: a raw exception from the
convergence orchestrator is wrapped as ``WorkflowError``, while a
``WorkflowError`` raised by the workflow itself — such as ``converge()``'s own
``ConvergenceJobConfig`` type-check — propagates unchanged. These tests stub
``ConvergenceOrchestrator`` instead of running a real multi-size sweep,
mirroring ``test_fit.py``'s pattern of standing in for the heavy collaborator.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlkit.common.errors import WorkflowError
from dlkit.common.results import ConvergenceResult
from dlkit.engine.workflows.entrypoints import convergence as convergence_module
from dlkit.infrastructure.config.job_config import ConvergenceJobConfig, TrainingJobConfig


class _StubConvergenceOrchestrator:
    """Stands in for ConvergenceOrchestrator: returns a canned result or
    raises, instead of running a real multi-size sweep."""

    result: ConvergenceResult | None = None
    exception: Exception | None = None

    def __init__(self, multi_run: object) -> None:
        del multi_run

    def execute(self, settings: ConvergenceJobConfig) -> ConvergenceResult:
        del settings
        if _StubConvergenceOrchestrator.exception is not None:
            raise _StubConvergenceOrchestrator.exception
        assert _StubConvergenceOrchestrator.result is not None
        return _StubConvergenceOrchestrator.result


@pytest.fixture(autouse=True)
def stub_convergence_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(convergence_module, "ConvergenceOrchestrator", _StubConvergenceOrchestrator)
    _StubConvergenceOrchestrator.result = None
    _StubConvergenceOrchestrator.exception = None


@pytest.fixture
def convergence_settings(
    minimal_dataset: dict[str, Path], sqlite_tracking_uri: str
) -> ConvergenceJobConfig:
    return ConvergenceJobConfig.model_validate(
        {
            "run": {"type": "convergence", "seed": 0},
            "model": {"class": "Dummy", "module_path": "dlkit.domain.nn"},
            "data": {
                "batch_size": 4,
                "num_workers": 0,
                "features": [
                    {"name": "x", "path": str(minimal_dataset["features"]), "format": "npy"}
                ],
                "targets": [
                    {"name": "y", "path": str(minimal_dataset["targets"]), "format": "npy"}
                ],
            },
            "training": {
                "loss": "mse",
                "trainer": {"fast_dev_run": True, "accelerator": "cpu"},
                "optimizer": {"name": "AdamW", "lr": 1e-3},
            },
            "convergence": {"sizes": [4, 8], "repeats": 1},
            "tracking": {"backend": "mlflow", "uri": sqlite_tracking_uri},
        }
    )


def test_converge_returns_orchestrator_result_on_success(
    convergence_settings: ConvergenceJobConfig,
) -> None:
    _StubConvergenceOrchestrator.result = ConvergenceResult(
        points=(), n_star=None, duration_seconds=3.0
    )

    result = convergence_module.converge(convergence_settings)

    assert result.duration_seconds == 3.0


def test_converge_wraps_unexpected_exception_as_workflow_error(
    convergence_settings: ConvergenceJobConfig,
) -> None:
    _StubConvergenceOrchestrator.exception = RuntimeError("boom")

    with pytest.raises(WorkflowError, match="Convergence study failed"):
        convergence_module.converge(convergence_settings)


def test_converge_reraises_workflow_error_unchanged(
    convergence_settings: ConvergenceJobConfig,
) -> None:
    original = WorkflowError("sweep-specific failure", {"stage": "size=8"})
    _StubConvergenceOrchestrator.exception = original

    with pytest.raises(WorkflowError) as exc_info:
        convergence_module.converge(convergence_settings)

    assert exc_info.value is original


def test_converge_reraises_type_mismatch_workflow_error_unwrapped(
    train_job_configs: tuple[TrainingJobConfig, TrainingJobConfig],
) -> None:
    """A settings object that isn't ``ConvergenceJobConfig`` fails converge()'s
    own type-check with a ``WorkflowError`` — asserting on its message (rather
    than ``ConvergenceOrchestrator``'s "Convergence study failed" wrapper
    message) confirms ``EntrypointContext.run()`` re-raised it unchanged."""
    training_settings, _ = train_job_configs

    with pytest.raises(WorkflowError, match="requires ConvergenceJobConfig"):
        convergence_module.converge(training_settings)  # ty: ignore[invalid-argument-type]
