"""Tests for the evaluate() entrypoint's shared ``EntrypointContext.run()`` plumbing.

``evaluate()`` is one of five entrypoints (``fit``, ``training``,
``optimization``, ``convergence``, ``evaluate``) that route their workflow
execution through ``EntrypointContext.run()``: a raw exception from checkpoint
loading/evaluation is wrapped as ``WorkflowError``, while the pre-existing
``settings.data.targets`` validation still raises ``ConfigurationError``
unwrapped (it isn't routed through ``context.run()`` — bad input, not a
workflow failure). These tests stub out model loading, datamodule
construction, and evaluation instead of exercising a real checkpoint,
mirroring ``test_fit.py``'s pattern of standing in for heavy collaborators.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from dlkit.common import ConfigurationError, EvaluationResult
from dlkit.common.errors import WorkflowError
from dlkit.infrastructure.config.job_config import InferenceJobConfig

# `dlkit.engine.workflows.entrypoints`'s `__init__.py` does
# `from .evaluate import evaluate`, which rebinds the package's `evaluate`
# attribute from the submodule to the function of the same name. Importing
# via `importlib` reads straight from `sys.modules` instead of resolving
# through that shadowed package attribute, so this reliably gets the module.
evaluate_module = importlib.import_module("dlkit.engine.workflows.entrypoints.evaluate")


class _StubPredictor:
    """Stands in for the real predictor: ``unload()`` is the only method
    ``evaluate()`` calls on it directly."""

    def unload(self) -> None:
        pass


def _stub_load_model_from_settings(
    settings: object,
    *,
    checkpoint_path: object,
    device: object,
    batch_size: object,
    apply_transforms: object,
) -> _StubPredictor:
    del settings, checkpoint_path, device, batch_size, apply_transforms
    return _StubPredictor()


def _stub_build_inference_datamodule(settings: object, *, checkpoint_override: object) -> object:
    del settings, checkpoint_override
    return object()


@pytest.fixture(autouse=True)
def stub_model_and_data_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluate_module, "load_model_from_settings", _stub_load_model_from_settings)
    monkeypatch.setattr(
        evaluate_module, "build_inference_datamodule", _stub_build_inference_datamodule
    )


@pytest.fixture
def evaluation_settings(minimal_dataset: dict[str, Path]) -> InferenceJobConfig:
    return InferenceJobConfig.model_validate(
        {
            "run": {"type": "predict", "seed": 0},
            "model": {
                "class": "Dummy",
                "module_path": "dlkit.domain.nn",
                "checkpoint": str(minimal_dataset["features"]),
            },
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
        }
    )


def test_evaluate_returns_result_from_evaluate_checkpoint_on_success(
    monkeypatch: pytest.MonkeyPatch, evaluation_settings: InferenceJobConfig
) -> None:
    expected = EvaluationResult(
        predictions=None, targets=None, metrics={"mae": 0.1}, figures={}, duration_seconds=0.5
    )
    monkeypatch.setattr(
        evaluate_module,
        "evaluate_checkpoint",
        lambda predictor, datamodule, plots, *, split: expected,
    )

    result = evaluate_module.evaluate(evaluation_settings)

    assert result is expected


def test_evaluate_raises_configuration_error_unwrapped_without_targets(
    evaluation_settings: InferenceJobConfig,
) -> None:
    """The targets-required check runs before ``context.run()``, so it must
    stay a ``ConfigurationError`` rather than get wrapped as ``WorkflowError``."""
    no_targets_settings = evaluation_settings.patch({"data": {"targets": []}})

    with pytest.raises(ConfigurationError, match="settings.data.targets"):
        evaluate_module.evaluate(no_targets_settings)


def test_evaluate_wraps_unexpected_exception_as_workflow_error(
    monkeypatch: pytest.MonkeyPatch, evaluation_settings: InferenceJobConfig
) -> None:
    def _raise(predictor: object, datamodule: object, plots: object, *, split: object) -> None:
        del predictor, datamodule, plots, split
        raise RuntimeError("boom")

    monkeypatch.setattr(evaluate_module, "evaluate_checkpoint", _raise)

    with pytest.raises(WorkflowError, match="Evaluation failed"):
        evaluate_module.evaluate(evaluation_settings)


def test_evaluate_reraises_workflow_error_unchanged(
    monkeypatch: pytest.MonkeyPatch, evaluation_settings: InferenceJobConfig
) -> None:
    original = WorkflowError("checkpoint-specific failure", {"stage": "predict"})

    def _raise(predictor: object, datamodule: object, plots: object, *, split: object) -> None:
        del predictor, datamodule, plots, split
        raise original

    monkeypatch.setattr(evaluate_module, "evaluate_checkpoint", _raise)

    with pytest.raises(WorkflowError) as exc_info:
        evaluate_module.evaluate(evaluation_settings)

    assert exc_info.value is original
