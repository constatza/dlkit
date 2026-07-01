"""Tests for ArtifactLogger model logging behavior."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch import Tensor

from dlkit.common.hooks import ParamValue
from dlkit.engine.adapters.lightning.base import ProcessingLightningWrapper
from dlkit.engine.tracking.artifact_logger import (
    DEFAULT_MODEL_ARTIFACT_PATH,
    TAG_LOGGED_MODEL_ARTIFACT_PATH,
    TAG_LOGGED_MODEL_URI,
    TAG_MODEL_CLASS,
    ArtifactLogger,
    _build_input_example,
    _resolve_model_class_name,
)
from dlkit.engine.tracking.interfaces import IRunContext
from dlkit.engine.training.components import RuntimeComponents
from dlkit.infrastructure.config.experiment_settings import ExperimentSettings
from dlkit.infrastructure.config.job_config import JobConfig
from dlkit.infrastructure.config.run_settings import RunSettings
from dlkit.infrastructure.config.tracking_settings import TrackingSettings


class _InnerNet:
    """Stub inner model with a distinctive class name."""


class _ConcreteWrapper(ProcessingLightningWrapper):
    """Minimal concrete wrapper used only to satisfy isinstance checks."""

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        raise NotImplementedError

    def _run_step(self, batch: Any, batch_idx: int, stage: str) -> tuple[Tensor, int | None, Any]:
        raise NotImplementedError


@pytest.fixture
def wrapped_model() -> _ConcreteWrapper:
    """A ProcessingLightningWrapper instance whose inner model is _InnerNet."""
    wrapper: _ConcreteWrapper = object.__new__(_ConcreteWrapper)
    wrapper.model = cast("Any", _InnerNet())
    return wrapper


class _RecordingRunContext(IRunContext):
    """Recording run context for artifact-logging assertions."""

    def __init__(self):
        self._run_id = "test-run"
        self.logged_model_calls: list[dict[str, Any]] = []
        self.tags: dict[str, str] = {}

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def experiment_id(self) -> str | None:
        return "artifact-experiment"

    @property
    def tracking_uri(self) -> str | None:
        return "sqlite:///tmp/mlflow.db"

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        pass

    def log_params(self, params: Mapping[str, ParamValue]) -> None:
        pass

    def log_artifact_content(self, content: str | bytes, artifact_file: str) -> None:
        pass

    def log_artifact(self, artifact_path: Path, artifact_dir: str = "") -> None:
        pass

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value

    def log_dataset(
        self,
        dataset: Any,
        context: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        pass

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        *,
        registered_model_name: str | None = None,
        signature: Any | None = None,
        input_example: Any | None = None,
    ) -> str | None:
        self.logged_model_calls.append(
            {
                "artifact_path": artifact_path,
                "registered_model_name": registered_model_name,
                "input_example": input_example,
            }
        )
        return f"runs:/{self._run_id}/{artifact_path}"


def _build_components(model: Any) -> RuntimeComponents:
    return RuntimeComponents(
        model=model,
        datamodule=Mock(),
        trainer=None,
        meta={},
    )


@pytest.fixture
def job_config() -> JobConfig:
    return JobConfig(
        run=RunSettings(type="train"),
        experiment=ExperimentSettings(),
        tracking=TrackingSettings(backend="mlflow"),
    )


def test_logs_model_artifact_without_registry(job_config: JobConfig) -> None:
    @dataclass(frozen=True, slots=True)
    class FancyNet:
        pass

    artifact_logger = ArtifactLogger(tracker=Mock())
    run_context = _RecordingRunContext()
    components = _build_components(FancyNet())

    artifact_logger._log_model_artifact(run_context=run_context, model=components.model)

    assert len(run_context.logged_model_calls) == 1
    assert run_context.logged_model_calls[0]["artifact_path"] == DEFAULT_MODEL_ARTIFACT_PATH
    assert run_context.logged_model_calls[0]["registered_model_name"] is None
    assert run_context.tags[TAG_LOGGED_MODEL_URI] == f"runs:/test-run/{DEFAULT_MODEL_ARTIFACT_PATH}"
    assert run_context.tags[TAG_LOGGED_MODEL_ARTIFACT_PATH] == DEFAULT_MODEL_ARTIFACT_PATH
    assert run_context.tags[TAG_MODEL_CLASS] == "FancyNet"


def test_logs_model_uses_inner_class_name_for_wrapper(
    wrapped_model: _ConcreteWrapper,
) -> None:
    artifact_logger = ArtifactLogger(tracker=Mock())
    run_context = _RecordingRunContext()

    artifact_logger._log_model_artifact(run_context=run_context, model=wrapped_model)

    assert run_context.tags[TAG_MODEL_CLASS] == "_InnerNet"


# ---------------------------------------------------------------------------
# _build_input_example / input_example threading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ShapeContext:
    input_shapes: Mapping[str, tuple[int, ...]]


@dataclass(frozen=True)
class _FakeCheckpointMetadata:
    context: _ShapeContext


@pytest.fixture
def model_with_shapes() -> nn.Module:
    model = nn.Linear(4, 2)
    object.__setattr__(
        model,
        "_checkpoint_metadata",
        _FakeCheckpointMetadata(context=_ShapeContext(input_shapes={"x": (4,)})),
    )
    return model


@pytest.fixture
def model_without_metadata() -> nn.Module:
    return nn.Linear(4, 2)


@pytest.fixture
def model_with_multi_shapes() -> nn.Module:
    model = nn.Linear(4, 2)
    object.__setattr__(
        model,
        "_checkpoint_metadata",
        _FakeCheckpointMetadata(context=_ShapeContext(input_shapes={"x": (4,), "y": (8,)})),
    )
    return model


def test_build_input_example_is_valid_for_pt2_format(model_with_shapes: nn.Module) -> None:
    """Regression: _build_input_example output must pass MLflow pt2 iteration check."""
    result = _build_input_example(model_with_shapes)
    assert result is not None
    # reproduce exact MLflow pt2 validation (mlflow.pytorch.save_model lines 178-188)
    if isinstance(result, (np.ndarray, torch.Tensor)):
        result = (result,)
    assert all(isinstance(v, (np.ndarray, torch.Tensor)) for v in result), (
        "MLflow pt2 validation rejects this format — must be ndarray/Tensor, not dict"
    )


def test_mlflow_log_model_pt2_does_not_raise(model_with_shapes: nn.Module, tmp_path: Path) -> None:
    """Reproduction: mlflow.pytorch.log_model with serialization_format='pt2' must not raise."""
    import mlflow

    from dlkit.engine.tracking.artifact_logger import _build_pt2_signature

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment_id = mlflow.create_experiment("pt2-test")
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.pytorch.log_model(
            pytorch_model=model_with_shapes,
            name="model",
            serialization_format="pt2",
            input_example=_build_input_example(model_with_shapes),
            signature=_build_pt2_signature(model_with_shapes),
        )


def test_log_model_artifact_passes_input_example(model_with_shapes: nn.Module) -> None:
    """Reproduction: _log_model_artifact must pass a non-None input_example when shapes exist."""
    artifact_logger = ArtifactLogger(tracker=Mock())
    run_context = _RecordingRunContext()

    artifact_logger._log_model_artifact(run_context=run_context, model=model_with_shapes)

    call = run_context.logged_model_calls[0]
    assert call["input_example"] is not None
    assert isinstance(call["input_example"], np.ndarray)
    assert call["input_example"].shape == (1, 4)


def test_log_model_artifact_multi_input_returns_tuple(
    model_with_multi_shapes: nn.Module,
) -> None:
    """Multi-input models produce a tuple of arrays, one per input shape."""
    artifact_logger = ArtifactLogger(tracker=Mock())
    run_context = _RecordingRunContext()

    artifact_logger._log_model_artifact(run_context=run_context, model=model_with_multi_shapes)

    call = run_context.logged_model_calls[0]
    result = call["input_example"]
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0].shape == (1, 4)
    assert result[1].shape == (1, 8)


def test_log_model_artifact_none_example_when_no_metadata(
    model_without_metadata: nn.Module,
) -> None:
    """Regression: gracefully pass input_example=None when the model has no shape metadata."""
    artifact_logger = ArtifactLogger(tracker=Mock())
    run_context = _RecordingRunContext()

    artifact_logger._log_model_artifact(run_context=run_context, model=model_without_metadata)

    assert run_context.logged_model_calls[0]["input_example"] is None


# ---------------------------------------------------------------------------
# _resolve_model_class_name
# ---------------------------------------------------------------------------


def test_resolve_model_class_name_returns_plain_model_class() -> None:
    class _PlainModel:
        pass

    assert _resolve_model_class_name(_PlainModel()) == "_PlainModel"


def test_resolve_model_class_name_unwraps_processing_lightning_wrapper(
    wrapped_model: _ConcreteWrapper,
) -> None:
    assert _resolve_model_class_name(wrapped_model) == "_InnerNet"
