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
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import Tensor

from dlkit.common.hooks import ParamValue
from dlkit.common.metric_stages import MetricStage
from dlkit.engine.adapters.lightning.base import ProcessingLightningWrapper
from dlkit.engine.artifacts import (
    ArtifactPolicy,
    FileArtifactPayload,
    ProducedArtifact,
    RuntimeArtifactManifest,
    attach_checkpoint_artifacts,
)
from dlkit.engine.tracking.artifact_logger import (
    CHECKPOINT_ARTIFACT_DIR,
    DEFAULT_MODEL_ARTIFACT_PATH,
    TAG_LOGGED_MODEL_ARTIFACT_PATH,
    TAG_LOGGED_MODEL_URI,
    TAG_MODEL_CLASS,
    ArtifactLogger,
    _build_input_example,
    _build_pt2_signature,
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

    def _run_step(
        self, batch: Any, batch_idx: int, stage: MetricStage
    ) -> tuple[Tensor, int | None, Any]:
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
        self.logged_artifact_calls: list[tuple[Path, str]] = []
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
        self.logged_artifact_calls.append((artifact_path, artifact_dir))

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
        model_serialization_format: str | None = None,
    ) -> str | None:
        self.logged_model_calls.append(
            {
                "artifact_path": artifact_path,
                "registered_model_name": registered_model_name,
                "input_example": input_example,
                "model_serialization_format": model_serialization_format,
            }
        )
        return f"runs:/{self._run_id}/{artifact_path}"


def _build_components(
    model: Any,
    trainer: Any = None,
    artifacts: RuntimeArtifactManifest | None = None,
) -> RuntimeComponents:
    return RuntimeComponents(
        model=model,
        datamodule=Mock(),
        trainer=trainer,
        meta={},
        artifacts=artifacts if artifacts is not None else RuntimeArtifactManifest(),
    )


@pytest.fixture
def job_config() -> JobConfig:
    return JobConfig(
        run=RunSettings(type="train"),
        experiment=ExperimentSettings(),
        tracking=TrackingSettings(backend="mlflow"),
    )


@pytest.fixture
def pt2_job_config() -> JobConfig:
    return JobConfig(
        run=RunSettings(type="train"),
        experiment=ExperimentSettings(),
        tracking=TrackingSettings(backend="mlflow", model_serialization_format="pt2"),
    )


def test_logs_model_artifact_without_registry(job_config: JobConfig) -> None:
    @dataclass(frozen=True, slots=True)
    class FancyNet:
        pass

    artifact_logger = ArtifactLogger(tracker=Mock())
    run_context = _RecordingRunContext()
    components = _build_components(FancyNet())

    artifact_logger._log_model_artifact(
        run_context=run_context, model=components.model, settings=job_config
    )

    assert len(run_context.logged_model_calls) == 1
    assert run_context.logged_model_calls[0]["artifact_path"] == DEFAULT_MODEL_ARTIFACT_PATH
    assert run_context.logged_model_calls[0]["registered_model_name"] is None
    assert run_context.logged_model_calls[0]["model_serialization_format"] == "pickle"
    assert run_context.tags[TAG_LOGGED_MODEL_URI] == f"runs:/test-run/{DEFAULT_MODEL_ARTIFACT_PATH}"
    assert run_context.tags[TAG_LOGGED_MODEL_ARTIFACT_PATH] == DEFAULT_MODEL_ARTIFACT_PATH
    assert run_context.tags[TAG_MODEL_CLASS] == "FancyNet"


def test_logs_model_uses_inner_class_name_for_wrapper(
    wrapped_model: _ConcreteWrapper,
    job_config: JobConfig,
) -> None:
    artifact_logger = ArtifactLogger(tracker=Mock())
    run_context = _RecordingRunContext()

    artifact_logger._log_model_artifact(
        run_context=run_context, model=wrapped_model, settings=job_config
    )

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
    experiment_id = mlflow.create_experiment(
        "pt2-test", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.pytorch.log_model(
            pytorch_model=model_with_shapes,
            name="model",
            serialization_format="pt2",
            input_example=_build_input_example(model_with_shapes),
            signature=_build_pt2_signature(model_with_shapes),
        )


def test_log_model_artifact_passes_input_example(
    model_with_shapes: nn.Module,
    job_config: JobConfig,
) -> None:
    """Reproduction: _log_model_artifact must pass a non-None input_example when shapes exist."""
    artifact_logger = ArtifactLogger(tracker=Mock())
    run_context = _RecordingRunContext()

    artifact_logger._log_model_artifact(
        run_context=run_context, model=model_with_shapes, settings=job_config
    )

    call = run_context.logged_model_calls[0]
    assert call["input_example"] is not None
    assert isinstance(call["input_example"], np.ndarray)
    assert call["input_example"].shape == (1, 4)


def test_log_model_artifact_multi_input_returns_tuple(
    model_with_multi_shapes: nn.Module,
    job_config: JobConfig,
) -> None:
    """Multi-input models produce a tuple of arrays, one per input shape."""
    artifact_logger = ArtifactLogger(tracker=Mock())
    run_context = _RecordingRunContext()

    artifact_logger._log_model_artifact(
        run_context=run_context, model=model_with_multi_shapes, settings=job_config
    )

    call = run_context.logged_model_calls[0]
    result = call["input_example"]
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0].shape == (1, 4)
    assert result[1].shape == (1, 8)


def test_build_pt2_signature_omits_name_for_single_input(model_with_shapes: nn.Module) -> None:
    """Regression: a named single-input TensorSpec breaks MLflow pyfunc predict.

    A named schema forces MLflow's schema enforcement to wrap the ndarray into
    a dict, which the pytorch flavor's pyfunc wrapper rejects outright.
    """
    signature = _build_pt2_signature(model_with_shapes)

    assert signature is not None
    (spec,) = signature.inputs.inputs
    assert spec.name is None


def test_build_pt2_signature_keeps_names_for_multi_input(
    model_with_multi_shapes: nn.Module,
) -> None:
    """Multi-input models still need named specs to disambiguate inputs."""
    signature = _build_pt2_signature(model_with_multi_shapes)

    assert signature is not None
    names = [spec.name for spec in signature.inputs.inputs]
    assert names == ["x", "y"]


def test_log_model_artifact_multi_input_logs_pyfunc_warning(
    model_with_multi_shapes: nn.Module,
    job_config: JobConfig,
) -> None:
    """Multi-input pyfunc/REST serving is unsupported by MLflow's pytorch flavor.

    MLflow itself fails silently for this case (no warning at all — see
    `test_mlflow_run_context.py`), so this must be surfaced explicitly instead
    of leaving it undiagnosable.
    """
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        artifact_logger = ArtifactLogger(tracker=Mock())
        run_context = _RecordingRunContext()
        artifact_logger._log_model_artifact(
            run_context=run_context, model=model_with_multi_shapes, settings=job_config
        )
    finally:
        logger.remove(sink_id)

    assert any("does not support pyfunc/REST serving for multi-input models" in m for m in messages)


def test_log_model_artifact_none_example_when_no_metadata(
    model_without_metadata: nn.Module,
    job_config: JobConfig,
) -> None:
    """Regression: gracefully pass input_example=None when the model has no shape metadata."""
    artifact_logger = ArtifactLogger(tracker=Mock())
    run_context = _RecordingRunContext()

    artifact_logger._log_model_artifact(
        run_context=run_context, model=model_without_metadata, settings=job_config
    )

    assert run_context.logged_model_calls[0]["input_example"] is None


def test_log_model_artifact_pt2_passes_serialization_format(
    model_with_shapes: nn.Module,
    pt2_job_config: JobConfig,
) -> None:
    artifact_logger = ArtifactLogger(tracker=Mock())
    run_context = _RecordingRunContext()

    artifact_logger._log_model_artifact(
        run_context=run_context, model=model_with_shapes, settings=pt2_job_config
    )

    assert run_context.logged_model_calls[0]["model_serialization_format"] == "pt2"
    assert run_context.logged_model_calls[0]["input_example"] is not None


def test_log_model_artifact_pt2_requires_shape_metadata(
    model_without_metadata: nn.Module,
    pt2_job_config: JobConfig,
) -> None:
    artifact_logger = ArtifactLogger(tracker=Mock())
    run_context = _RecordingRunContext()

    with pytest.raises(ValueError, match="PT2 model serialization requires checkpoint metadata"):
        artifact_logger._log_model_artifact(
            run_context=run_context, model=model_without_metadata, settings=pt2_job_config
        )


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


# ---------------------------------------------------------------------------
# log_checkpoints
# ---------------------------------------------------------------------------


def _mock_trainer(callbacks: list[object], checkpoint_callback: object | None = None) -> Mock:
    trainer = Mock()
    trainer.callbacks = callbacks
    trainer.checkpoint_callback = checkpoint_callback
    return trainer


def test_log_checkpoints_uploads_best_and_last_and_removes_when_policy_allows(
    tmp_path: Path, job_config: JobConfig
) -> None:
    """``ArtifactPolicy.remove_uploaded_files=True`` (a genuinely remote, tracked
    backend) removes the local copy after a successful upload."""
    best_path = tmp_path / "best.ckpt"
    last_path = tmp_path / "last.ckpt"
    best_path.write_bytes(b"best")
    last_path.write_bytes(b"last")

    checkpoint_cb = ModelCheckpoint()
    checkpoint_cb.best_model_path = str(best_path)
    checkpoint_cb.last_model_path = str(last_path)
    trainer = _mock_trainer(callbacks=[checkpoint_cb], checkpoint_callback=checkpoint_cb)

    artifact_logger = ArtifactLogger(tracker=Mock())
    run_context = _RecordingRunContext()
    components = _build_components(
        model=object(),
        trainer=trainer,
        artifacts=RuntimeArtifactManifest(policy=ArtifactPolicy(remove_uploaded_files=True)),
    )

    artifact_logger.log_checkpoints(components, run_context)

    uploaded = {path.name: artifact_dir for path, artifact_dir in run_context.logged_artifact_calls}
    assert uploaded == {
        "best.ckpt": CHECKPOINT_ARTIFACT_DIR,
        "last.ckpt": CHECKPOINT_ARTIFACT_DIR,
    }
    # Uploaded local copies are removed after upload when the policy allows it.
    assert not best_path.exists()
    assert not last_path.exists()


def test_log_checkpoints_keeps_local_files_when_policy_disallows_removal(
    tmp_path: Path, job_config: JobConfig
) -> None:
    """Default ``ArtifactPolicy.remove_uploaded_files=False`` (untracked or
    local-backend runs) keeps the local checkpoint on disk after upload, so
    ``TrainingResult.checkpoint_path`` still resolves."""
    best_path = tmp_path / "best.ckpt"
    last_path = tmp_path / "last.ckpt"
    best_path.write_bytes(b"best")
    last_path.write_bytes(b"last")

    checkpoint_cb = ModelCheckpoint()
    checkpoint_cb.best_model_path = str(best_path)
    checkpoint_cb.last_model_path = str(last_path)
    trainer = _mock_trainer(callbacks=[checkpoint_cb], checkpoint_callback=checkpoint_cb)

    artifact_logger = ArtifactLogger(tracker=Mock())
    run_context = _RecordingRunContext()
    components = _build_components(model=object(), trainer=trainer)

    artifact_logger.log_checkpoints(components, run_context)

    uploaded = {path.name: artifact_dir for path, artifact_dir in run_context.logged_artifact_calls}
    assert uploaded == {
        "best.ckpt": CHECKPOINT_ARTIFACT_DIR,
        "last.ckpt": CHECKPOINT_ARTIFACT_DIR,
    }
    # Local copies are still uploaded but not removed when the policy forbids it.
    assert best_path.exists()
    assert last_path.exists()


class _FailingUploadRunContext(_RecordingRunContext):
    """Run context whose log_artifact always raises, simulating a failed upload."""

    def log_artifact(self, artifact_path: Path, artifact_dir: str = "") -> None:
        raise RuntimeError("simulated upload failure")


def test_log_checkpoints_keeps_local_file_and_propagates_when_upload_fails(
    tmp_path: Path, job_config: JobConfig
) -> None:
    """A failed upload must not delete the local checkpoint, even when the
    policy would otherwise allow removal — deletion is only reachable after
    ``log_artifact`` succeeds, and the failure itself must propagate rather
    than being silently swallowed."""
    best_path = tmp_path / "best.ckpt"
    best_path.write_bytes(b"best")

    checkpoint_cb = ModelCheckpoint()
    checkpoint_cb.best_model_path = str(best_path)
    checkpoint_cb.last_model_path = str(best_path)
    trainer = _mock_trainer(callbacks=[checkpoint_cb], checkpoint_callback=checkpoint_cb)

    artifact_logger = ArtifactLogger(tracker=Mock())
    run_context = _FailingUploadRunContext()
    components = _build_components(
        model=object(),
        trainer=trainer,
        artifacts=RuntimeArtifactManifest(policy=ArtifactPolicy(remove_uploaded_files=True)),
    )

    with pytest.raises(RuntimeError, match="simulated upload failure"):
        artifact_logger.log_checkpoints(components, run_context)

    assert best_path.exists()


def test_log_checkpoints_finds_callback_via_callbacks_fallback(
    tmp_path: Path, job_config: JobConfig
) -> None:
    """Pins the fix for the previously-broken fallback: trainer.checkpoint_callback
    unset -> must still find a real ModelCheckpoint via trainer.callbacks."""
    best_path = tmp_path / "best.ckpt"
    best_path.write_bytes(b"best")

    checkpoint_cb = ModelCheckpoint()
    checkpoint_cb.best_model_path = str(best_path)
    trainer = _mock_trainer(callbacks=[checkpoint_cb], checkpoint_callback=None)

    artifact_logger = ArtifactLogger(tracker=Mock())
    run_context = _RecordingRunContext()
    components = _build_components(model=object(), trainer=trainer)

    artifact_logger.log_checkpoints(components, run_context)

    assert len(run_context.logged_artifact_calls) == 1
    assert run_context.logged_artifact_calls[0][0].name == "best.ckpt"


def test_log_checkpoints_is_noop_with_no_trainer(job_config: JobConfig) -> None:
    artifact_logger = ArtifactLogger(tracker=Mock())
    run_context = _RecordingRunContext()
    components = _build_components(model=object(), trainer=None)

    artifact_logger.log_checkpoints(components, run_context)

    assert run_context.logged_artifact_calls == []


def test_log_checkpoints_publishes_artifacts_attached_to_model_with_no_trainer(
    tmp_path: Path, job_config: JobConfig
) -> None:
    """One-shot-fit path: no Trainer, but a checkpoint artifact attached to the
    live model instance via ``attach_checkpoint_artifacts`` must still be
    published (see ``OneShotFitExecutor``)."""
    ckpt_path = tmp_path / "fitted.ckpt"
    ckpt_path.write_bytes(b"fitted")

    class _FittedModel:
        pass

    model = _FittedModel()
    attach_checkpoint_artifacts(
        model,
        (
            ProducedArtifact(
                kind="checkpoint",
                artifact_path=f"{CHECKPOINT_ARTIFACT_DIR}/fitted.ckpt",
                payload=FileArtifactPayload(file_path=ckpt_path),
            ),
        ),
    )

    artifact_logger = ArtifactLogger(tracker=Mock())
    run_context = _RecordingRunContext()
    components = _build_components(model=model, trainer=None)

    artifact_logger.log_checkpoints(components, run_context)

    assert run_context.logged_artifact_calls == [(ckpt_path, CHECKPOINT_ARTIFACT_DIR)]
    # Default ArtifactPolicy.remove_uploaded_files=False keeps the local file.
    assert ckpt_path.exists()


def test_log_checkpoints_removes_one_shot_fit_checkpoint_when_policy_allows(
    tmp_path: Path, job_config: JobConfig
) -> None:
    """One-shot-fit checkpoints follow the same remove-after-upload policy as
    trainer checkpoints (regression guard: this used to be trainer-only)."""
    ckpt_path = tmp_path / "fitted.ckpt"
    ckpt_path.write_bytes(b"fitted")

    class _FittedModel:
        pass

    model = _FittedModel()
    attach_checkpoint_artifacts(
        model,
        (
            ProducedArtifact(
                kind="checkpoint",
                artifact_path=f"{CHECKPOINT_ARTIFACT_DIR}/fitted.ckpt",
                payload=FileArtifactPayload(file_path=ckpt_path),
            ),
        ),
    )

    artifact_logger = ArtifactLogger(tracker=Mock())
    run_context = _RecordingRunContext()
    components = _build_components(
        model=model,
        trainer=None,
        artifacts=RuntimeArtifactManifest(policy=ArtifactPolicy(remove_uploaded_files=True)),
    )

    artifact_logger.log_checkpoints(components, run_context)

    assert run_context.logged_artifact_calls == [(ckpt_path, CHECKPOINT_ARTIFACT_DIR)]
    assert not ckpt_path.exists()


def test_log_checkpoints_is_noop_with_no_checkpoint_callback(job_config: JobConfig) -> None:
    trainer = _mock_trainer(callbacks=[], checkpoint_callback=None)

    artifact_logger = ArtifactLogger(tracker=Mock())
    run_context = _RecordingRunContext()
    components = _build_components(model=object(), trainer=trainer)

    artifact_logger.log_checkpoints(components, run_context)

    assert run_context.logged_artifact_calls == []
