"""End-to-end test: train a tiny model, then evaluate its checkpoint without training.

Verifies the eval-only API (checkpoint + labeled test split -> stats/plots)
produces the same kind of regression metrics/figures that training produces,
using a real checkpoint (with genuine ``predict_target_key``/``feature_names``
metadata from the checkpoint serializer) rather than a hand-built one.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dlkit.common import ConfigurationError, EvaluationResult, TrainingResult
from dlkit.common.checkpoint_source import LatestRunCheckpoint, RunCheckpoint
from dlkit.common.hooks import LifecycleHooks, RunCreatedEvent
from dlkit.infrastructure.config.job_config import InferenceJobConfig, TrainingJobConfig
from dlkit.infrastructure.config.tracking_settings import TrackingSettings
from dlkit.interfaces.api.domain.override_types import EvaluationOverrides
from dlkit.interfaces.api.functions import train as api_train
from dlkit.interfaces.inference import evaluate as api_evaluate

# Same model/data shape as tests/integration/conftest.py's _make_training_job_config,
# since InferenceJobConfig must describe the identical model + data the checkpoint was
# trained with (component "name"/"class" is write-only in the settings models, so it
# can't be recovered by dumping a validated TrainingJobConfig back out).
FEATURE_SIZE = 4
TARGET_SIZE = 2


def _build_inference_settings(
    minimal_dataset: dict[str, Path], checkpoint: Path | str | None = None
) -> InferenceJobConfig:
    """Build an InferenceJobConfig matching the FFNN trained by `training_settings`.

    `checkpoint` seeds `settings.model.checkpoint`, which `InferenceJobConfig`
    requires to be non-None even when the caller intends to resolve the real
    checkpoint via `run_checkpoint` instead: the resolved run-based path
    always takes precedence over `settings.model.checkpoint` inside
    `evaluate()`, so an unused placeholder is fine when `checkpoint` is
    omitted.
    """
    return InferenceJobConfig.model_validate(
        {
            "run": {"type": "predict"},
            "experiment": {"name": "integration_test"},
            "model": {
                "class": "FFNN",
                "module_path": "dlkit.domain.nn",
                "hidden_size": FEATURE_SIZE,
                "num_layers": 0,
                "checkpoint": str(checkpoint)
                if checkpoint is not None
                else "unused-placeholder.ckpt",
            },
            "data": {
                "class": "FlexibleDataset",
                "module_path": "dlkit.engine.data.datasets",
                "batch_size": 4,
                "num_workers": 0,
                "shuffle": False,
                "pin_memory": False,
                "persistent_workers": False,
                "features": [
                    {"name": "x", "path": str(minimal_dataset["features"]), "format": "npy"}
                ],
                "targets": [
                    {"name": "y", "path": str(minimal_dataset["targets"]), "format": "npy"}
                ],
            },
        }
    )


@pytest.fixture
def training_settings_with_checkpoint(
    training_settings: TrainingJobConfig, tmp_path: Path
) -> TrainingJobConfig:
    """Enable real checkpointing — the shared fixture disables it for speed."""
    training_cfg = training_settings.training
    assert training_cfg is not None
    trainer_cfg = training_cfg.trainer
    assert trainer_cfg is not None
    return training_settings.model_copy(
        update={
            "training": training_cfg.model_copy(
                update={
                    "trainer": trainer_cfg.model_copy(
                        update={
                            "enable_checkpointing": True,
                            "default_root_dir": tmp_path / "training_output",
                            "fast_dev_run": False,
                        }
                    )
                }
            )
        }
    )


@pytest.fixture
def training_settings_with_checkpoint_and_mlflow(
    training_settings_with_checkpoint: TrainingJobConfig, tmp_path: Path
) -> TrainingJobConfig:
    """Compose real checkpointing with a real local MLflow tracking backend.

    Run-based checkpoint selection needs an actual queryable MLflow run to
    resolve against, not just a checkpoint file sitting on local disk — this
    layers ``mlflow_settings``'s sqlite-tracking-backend pattern on top of
    ``training_settings_with_checkpoint``'s real (non ``fast_dev_run``)
    checkpointing.
    """
    mlruns_dir = tmp_path / "run_checkpoint_mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow_uri = f"sqlite:///{(mlruns_dir / 'mlflow.db').as_posix()}"
    return training_settings_with_checkpoint.model_copy(
        update={"tracking": TrackingSettings(backend="mlflow", uri=mlflow_uri)}
    )


@pytest.fixture
def trained_result(training_settings_with_checkpoint: TrainingJobConfig) -> TrainingResult:
    return api_train(training_settings_with_checkpoint)


def _real_checkpoint_path(result: TrainingResult) -> Path:
    """The checkpoint file that actually exists on disk after training.

    ``TrainingResult.checkpoint_path`` prefers a "best_checkpoint" artifact
    entry that Lightning's post-fit reload/test calls can leave stale (it
    reports a path last written mid-fit, while the final on-disk file ends up
    named ``last.ckpt``) — a pre-existing training-pipeline quirk orthogonal
    to eval, so this just picks whichever recorded path is real.
    """
    checkpoint = result.checkpoint_path
    if checkpoint is not None and checkpoint.exists():
        return checkpoint
    fallback = result.artifacts.get("last_checkpoint")
    assert fallback is not None and fallback.exists(), (
        f"Neither checkpoint_path={checkpoint} nor last_checkpoint artifact exist on disk"
    )
    return fallback


@pytest.fixture
def trained_checkpoint_path(trained_result: TrainingResult) -> Path:
    """The checkpoint file that actually exists on disk after training."""
    return _real_checkpoint_path(trained_result)


def _split_filepath(training_settings: TrainingJobConfig) -> Path:
    """The single `splits/*.json` file training persisted for this run.

    A `run_checkpoint`-resolved checkpoint is downloaded to an arbitrary temp
    directory, not colocated next to a `splits/` directory the way a local
    training-output checkpoint is — so `evaluate()`'s default colocated-split
    auto-location (`DatasetBuilder._resolve_colocated_split_filepath`) can't
    apply there. Real callers evaluating a run-based checkpoint must set
    `data.splits.filepath` explicitly; these tests do the same.
    """
    training_cfg = training_settings.training
    assert training_cfg is not None
    trainer_cfg = training_cfg.trainer
    assert trainer_cfg is not None
    split_files = list(Path(trainer_cfg.default_root_dir).glob("splits/*.json"))
    assert len(split_files) == 1, f"expected exactly one split file, found {split_files}"
    return split_files[0]


def _with_split_filepath(settings: InferenceJobConfig, split_filepath: Path) -> InferenceJobConfig:
    """Point `settings.data.splits.filepath` at an explicit split file."""
    assert settings.data is not None
    return settings.model_copy(
        update={
            "data": settings.data.model_copy(
                update={
                    "splits": settings.data.splits.model_copy(update={"filepath": split_filepath})
                }
            )
        }
    )


def test_evaluate_produces_metrics_and_figures_for_a_trained_checkpoint(
    minimal_dataset: dict[str, Path],
    trained_checkpoint_path: Path,
) -> None:
    inference_settings = _build_inference_settings(minimal_dataset, trained_checkpoint_path)

    result = api_evaluate(inference_settings)

    try:
        assert isinstance(result, EvaluationResult)
        assert result.metrics.keys() == {"mae", "rmse", "r2"}
        assert all(isinstance(v, float) for v in result.metrics.values())
        assert result.predictions.shape[0] == result.targets.shape[0] > 0
        assert set(result.figures) == {
            "parity_plot",
            "residual_plot",
            "error_histogram",
            "residual_vs_index",
        }
        assert result.mlflow_run_id is None
    finally:
        for fig in result.figures.values():
            plt.close(fig)


def test_evaluate_fires_on_run_created_when_logging_to_mlflow(
    minimal_dataset: dict[str, Path],
    trained_checkpoint_path: Path,
    tmp_path: Path,
) -> None:
    """evaluate(log_to_mlflow=True, hooks=...) fires on_run_created atomically
    at run creation, mirroring TrackingDecorator's train()-time behavior —
    the extension point neuralls' eval-only workflow needs to nest its child
    run under a session parent without a post-hoc, close-then-tag patch.
    """
    mlruns_dir = tmp_path / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow_uri = f"sqlite:///{(mlruns_dir / 'mlflow.db').as_posix()}"

    inference_settings = _build_inference_settings(minimal_dataset, trained_checkpoint_path)
    inference_settings = inference_settings.model_copy(
        update={"tracking": TrackingSettings(backend="mlflow", uri=mlflow_uri)}
    )

    recorded: list[RunCreatedEvent] = []
    hooks = LifecycleHooks(on_run_created=recorded.append)

    result = api_evaluate(
        inference_settings,
        EvaluationOverrides(run_name="eval-run"),
        hooks=hooks,
    )

    try:
        assert result.mlflow_run_id is not None
        assert recorded == [
            RunCreatedEvent(
                run_id=result.mlflow_run_id,
                tracking_uri=result.mlflow_tracking_uri,
                kind="evaluate",
                is_outermost=True,
            )
        ]
    finally:
        for fig in result.figures.values():
            plt.close(fig)


def test_evaluate_raises_without_configured_targets(
    minimal_dataset: dict[str, Path],
    trained_checkpoint_path: Path,
) -> None:
    inference_settings = _build_inference_settings(minimal_dataset, trained_checkpoint_path)
    no_targets_settings = inference_settings.model_copy(
        update={"data": inference_settings.data.model_copy(update={"targets": ()})}
    )

    with pytest.raises(ConfigurationError, match="settings.data.targets"):
        api_evaluate(no_targets_settings)


def test_evaluate_resolves_explicit_run_checkpoint(
    minimal_dataset: dict[str, Path],
    training_settings_with_checkpoint_and_mlflow: TrainingJobConfig,
) -> None:
    """A `RunCheckpoint(run_id=...)` model.checkpoint downloads and evaluates that exact run."""
    trained = api_train(training_settings_with_checkpoint_and_mlflow)
    assert trained.mlflow_run_id is not None

    inference_settings = _build_inference_settings(minimal_dataset).model_copy(
        update={"tracking": training_settings_with_checkpoint_and_mlflow.tracking}
    )
    inference_settings = _with_split_filepath(
        inference_settings, _split_filepath(training_settings_with_checkpoint_and_mlflow)
    )
    inference_settings = inference_settings.patch(
        {"model": {"checkpoint": RunCheckpoint(run_id=trained.mlflow_run_id)}}
    )

    result = api_evaluate(inference_settings)

    try:
        assert isinstance(result, EvaluationResult)
        assert result.metrics.keys() == {"mae", "rmse", "r2"}
        assert all(isinstance(v, float) for v in result.metrics.values())
        assert result.predictions.shape[0] == result.targets.shape[0] > 0
    finally:
        for fig in result.figures.values():
            plt.close(fig)


def test_evaluate_resolves_latest_run_checkpoint(
    minimal_dataset: dict[str, Path],
    training_settings_with_checkpoint_and_mlflow: TrainingJobConfig,
    tmp_path: Path,
) -> None:
    """`run_checkpoint=LatestRunCheckpoint()` picks the later of two runs.

    Two runs are trained into the same experiment with different seeds (so
    their tiny, barely-trained FFNNs end up with genuinely different
    weights), the second started after the first. Evaluating via
    `LatestRunCheckpoint()` must reproduce the exact predictions of directly
    evaluating the second run's own checkpoint file, and must differ from
    directly evaluating the first run's.
    """
    first_settings = training_settings_with_checkpoint_and_mlflow
    training_cfg = first_settings.training
    assert training_cfg is not None
    trainer_cfg = training_cfg.trainer
    assert trainer_cfg is not None
    second_root = tmp_path / "training_output_second"
    second_root.mkdir(parents=True, exist_ok=True)
    second_settings = first_settings.model_copy(
        update={
            "run": first_settings.run.model_copy(update={"seed": 4242}),
            "training": training_cfg.model_copy(
                update={"trainer": trainer_cfg.model_copy(update={"default_root_dir": second_root})}
            ),
        }
    )

    first_run = api_train(first_settings)
    second_run = api_train(second_settings)
    assert first_run.mlflow_run_id is not None
    assert second_run.mlflow_run_id is not None
    assert first_run.mlflow_run_id != second_run.mlflow_run_id

    tracking = training_settings_with_checkpoint_and_mlflow.tracking
    latest_settings = _build_inference_settings(minimal_dataset).model_copy(
        update={"tracking": tracking}
    )
    # The latest run is `second_settings` (later start_time, different seed) —
    # its own split file must be used so `latest_result` and
    # `second_direct_result` evaluate against the identical test partition;
    # using `first_settings`'s split (a different seed => different
    # train/val/test partition of the same 20 samples) would make the
    # equality assertion below spuriously fail even with identical weights.
    latest_settings = _with_split_filepath(latest_settings, _split_filepath(second_settings))
    second_direct_settings = _build_inference_settings(
        minimal_dataset, _real_checkpoint_path(second_run)
    ).model_copy(update={"tracking": tracking})
    first_direct_settings = _build_inference_settings(
        minimal_dataset, _real_checkpoint_path(first_run)
    ).model_copy(update={"tracking": tracking})

    latest_settings = latest_settings.patch({"model": {"checkpoint": LatestRunCheckpoint()}})
    latest_result = api_evaluate(latest_settings)
    second_direct_result = api_evaluate(second_direct_settings)
    first_direct_result = api_evaluate(first_direct_settings)

    try:
        assert np.array_equal(latest_result.predictions, second_direct_result.predictions)
        assert not np.array_equal(latest_result.predictions, first_direct_result.predictions)
    finally:
        for result in (latest_result, second_direct_result, first_direct_result):
            for fig in result.figures.values():
                plt.close(fig)


def test_evaluate_checkpoint_field_accepts_either_a_path_or_a_run_checkpoint(
    minimal_dataset: dict[str, Path],
    trained_checkpoint_path: Path,
) -> None:
    """`model.checkpoint` is one field (`Path | str | CheckpointSource`), not two
    mutually exclusive kwargs — passing a `RunCheckpoint` simply replaces
    whatever path was there. "Conflicting checkpoint sources" is now
    structurally unreachable rather than a runtime validation error.
    """
    inference_settings = _build_inference_settings(minimal_dataset, trained_checkpoint_path)

    repatched = inference_settings.patch(
        {"model": {"checkpoint": RunCheckpoint(run_id="does-not-matter")}}
    )
    assert repatched.model.checkpoint == RunCheckpoint(run_id="does-not-matter")
