"""End-to-end test: train a tiny model, then evaluate its checkpoint without training.

Verifies the eval-only API (checkpoint + labeled test split -> stats/plots)
produces the same kind of regression metrics/figures that training produces,
using a real checkpoint (with genuine ``predict_target_key``/``feature_names``
metadata from the checkpoint serializer) rather than a hand-built one.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from dlkit.common import EvaluationResult, TrainingResult
from dlkit.infrastructure.config.job_config import InferenceJobConfig, TrainingJobConfig
from dlkit.interfaces.api.functions import train as api_train
from dlkit.interfaces.inference import evaluate as api_evaluate

# Same model/data shape as tests/integration/conftest.py's _make_training_job_config,
# since InferenceJobConfig must describe the identical model + data the checkpoint was
# trained with (component "name"/"class" is write-only in the settings models, so it
# can't be recovered by dumping a validated TrainingJobConfig back out).
FEATURE_SIZE = 4
TARGET_SIZE = 2


def _build_inference_settings(
    minimal_dataset: dict[str, Path], checkpoint: Path
) -> InferenceJobConfig:
    """Build an InferenceJobConfig matching the FFNN trained by `training_settings`."""
    return InferenceJobConfig.model_validate(
        {
            "run": {"type": "predict"},
            "experiment": {"name": "integration_test"},
            "model": {
                "class": "FFNN",
                "module_path": "dlkit.domain.nn",
                "hidden_size": FEATURE_SIZE,
                "num_layers": 0,
                "checkpoint": str(checkpoint),
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
def trained_result(training_settings_with_checkpoint: TrainingJobConfig) -> TrainingResult:
    return api_train(training_settings_with_checkpoint)


@pytest.fixture
def trained_checkpoint_path(trained_result: TrainingResult) -> Path:
    """The checkpoint file that actually exists on disk after training.

    ``TrainingResult.checkpoint_path`` prefers a "best_checkpoint" artifact
    entry that Lightning's post-fit reload/test calls can leave stale (it
    reports a path last written mid-fit, while the final on-disk file ends up
    named ``last.ckpt``) — a pre-existing training-pipeline quirk orthogonal
    to eval, so this fixture just picks whichever recorded path is real.
    """
    checkpoint = trained_result.checkpoint_path
    if checkpoint is not None and checkpoint.exists():
        return checkpoint
    fallback = trained_result.artifacts.get("last_checkpoint")
    assert fallback is not None and fallback.exists(), (
        f"Neither checkpoint_path={checkpoint} nor last_checkpoint artifact exist on disk"
    )
    return fallback


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


def test_evaluate_raises_without_configured_targets(
    minimal_dataset: dict[str, Path],
    trained_checkpoint_path: Path,
) -> None:
    inference_settings = _build_inference_settings(minimal_dataset, trained_checkpoint_path)
    no_targets_settings = inference_settings.model_copy(
        update={"data": inference_settings.data.model_copy(update={"targets": ()})}
    )

    from dlkit.common import ConfigurationError

    with pytest.raises(ConfigurationError, match="settings.data.targets"):
        api_evaluate(no_targets_settings)
