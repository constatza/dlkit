"""Regression test proving eval-time R2 diverges from train-time R2.

Root cause (see the fix plan's evidence chain): ``evaluate()`` recomputes a
brand-new, **unseeded** random train/val/test split instead of reusing the
split that was actually used during training
(``src/dlkit/infrastructure/io/split_provider.py``'s ``get_or_create_split()``
builds a fresh ``RatioSplitStrategy`` -- backed by an unseeded
``torch.randperm`` -- whenever no ``explicit_filepath`` is configured, which
is the default on both the training and evaluation call sites). Training
happens to get a deterministic split only incidentally, because the whole
process is seeded once via ``seed_everything`` before the split is drawn;
``evaluate()`` seeds nothing at all.

This test trains a small FFNN via the real ``api_train()`` entrypoint with
checkpointing enabled and a fixed ``run.seed``, captures the training-time
test-split R2, then evaluates the resulting checkpoint through the real
``api_evaluate()`` API against a freshly-constructed ``InferenceJobConfig``
that shares no in-memory state with the training call (mirroring a separate
process/session reloading a checkpoint later). It asserts the two R2 values
stay close -- which they must, since both calls score the *same* trained
weights and, if the split were actually reused, would do so against the
*same* rows of the *same* dataset (a purely deterministic computation).

The synthetic dataset uses a purely linear target (``y = X @ W + noise``)
with no target transforms at all, so both R2 computations happen in the same
(raw) numeric space -- isolating the split-reproducibility bug from the
unrelated normalized-vs-raw-space R2 confound noted in the fix plan's final
review (item 6): training computes R2 in transformed space while evaluation
computes it in raw space, which would only matter for non-affine transforms.

This test is expected to stay in the suite permanently and flip from FAILING
to PASSING once the fix plan's Phase 1 split-resolution redesign lands.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dlkit.common import TrainingResult
from dlkit.infrastructure.config.job_config import InferenceJobConfig, TrainingJobConfig
from dlkit.interfaces.api.functions import train as api_train
from dlkit.interfaces.inference import evaluate as api_evaluate

# Dataset shape -- large enough that the test-split R2 is a stable estimate
# (not dominated by 2-3-sample sampling noise), small enough to train in a
# couple of seconds on CPU. Deliberately distinct from
# tests/integration/conftest.py's `minimal_dataset`, whose independent
# random features/binary targets have no learnable relationship and would
# make the test-split R2 an unstably noisy, near-degenerate signal.
NUM_SAMPLES: int = 120
FEATURE_SIZE: int = 4
TARGET_SIZE: int = 1
DATA_SEED: int = 7
NOISE_STD: float = 0.3

# Model/training shape -- one hidden layer gives the model enough capacity to
# fit the linear relationship. EPOCHS is deliberately 1 (matching
# tests/integration/conftest.py's convention): with more than one epoch, the
# best-checkpoint and last-checkpoint paths diverge and BOTH get deleted by
# the (separately tracked) checkpoint-deletion bug covered by
# tests/integration/test_checkpoint_artifact_lifecycle.py, which would block
# this test on an unrelated failure. With exactly one epoch, best == last, so
# only the (unused) best-checkpoint copy is deleted and last.ckpt -- which
# `checkpoint_path` falls back to below -- survives, keeping this test
# isolated to the split-reproducibility bug it actually targets.
HIDDEN_SIZE: int = 16
NUM_LAYERS: int = 1
EPOCHS: int = 1
BATCH_SIZE: int = 8
RUN_SEED: int = 42

# Training-time test-split R2 is logged under this key; see
# dlkit.common.metric_stages.metric_key(MetricStage.TEST, "R2Score") and
# engine.adapters.lightning.metrics_routing.RoutedMetricsUpdater.compute(),
# which keys computed metrics by the metric class's __name__.
TRAIN_TEST_R2_KEY: str = "test/R2Score"
# Evaluation-time R2 is reported under this key; see
# tests/integration/test_evaluate_integration.py's
# `result.metrics.keys() == {"mae", "rmse", "r2"}` assertion.
EVAL_R2_KEY: str = "r2"
R2_TOLERANCE: float = 0.05


@pytest.fixture
def linear_regression_dataset(tmp_path: Path) -> dict[str, Path]:
    """A synthetic, purely linear (affine, noise-perturbed) regression dataset.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Dictionary of paths to the generated feature/target ``.npy`` files.
    """
    rng = np.random.default_rng(DATA_SEED)
    features = rng.standard_normal((NUM_SAMPLES, FEATURE_SIZE)).astype(np.float32)
    true_weights = rng.standard_normal((FEATURE_SIZE, TARGET_SIZE)).astype(np.float32)
    noise = NOISE_STD * rng.standard_normal((NUM_SAMPLES, TARGET_SIZE)).astype(np.float32)
    targets = (features @ true_weights + noise).astype(np.float32)

    data_dir = tmp_path / "dataflow"
    data_dir.mkdir(parents=True, exist_ok=True)
    features_path = data_dir / "features.npy"
    targets_path = data_dir / "targets.npy"
    np.save(features_path, features)
    np.save(targets_path, targets)

    return {"features": features_path, "targets": targets_path, "data_dir": data_dir}


def _build_model_and_data_payload(
    dataset: dict[str, Path], *, checkpoint: Path | None = None
) -> dict[str, object]:
    """Shared model + data config fragment for both the training and eval configs.

    Args:
        dataset: Fixture-provided feature/target file paths.
        checkpoint: Optional checkpoint path to attach to the model config
            (set for the eval-side config, omitted for the training config).

    Returns:
        Partial settings payload with matching ``model``/``data`` sections so
        the trained checkpoint's weights load cleanly into the eval config's
        model.
    """
    model_payload: dict[str, object] = {
        "class": "FFNN",
        "module_path": "dlkit.domain.nn",
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
    }
    if checkpoint is not None:
        model_payload["checkpoint"] = str(checkpoint)

    return {
        "model": model_payload,
        "data": {
            "class": "FlexibleDataset",
            "module_path": "dlkit.engine.data.datasets",
            "batch_size": BATCH_SIZE,
            "num_workers": 0,
            "shuffle": True,
            "pin_memory": False,
            "persistent_workers": False,
            "features": [{"name": "x", "path": str(dataset["features"]), "format": "npy"}],
            "targets": [{"name": "y", "path": str(dataset["targets"]), "format": "npy"}],
        },
    }


@pytest.fixture
def regression_training_settings(
    linear_regression_dataset: dict[str, Path], tmp_path: Path
) -> TrainingJobConfig:
    """TrainingJobConfig for the linear regression dataset, checkpointing enabled.

    Mirrors ``tests/integration/test_evaluate_integration.py``'s
    ``training_settings_with_checkpoint`` (real, non-``fast_dev_run`` trainer,
    checkpointing on) composed with an R2Score metric so the training-time
    test-split R2 is captured in ``TrainingResult.metrics``.

    Args:
        linear_regression_dataset: Fixture providing dataset paths.
        tmp_path: Pytest temporary directory fixture.

    Returns:
        TrainingJobConfig ready for a real (checkpointed) training run.
    """
    output_dir = tmp_path / "training_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "run": {"type": "train", "seed": RUN_SEED},
        "experiment": {"name": "split_reproducibility_regression"},
        "training": {
            "loss": "mse",
            "trainer": {
                "fast_dev_run": False,
                "enable_checkpointing": True,
                "default_root_dir": str(output_dir),
                "accelerator": "cpu",
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "max_epochs": EPOCHS,
            },
            "optimizer": {"name": "AdamW", "lr": 1e-3},
            "metrics": [
                {"name": "MeanSquaredError", "module_path": "dlkit.domain.metrics"},
                {"name": "R2Score", "module_path": "dlkit.domain.metrics"},
            ],
        },
        **_build_model_and_data_payload(linear_regression_dataset),
    }
    return TrainingJobConfig.model_validate(payload)


@pytest.fixture
def regression_trained_result(
    regression_training_settings: TrainingJobConfig,
) -> TrainingResult:
    """Real ``api_train()`` result for the linear regression dataset.

    Args:
        regression_training_settings: Training config fixture.

    Returns:
        TrainingResult from the completed training run.
    """
    return api_train(regression_training_settings)


@pytest.fixture
def regression_checkpoint_path(regression_trained_result: TrainingResult) -> Path:
    """The checkpoint file that actually exists on disk after training.

    Mirrors ``test_evaluate_integration.py``'s ``trained_checkpoint_path``
    fallback: ``TrainingResult.checkpoint_path`` prefers a "best_checkpoint"
    artifact entry that Lightning's post-fit reload/test calls can leave
    stale, so this fixture picks whichever recorded path is actually real.

    Args:
        regression_trained_result: Completed TrainingResult fixture.

    Returns:
        Path to a checkpoint file confirmed to exist on disk.
    """
    checkpoint = regression_trained_result.checkpoint_path
    if checkpoint is not None and checkpoint.exists():
        return checkpoint
    fallback = regression_trained_result.artifacts.get("last_checkpoint")
    assert fallback is not None and fallback.exists(), (
        f"Neither checkpoint_path={checkpoint} nor last_checkpoint artifact exist on disk"
    )
    return fallback


def test_eval_time_r2_matches_train_time_r2_for_same_checkpoint(
    linear_regression_dataset: dict[str, Path],
    regression_trained_result: TrainingResult,
    regression_checkpoint_path: Path,
) -> None:
    """Eval-time test-split R2 should closely match training-time test-split R2.

    Both calls score the same trained weights against nominally "the test
    split" of the same underlying dataset. If ``evaluate()`` reused the
    split that was actually used during training, the two R2 values would be
    identical (same rows, same weights, same deterministic forward pass).
    Today they diverge because ``evaluate()`` recomputes an unseeded, brand
    new train/val/test partition instead of reusing the training split --
    this test is expected to FAIL until that split-resolution bug is fixed.

    Args:
        linear_regression_dataset: Fixture providing dataset paths, reused to
            build a fresh ``InferenceJobConfig`` with no shared in-memory
            state from the training call.
        regression_trained_result: Completed TrainingResult, carrying the
            training-time test-split R2 in ``.metrics``.
        regression_checkpoint_path: Path to the checkpoint written by training.
    """
    train_time_r2 = regression_trained_result.metrics[TRAIN_TEST_R2_KEY]

    inference_payload: dict[str, object] = {
        "run": {"type": "predict"},
        "experiment": {"name": "split_reproducibility_regression"},
        **_build_model_and_data_payload(
            linear_regression_dataset, checkpoint=regression_checkpoint_path
        ),
    }
    inference_settings = InferenceJobConfig.model_validate(inference_payload)

    result = api_evaluate(inference_settings)
    try:
        eval_time_r2 = result.metrics[EVAL_R2_KEY]
        assert eval_time_r2 == pytest.approx(train_time_r2, abs=R2_TOLERANCE), (
            f"train-time test R2={train_time_r2!r} vs eval-time R2={eval_time_r2!r} "
            "diverge by more than the tolerance -- evaluate() is almost certainly "
            "scoring against a different (unseeded, freshly regenerated) test split "
            "than the one used during training."
        )
    finally:
        for fig in result.figures.values():
            plt.close(fig)
