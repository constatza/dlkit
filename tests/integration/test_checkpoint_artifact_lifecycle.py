"""Integration test proving trained checkpoints are deleted from local disk
immediately after MLflow upload, even for local (sqlite) backends where the
artifact policy says they should not be removed.

Root cause: ``ArtifactLogger._log_or_skip_checkpoint()``
(``src/dlkit/engine/tracking/artifact_logger.py:165-186``) unconditionally
calls ``ckpt_path.unlink()`` right after uploading the checkpoint to MLflow,
regardless of ``ArtifactPolicy.remove_uploaded_files``
(``src/dlkit/engine/artifacts.py:60``) -- which ``tracking_decorator.py``
already computes correctly as ``False`` for local backends, but which
``_log_or_skip_checkpoint`` never reads. ``TrainingResult.checkpoint_path``
(``src/dlkit/common/results.py:48-55``) performs no existence check, so a
caller naively reusing it after a tracked local training run hits a
``FileNotFoundError`` rather than a clear, documented behavior.

This is a secondary bug relative to the eval-time R2 collapse (the primary
bug is the unseeded split, covered by
``tests/integration/test_split_reproducibility_regression.py`` and
``tests/infrastructure/io/test_split_determinism.py``), but it is fixed in
the same pass, so it gets its own Phase 0 proof here.

This test is expected to stay in the suite permanently and flip from FAILING
to PASSING once the fix plan's artifact-policy fix lands.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlkit.common import TrainingResult
from dlkit.infrastructure.config.job_config import TrainingJobConfig
from dlkit.interfaces.api.functions import train as api_train


@pytest.fixture
def tracked_training_settings_with_checkpoint(
    mlflow_settings: TrainingJobConfig, tmp_path: Path
) -> TrainingJobConfig:
    """MLflow-tracked TrainingJobConfig with real (non-``fast_dev_run``) checkpointing.

    Composes two independent overrides already established in
    ``tests/integration/conftest.py``: ``mlflow_settings`` (local sqlite
    MLflow backend) and the checkpoint-enabling trainer patch used by
    ``tests/integration/test_evaluate_integration.py``'s
    ``training_settings_with_checkpoint`` fixture. Both mutate the same base
    ``training_settings`` fixture along independent axes (tracking backend
    vs. trainer checkpointing), so composing them here avoids duplicating
    either fixture's setup.

    Args:
        mlflow_settings: Base TrainingJobConfig with MLflow tracking configured.
        tmp_path: Pytest temporary directory fixture.

    Returns:
        TrainingJobConfig with both MLflow tracking and real checkpointing enabled.
    """
    training_cfg = mlflow_settings.training
    assert training_cfg is not None
    trainer_cfg = training_cfg.trainer
    assert trainer_cfg is not None
    return mlflow_settings.model_copy(
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


def test_checkpoint_path_exists_immediately_after_tracked_training(
    tracked_training_settings_with_checkpoint: TrainingJobConfig,
) -> None:
    """``TrainingResult.checkpoint_path`` should point at a file that still exists.

    Expected to FAIL today: ``ArtifactLogger._log_or_skip_checkpoint()``
    unconditionally deletes the local checkpoint after MLflow upload, so the
    path recorded in ``TrainingResult.artifacts`` points at a file that no
    longer exists on disk by the time ``api_train()`` returns.

    Args:
        tracked_training_settings_with_checkpoint: MLflow-tracked, checkpointed
            training config fixture.
    """
    result: TrainingResult = api_train(tracked_training_settings_with_checkpoint)

    assert result.checkpoint_path is not None, "no checkpoint artifact was recorded at all"
    assert result.checkpoint_path.exists(), (
        f"checkpoint_path={result.checkpoint_path.as_posix()} was recorded but does not "
        "exist on disk -- ArtifactLogger._log_or_skip_checkpoint() deletes the local "
        "file unconditionally after MLflow upload, ignoring "
        "ArtifactPolicy.remove_uploaded_files."
    )
