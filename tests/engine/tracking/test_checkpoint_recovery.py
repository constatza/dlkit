"""Tests for `checkpoint_recovery.download_checkpoint_artifact`.

All tests run against real MLflow-tracked training runs (via the
`default_checkpoint_run`/`custom_filename_checkpoint_run`/`no_checkpoint_run`/
`multiple_checkpoint_files_with_best_run`/
`multiple_checkpoint_files_without_best_run` fixtures in `conftest.py`,
which call `api_train()` against a real local sqlite MLflow backend), not
mocks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlkit.common import TrainingResult
from dlkit.common.errors import WorkflowError
from dlkit.engine.tracking.checkpoint_recovery import download_checkpoint_artifact


def test_downloads_checkpoint_for_default_config_run(
    default_checkpoint_run: TrainingResult, tracking_uri: str, tmp_path: Path
) -> None:
    assert default_checkpoint_run.mlflow_run_id is not None
    local_checkpoints_dir = tmp_path / "default_checkpoint_output" / "checkpoints"
    local_checkpoints = list(local_checkpoints_dir.glob("*.ckpt"))
    assert len(local_checkpoints) == 1, "expected exactly one local checkpoint file"
    expected_checkpoint = local_checkpoints[0]
    destination = tmp_path / "downloaded_default"

    result = download_checkpoint_artifact(
        default_checkpoint_run.mlflow_run_id,
        destination,
        tracking_uri=tracking_uri,
    )

    assert result.exists()
    assert result.name == expected_checkpoint.name
    assert result.read_bytes() == expected_checkpoint.read_bytes()


def test_downloads_checkpoint_with_custom_filename_template(
    custom_filename_checkpoint_run: TrainingResult, tracking_uri: str, tmp_path: Path
) -> None:
    """A custom `ModelCheckpoint(filename=...)` override must still be found and downloaded."""
    assert custom_filename_checkpoint_run.mlflow_run_id is not None
    destination = tmp_path / "downloaded_custom"

    result = download_checkpoint_artifact(
        custom_filename_checkpoint_run.mlflow_run_id,
        destination,
        tracking_uri=tracking_uri,
    )

    assert result.exists()
    assert result.name == "my-custom-name.ckpt"


def test_raises_for_run_with_no_checkpoint_artifacts_at_all(
    no_checkpoint_run: TrainingResult, tracking_uri: str, tmp_path: Path
) -> None:
    assert no_checkpoint_run.mlflow_run_id is not None

    with pytest.raises(WorkflowError, match="No checkpoint artifact"):
        download_checkpoint_artifact(
            no_checkpoint_run.mlflow_run_id,
            tmp_path / "downloaded_none",
            tracking_uri=tracking_uri,
        )


def test_downloads_best_ckpt_when_multiple_files_include_it(
    multiple_checkpoint_files_with_best_run: TrainingResult, tracking_uri: str, tmp_path: Path
) -> None:
    """`best.ckpt` disambiguates when multiple checkpoint files are present."""
    assert multiple_checkpoint_files_with_best_run.mlflow_run_id is not None
    destination = tmp_path / "downloaded_disambiguated"

    result = download_checkpoint_artifact(
        multiple_checkpoint_files_with_best_run.mlflow_run_id,
        destination,
        tracking_uri=tracking_uri,
    )

    assert result.exists()
    assert result.name == "best.ckpt"


def test_raises_for_multiple_checkpoint_files_without_best_ckpt(
    multiple_checkpoint_files_without_best_run: TrainingResult, tracking_uri: str, tmp_path: Path
) -> None:
    """No `best.ckpt` among multiple files leaves no way to disambiguate."""
    assert multiple_checkpoint_files_without_best_run.mlflow_run_id is not None

    with pytest.raises(WorkflowError, match="Found 2 checkpoint files"):
        download_checkpoint_artifact(
            multiple_checkpoint_files_without_best_run.mlflow_run_id,
            tmp_path / "downloaded_ambiguous",
            tracking_uri=tracking_uri,
        )


def test_raises_for_nonexistent_run_id(tracking_uri: str, tmp_path: Path) -> None:
    with pytest.raises(WorkflowError, match="not found"):
        download_checkpoint_artifact(
            "does-not-exist",
            tmp_path / "downloaded_missing_run",
            tracking_uri=tracking_uri,
        )
