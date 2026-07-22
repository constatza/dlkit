"""Tests for `checkpoint_recovery.download_checkpoint_artifact`.

All tests run against real MLflow-tracked training runs (via the
`default_checkpoint_run`/`no_checkpoint_run` fixtures in `conftest.py`,
which call `api_train()` against a real local sqlite MLflow backend), not
mocks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlkit.common import TrainingResult
from dlkit.common.errors import WorkflowError
from dlkit.engine.tracking.checkpoint_recovery import download_checkpoint_artifact


def test_downloads_best_checkpoint_for_default_config_run(
    default_checkpoint_run: TrainingResult, tracking_uri: str, tmp_path: Path
) -> None:
    assert default_checkpoint_run.mlflow_run_id is not None
    destination = tmp_path / "downloaded_best"

    result = download_checkpoint_artifact(
        default_checkpoint_run.mlflow_run_id,
        destination,
        tracking_uri=tracking_uri,
    )

    assert result.exists()
    assert result.name == "best.ckpt"


def test_raises_for_run_with_no_checkpoint_artifacts_at_all(
    no_checkpoint_run: TrainingResult, tracking_uri: str, tmp_path: Path
) -> None:
    assert no_checkpoint_run.mlflow_run_id is not None

    with pytest.raises(WorkflowError, match="best"):
        download_checkpoint_artifact(
            no_checkpoint_run.mlflow_run_id,
            tmp_path / "downloaded_none",
            tracking_uri=tracking_uri,
        )


def test_raises_for_nonexistent_run_id(tracking_uri: str, tmp_path: Path) -> None:
    with pytest.raises(WorkflowError, match="not found"):
        download_checkpoint_artifact(
            "does-not-exist",
            tmp_path / "downloaded_missing_run",
            tracking_uri=tracking_uri,
        )
