"""Tests for `checkpoint_recovery.download_checkpoint_artifact`.

All tests run against real MLflow-tracked training runs (via the
`default_checkpoint_run`/`save_last_checkpoint_run`/`no_checkpoint_run`
fixtures in `conftest.py`, which call `api_train()` against a real local
sqlite MLflow backend), not mocks.
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
        "best",
        destination,
        tracking_uri=tracking_uri,
    )

    assert result.exists()
    assert result.name == "best.ckpt"


def test_downloads_last_checkpoint_for_save_last_override_run(
    save_last_checkpoint_run: TrainingResult, tracking_uri: str, tmp_path: Path
) -> None:
    assert save_last_checkpoint_run.mlflow_run_id is not None
    destination = tmp_path / "downloaded_last"

    result = download_checkpoint_artifact(
        save_last_checkpoint_run.mlflow_run_id,
        "last",
        destination,
        tracking_uri=tracking_uri,
    )

    assert result.exists()
    assert result.name == "last.ckpt"


def test_raises_for_last_variant_on_best_only_default_config_run(
    default_checkpoint_run: TrainingResult, tracking_uri: str, tmp_path: Path
) -> None:
    """After Task 1, the default checkpoint config is best-only (no `save_last=True`).

    Requesting `variant="last"` for a default-config run must raise, not
    silently fall back to `best.ckpt` — there is no "last" artifact to find.
    """
    assert default_checkpoint_run.mlflow_run_id is not None

    with pytest.raises(WorkflowError, match="last"):
        download_checkpoint_artifact(
            default_checkpoint_run.mlflow_run_id,
            "last",
            tmp_path / "downloaded_last_missing",
            tracking_uri=tracking_uri,
        )


def test_raises_for_run_with_no_checkpoint_artifacts_at_all(
    no_checkpoint_run: TrainingResult, tracking_uri: str, tmp_path: Path
) -> None:
    assert no_checkpoint_run.mlflow_run_id is not None

    with pytest.raises(WorkflowError, match="best"):
        download_checkpoint_artifact(
            no_checkpoint_run.mlflow_run_id,
            "best",
            tmp_path / "downloaded_none",
            tracking_uri=tracking_uri,
        )


def test_raises_for_nonexistent_run_id(tracking_uri: str, tmp_path: Path) -> None:
    with pytest.raises(WorkflowError, match="not found"):
        download_checkpoint_artifact(
            "does-not-exist",
            "best",
            tmp_path / "downloaded_missing_run",
            tracking_uri=tracking_uri,
        )
