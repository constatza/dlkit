"""Recovery helper for downloading a previously trained run's checkpoint.

Standalone, explicitly-invoked utility mirroring ``split_recovery.py``.
Every MLflow-tracked run that trained with checkpointing enabled logs its
checkpoint(s) under a ``checkpoints/`` artifact directory with a canonical
filename (see ``ArtifactLogger.log_checkpoints`` and
``CHECKPOINT_ARTIFACT_DIR``): ``best.ckpt`` always, plus ``last.ckpt`` only
when the run's ``ModelCheckpoint`` callback was configured with
``save_last=True``. This module lets a caller download either variant for a
resolved run id (see ``run_queries.py`` for resolving that run id from an
experiment name or a parent run). ``evaluate()`` never calls this
automatically — the caller downloads the checkpoint first, then points the
model's checkpoint override at the result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException

from dlkit.common.errors import WorkflowError
from dlkit.infrastructure.utils.logging_config import get_logger

from .artifact_logger import CHECKPOINT_ARTIFACT_DIR
from .mlflow_client_factory import MLflowClientFactory

logger = get_logger(__name__)

_ARTIFACT_NAME_BY_VARIANT: dict[Literal["best", "last"], str] = {
    "best": "best.ckpt",
    "last": "last.ckpt",
}


def download_checkpoint_artifact(
    run_id: str,
    variant: Literal["best", "last"],
    destination: Path,
    *,
    tracking_uri: str | None = None,
) -> Path:
    """Download a run's logged checkpoint artifact to a local directory.

    Args:
        run_id: MLflow run id that logged a checkpoint artifact during
            training (via ``ArtifactLogger.log_checkpoints``).
        variant: Which checkpoint to download — ``"best"`` (always present
            for a checkpointed run) or ``"last"`` (only present when the
            run's ``ModelCheckpoint`` callback set ``save_last=True``).
        destination: Local directory to download the checkpoint file into.
        tracking_uri: Optional explicit MLflow tracking URI override.

    Returns:
        Path to the downloaded checkpoint file.

    Raises:
        WorkflowError: If ``run_id`` does not exist, or the run has no
            ``{variant}.ckpt`` artifact under ``checkpoints/`` (e.g.
            checkpointing was disabled for the run, or ``variant="last"``
            was requested for a run that used the best-only default).
    """
    artifact_path = f"{CHECKPOINT_ARTIFACT_DIR}/{_ARTIFACT_NAME_BY_VARIANT[variant]}"
    client = MLflowClientFactory.create_client(tracking_uri)
    try:
        client.get_run(run_id)
    except MlflowException as exc:
        raise WorkflowError(f"Run {run_id!r} not found: {exc}", {"run_id": run_id}) from exc

    destination.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
            dst_path=str(destination),
            tracking_uri=tracking_uri,
        )
    except MlflowException as exc:
        raise WorkflowError(
            f"No {variant!r} checkpoint artifact ({artifact_path!r}) for run "
            f"{run_id!r} — was it logged before this naming convention shipped, "
            "or was checkpointing disabled for this run?",
            {"run_id": run_id, "variant": variant, "artifact_path": artifact_path},
        ) from exc
    logger.info("Downloaded {} checkpoint artifact for run {} to {}", variant, run_id, downloaded)
    return Path(downloaded)
