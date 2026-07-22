"""Recovery helper for downloading a previously trained run's best checkpoint.

Standalone, explicitly-invoked utility mirroring ``split_recovery.py``.
Every MLflow-tracked run that trained with checkpointing enabled logs its
best checkpoint under a ``checkpoints/`` artifact directory with a canonical
filename (see ``ArtifactLogger.log_checkpoints`` and
``CHECKPOINT_ARTIFACT_DIR``): ``best.ckpt``. This module lets a caller
download that checkpoint for a resolved run id (see ``run_queries.py`` for
resolving that run id from an experiment name or a parent run).
``evaluate()`` never calls this automatically — the caller downloads the
checkpoint first, then points the model's checkpoint override at the
result.
"""

from __future__ import annotations

from pathlib import Path

from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException

from dlkit.common.errors import WorkflowError
from dlkit.infrastructure.utils.logging_config import get_logger

from .artifact_logger import CHECKPOINT_ARTIFACT_DIR
from .mlflow_client_factory import MLflowClientFactory

logger = get_logger(__name__)

_BEST_CHECKPOINT_ARTIFACT_NAME = "best.ckpt"


def download_checkpoint_artifact(
    run_id: str,
    destination: Path,
    *,
    tracking_uri: str | None = None,
) -> Path:
    """Download a run's logged best-checkpoint artifact to a local directory.

    Args:
        run_id: MLflow run id that logged a checkpoint artifact during
            training (via ``ArtifactLogger.log_checkpoints``).
        destination: Local directory to download the checkpoint file into.
        tracking_uri: Optional explicit MLflow tracking URI override.

    Returns:
        Path to the downloaded checkpoint file.

    Raises:
        WorkflowError: If ``run_id`` does not exist, or the run has no
            ``best.ckpt`` artifact under ``checkpoints/`` (e.g. checkpointing
            was disabled for the run).
    """
    artifact_path = f"{CHECKPOINT_ARTIFACT_DIR}/{_BEST_CHECKPOINT_ARTIFACT_NAME}"
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
            f"No best checkpoint artifact ({artifact_path!r}) for run "
            f"{run_id!r} — was it logged before this naming convention shipped, "
            "or was checkpointing disabled for this run?",
            {"run_id": run_id, "artifact_path": artifact_path},
        ) from exc
    logger.info("Downloaded best checkpoint artifact for run {} to {}", run_id, downloaded)
    return Path(downloaded)
