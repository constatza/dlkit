"""Recovery helper for reloading a previously trained run's persisted split.

Standalone, explicitly-invoked utility. Every MLflow-tracked run already
logs its split under a ``splits/`` artifact directory (see
``ArtifactLogger.log_split_artifact``), independent of the seeded
producer/consumer split-resolution pair in
``infrastructure.io.split_provider``. This module lets a caller recover that
artifact for a run whose local split file has been lost (e.g. an old run
predating the seeded-split fix, or a local split file deleted outside
DLKit). ``evaluate()`` never calls this automatically — the caller downloads
the split first, then points ``data.splits.filepath`` at the result.
"""

from __future__ import annotations

from pathlib import Path

from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException

from dlkit.common.errors import WorkflowError
from dlkit.infrastructure.utils.logging_config import get_logger

from .mlflow_client_factory import MLflowClientFactory

logger = get_logger(__name__)

SPLIT_ARTIFACT_DIR = "splits"


def download_run_split(
    run_id: str,
    destination: Path,
    *,
    tracking_uri: str | None = None,
) -> Path:
    """Download a run's logged ``splits/*.json`` artifact to a local directory.

    Args:
        run_id: MLflow run ID that logged a split artifact during training
            (via ``ArtifactLogger.log_split_artifact``).
        destination: Local directory to download the split file into.
        tracking_uri: Optional explicit MLflow tracking URI override.

    Returns:
        Path to the downloaded split file.

    Raises:
        WorkflowError: If the run's ``splits/`` artifact directory cannot be
            listed, is empty, or contains more than one file (ambiguous —
            should not happen for a correctly tracked run).
    """
    client = MLflowClientFactory.create_client(tracking_uri)
    try:
        artifacts = client.list_artifacts(run_id, path=SPLIT_ARTIFACT_DIR)
    except MlflowException as exc:
        raise WorkflowError(
            f"Could not list '{SPLIT_ARTIFACT_DIR}/' artifacts for run {run_id!r}: {exc}",
            {"run_id": run_id},
        ) from exc

    files = [artifact for artifact in artifacts if not artifact.is_dir]
    if len(files) != 1:
        raise WorkflowError(
            f"Expected exactly one split artifact under '{SPLIT_ARTIFACT_DIR}/' for "
            f"run {run_id!r}, found {len(files)}.",
            {"run_id": run_id, "artifacts": [artifact.path for artifact in files]},
        )

    destination.mkdir(parents=True, exist_ok=True)
    downloaded_path = download_artifacts(
        run_id=run_id,
        artifact_path=files[0].path,
        dst_path=str(destination),
        tracking_uri=tracking_uri,
    )
    logger.info("Downloaded split artifact for run {} to {}", run_id, downloaded_path)
    return Path(downloaded_path)
