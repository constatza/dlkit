"""Recovery helper for downloading a previously trained run's checkpoint.

Standalone, explicitly-invoked utility mirroring ``split_recovery.py``.
Every MLflow-tracked run that trained with checkpointing enabled logs its
checkpoint under a ``checkpoints/`` artifact directory (see
``ArtifactLogger.log_checkpoints`` and ``CHECKPOINT_ARTIFACT_DIR``). Task 1
made dlkit's default ``ModelCheckpoint`` config best-only
(``save_top_k=1``), so a correctly tracked run has at most one file under
that directory — but callers may supply a custom ``CallbackSettings``
``filename=`` template, so this module doesn't assume a fixed name: it
downloads the run's single checkpoint artifact, whatever it's named, falling
back to ``best.ckpt`` as a disambiguator only when more than one file is
present. This lets a caller download a checkpoint for a resolved run id
(see ``run_queries.py`` for resolving that run id from an experiment name or
a parent run). ``evaluate()`` never calls this automatically — the caller
downloads the checkpoint first, then points the model's checkpoint override
at the result.
"""

from __future__ import annotations

from pathlib import Path

from mlflow.artifacts import download_artifacts
from mlflow.entities import FileInfo
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
    """Download a run's logged checkpoint artifact to a local directory.

    Discovers whatever file(s) actually exist under the run's
    ``checkpoints/`` artifact directory rather than assuming a fixed
    filename, so this works regardless of the ``filename=`` template a
    caller's ``ModelCheckpoint`` callback used. Task 1's best-only default
    (``save_top_k=1``) guarantees at most one file per correctly tracked
    run, so a single file is downloaded as-is. If more than one file is
    present (an unusual custom override), ``best.ckpt`` is used as a
    disambiguator when it's among them; otherwise the ambiguity is fatal.

    Args:
        run_id: MLflow run id that logged a checkpoint artifact during
            training (via ``ArtifactLogger.log_checkpoints``).
        destination: Local directory to download the checkpoint file into.
        tracking_uri: Optional explicit MLflow tracking URI override.

    Returns:
        Path to the downloaded checkpoint file.

    Raises:
        WorkflowError: If ``run_id`` does not exist; if the run has no
            checkpoint file under ``checkpoints/`` (e.g. checkpointing was
            disabled for the run); or if the run has more than one
            checkpoint file there and none of them is named ``best.ckpt``
            (an unusual custom override with no way to disambiguate).
    """
    client = MLflowClientFactory.create_client(tracking_uri)
    try:
        client.get_run(run_id)
    except MlflowException as exc:
        raise WorkflowError(f"Run {run_id!r} not found: {exc}", {"run_id": run_id}) from exc

    artifacts = client.list_artifacts(run_id, CHECKPOINT_ARTIFACT_DIR)
    files = [artifact for artifact in artifacts if not artifact.is_dir]

    if not files:
        raise WorkflowError(
            f"No checkpoint artifact under {CHECKPOINT_ARTIFACT_DIR!r} for run "
            f"{run_id!r} — was checkpointing disabled for this run?",
            {"run_id": run_id, "artifact_dir": CHECKPOINT_ARTIFACT_DIR},
        )

    artifact_path = _resolve_checkpoint_artifact_path(files, run_id=run_id)

    destination.mkdir(parents=True, exist_ok=True)
    downloaded = download_artifacts(
        run_id=run_id,
        artifact_path=artifact_path,
        dst_path=str(destination),
        tracking_uri=tracking_uri,
    )
    logger.info("Downloaded checkpoint artifact for run {} to {}", run_id, downloaded)
    return Path(downloaded)


def _resolve_checkpoint_artifact_path(files: list[FileInfo], *, run_id: str) -> str:
    """Pick the single checkpoint artifact path from a non-empty file list.

    Args:
        files: Non-directory ``FileInfo`` entries found under
            ``checkpoints/`` (already known to be non-empty).
        run_id: MLflow run id, for error messages only.

    Returns:
        The artifact path to download: the sole file's path when there's
        exactly one, or ``best.ckpt``'s path when there are several and it
        is among them.

    Raises:
        WorkflowError: If there is more than one file and none is named
            ``best.ckpt``.
    """
    if len(files) == 1:
        return files[0].path

    best_match = next(
        (
            artifact
            for artifact in files
            if Path(artifact.path).name == _BEST_CHECKPOINT_ARTIFACT_NAME
        ),
        None,
    )
    if best_match is not None:
        logger.info(
            "Multiple checkpoint files found for run {}; using {!r} as the disambiguator",
            run_id,
            best_match.path,
        )
        return best_match.path

    found = [artifact.path for artifact in files]
    raise WorkflowError(
        f"Found {len(files)} checkpoint files under {CHECKPOINT_ARTIFACT_DIR!r} for "
        f"run {run_id!r}: {found}, and none is named {_BEST_CHECKPOINT_ARTIFACT_NAME!r}. "
        "dlkit's checkpoint config always produces at most one file per run (Task 1's "
        "best-only save_top_k=1 default), so this indicates an unusual custom "
        "ModelCheckpoint override with no way to disambiguate.",
        {"run_id": run_id, "artifact_dir": CHECKPOINT_ARTIFACT_DIR, "found": found},
    )
