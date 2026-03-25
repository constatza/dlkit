"""MLflow checkpoint router callback.

Redirects Lightning ModelCheckpoint output into the active MLflow run's local
artifact store, so checkpoints are co-located with other run artifacts and
never need to be re-uploaded.

Only activates when all of the following are true:
- An MLflow run is active at fit start
- The artifact URI uses the local ``file://`` scheme
- The ModelCheckpoint callback has ``dirpath`` still unset (``None``)

User-set ``dirpath`` values are NEVER modified.
"""

from __future__ import annotations

from pathlib import Path

from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger


def _resolve_local_artifact_dir() -> Path | None:
    """Return the local artifact directory for the active MLflow run, or None.

    Returns:
        Absolute path to the run's artifact directory when the active run uses
        a local ``file://`` artifact URI, otherwise ``None``.
    """
    try:
        import mlflow

        active_run = mlflow.active_run()
        if active_run is None:
            return None

        artifact_uri = mlflow.get_artifact_uri()
        if not artifact_uri or not artifact_uri.startswith("file://"):
            return None

        from dlkit.tools.io.url_utils import get_url_path

        raw_path = get_url_path(artifact_uri).lstrip("/")
        # On Unix the leading "/" is removed — restore it for absolute paths
        candidate = Path("/" + raw_path) if not Path(raw_path).is_absolute() else Path(raw_path)
        return candidate.resolve()
    except Exception as exc:
        logger.debug(f"MlflowCheckpointRouter: could not resolve artifact dir: {exc}")
        return None


def _redirect_checkpoint_callbacks(trainer: Trainer, checkpoint_dir: Path) -> None:
    """Set ``dirpath`` on any ModelCheckpoint callback that has not set one.

    Only modifies callbacks whose ``dirpath`` is ``None`` — user-configured
    paths are left untouched.

    Args:
        trainer: Active Lightning Trainer instance.
        checkpoint_dir: Target directory for checkpoint output.
    """
    try:
        from lightning.pytorch.callbacks import ModelCheckpoint
    except ImportError:
        try:
            from pytorch_lightning.callbacks import ModelCheckpoint
        except ImportError:
            logger.debug("MlflowCheckpointRouter: ModelCheckpoint not available")
            return

    for cb in getattr(trainer, "callbacks", []):
        if isinstance(cb, ModelCheckpoint) and cb.dirpath is None:
            cb.dirpath = str(checkpoint_dir)
            logger.debug(f"MlflowCheckpointRouter: redirected ModelCheckpoint → {checkpoint_dir}")


class MlflowCheckpointRouter(Callback):
    """Redirect unset ModelCheckpoint dirpaths into the MLflow artifact store.

    Injected by ``TrackingDecorator`` before training starts. On ``on_fit_start``,
    checks whether an active MLflow run with a local ``file://`` artifact URI exists
    and redirects any ModelCheckpoint callback whose ``dirpath`` is ``None`` to write
    directly into ``<artifact_uri>/checkpoints/``.

    This prevents stray checkpoint files from accumulating outside the artifact
    store while still letting Lightning manage checkpoint overwrite logic.
    """

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Redirect checkpoint callbacks when a local MLflow run is active.

        Args:
            trainer: Active Lightning Trainer instance.
            pl_module: The Lightning module being trained (unused).
        """
        artifact_dir = _resolve_local_artifact_dir()
        if artifact_dir is None:
            logger.debug(
                "MlflowCheckpointRouter: no local MLflow artifact dir found; skipping redirect"
            )
            return

        checkpoint_dir = artifact_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _redirect_checkpoint_callbacks(trainer, checkpoint_dir)
