"""Artifact logging service for MLflow tracking.

Single Responsibility: Log checkpoints, models, and user-defined artifacts to MLflow.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from lightning.pytorch import Trainer
from torch import nn

from dlkit.common import TrainingResult
from dlkit.common.hooks import ParamValue
from dlkit.engine.adapters.lightning.base import ProcessingLightningWrapper
from dlkit.engine.artifacts import (
    ArtifactPublisher,
    ContentArtifactPayload,
    FileArtifactPayload,
    ProducedArtifact,
)
from dlkit.engine.training.components import RuntimeComponents
from dlkit.infrastructure.config.job_config import JobConfig
from dlkit.infrastructure.utils.logging_config import get_logger

from .config_accessor import ConfigAccessor
from .interfaces import IExperimentTracker, IRunContext

type _WorkflowSettings = JobConfig

logger = get_logger(__name__)

DEFAULT_MODEL_ARTIFACT_PATH = "model"
CHECKPOINT_ARTIFACT_DIR = "checkpoints"
TAG_LOGGED_MODEL_URI = "mlflow_logged_model_uri"
TAG_LOGGED_MODEL_ARTIFACT_PATH = "mlflow_logged_model_artifact_path"
TAG_MODEL_CLASS = "mlflow_model_class"


class CheckpointCallbackLike(Protocol):
    best_model_path: str | Path | None
    last_model_path: str | Path | None


def _resolve_model_class_name(model: object) -> str:
    """Return the effective class name, unwrapping DLKit Lightning wrappers.

    Args:
        model: Model object, possibly a ProcessingLightningWrapper.

    Returns:
        Class name of the underlying nn.Module when model is a wrapper,
        otherwise the class name of model itself.
    """
    if isinstance(model, ProcessingLightningWrapper):
        return type(model.model).__name__
    return type(model).__name__


def _split_artifact_path(artifact_path: str) -> tuple[str, str]:
    path = Path(artifact_path)
    artifact_dir = path.parent.as_posix()
    artifact_name = path.name
    return ("" if artifact_dir == "." else artifact_dir, artifact_name)


class RunContextArtifactPublisher(ArtifactPublisher):
    """Publish typed produced artifacts through the active run context."""

    def __init__(self, run_context: IRunContext) -> None:
        self.run_context = run_context

    def publish(self, artifact: ProducedArtifact) -> None:
        artifact_dir, artifact_name = _split_artifact_path(artifact.artifact_path)
        match artifact.payload:
            case FileArtifactPayload(file_path=file_path):
                self.run_context.log_artifact(file_path, artifact_dir=artifact_dir)
            case ContentArtifactPayload(content=content):
                target = artifact_name if not artifact_dir else f"{artifact_dir}/{artifact_name}"
                self.run_context.log_artifact_content(content, target)
            case _:
                raise TypeError(f"Unsupported artifact payload: {type(artifact.payload).__name__}")


def _log_or_skip_checkpoint(
    run_context: IRunContext,
    ckpt_path: Path,
    artifact_dir: str,
) -> None:
    """Upload checkpoint to MLflow and remove the local copy.

    For remote backends the local file is removed after upload to free disk.
    For local backends the file is also uploaded (via MLflow's copy) and then
    the original is removed to avoid duplicates.

    Args:
        run_context: Active ``IRunContext`` for logging.
        ckpt_path: Path to the checkpoint file.
        artifact_dir: Sub-path within the artifact store for uploaded files.
    """
    run_context.log_artifact(ckpt_path, artifact_dir)
    try:
        ckpt_path.unlink()
        logger.debug("Removed local checkpoint after upload: {}", ckpt_path)
    except OSError as exc:
        logger.warning("Could not remove local checkpoint {}: {}", ckpt_path, exc)


class ArtifactLogger:
    """Handles artifact logging to MLflow.

    Single Responsibility: Log checkpoints, models, and user-defined artifacts.
    Delegates configuration access to ConfigAccessor.

    Args:
        tracker: Experiment tracker implementation
    """

    def __init__(self, tracker: IExperimentTracker):
        """Initialize with experiment tracker.

        Args:
            tracker: Experiment tracker implementation
        """
        self._tracker = tracker

    def log_training_artifacts(
        self,
        components: RuntimeComponents,
        settings: _WorkflowSettings,
        run_context: IRunContext,
    ) -> None:
        """Log all training artifacts (checkpoints and model artifact).

        Args:
            components: Build components containing trainer
            settings: Global settings
            run_context: Run context for logging
        """
        self.log_split_artifact(components, run_context)
        self.log_checkpoints(components, run_context)
        self._log_model_artifact(run_context=run_context, model=components.model)

    def log_split_artifact(
        self,
        components: RuntimeComponents,
        run_context: IRunContext,
    ) -> None:
        """Log the split used by the run without creating new local cache files."""
        try:
            split_artifact = components.artifacts.split_artifact
            if split_artifact is None:
                return
            RunContextArtifactPublisher(run_context).publish(split_artifact)
        except Exception as e:
            logger.warning("Failed to log split artifact: {}", e)

    def log_checkpoints(
        self,
        components: RuntimeComponents,
        run_context: IRunContext,
    ) -> None:
        """Log model checkpoints as artifacts.

        Logs best and last checkpoints from PyTorch Lightning trainer.
        Raises on failure so training aborts rather than silently missing the artifact.

        Args:
            components: Build components containing trainer
            run_context: Run context for logging
        """
        trainer = getattr(components, "trainer", None)
        if not trainer:
            logger.debug("No trainer found in components")
            return

        ckpt_cb = self._find_checkpoint_callback(trainer)
        if not ckpt_cb:
            logger.debug("No ModelCheckpoint callback found")
            return

        best = getattr(ckpt_cb, "best_model_path", None)
        last = getattr(ckpt_cb, "last_model_path", None)
        if best is not None and not isinstance(best, str | Path):
            best = None
        if last is not None and not isinstance(last, str | Path):
            last = None

        if best:
            _log_or_skip_checkpoint(run_context, Path(best), CHECKPOINT_ARTIFACT_DIR)
            logger.debug("Logged best checkpoint {}", best)
        if last and last != best:
            _log_or_skip_checkpoint(run_context, Path(last), CHECKPOINT_ARTIFACT_DIR)
            logger.debug("Logged last checkpoint {}", last)

    def _log_model_artifact(
        self,
        *,
        run_context: IRunContext,
        model: nn.Module,
    ) -> None:
        """Log the trained model as an MLflow artifact (no registry registration).

        Raises on failure so training aborts rather than silently missing the artifact.

        Args:
            run_context: Active run context for logging.
            model: Trained model to log.
        """
        model_uri = run_context.log_model(
            model=model,
            artifact_path=DEFAULT_MODEL_ARTIFACT_PATH,
        )
        if model_uri:
            run_context.set_tag(TAG_MODEL_CLASS, _resolve_model_class_name(model))
            run_context.set_tag(TAG_LOGGED_MODEL_URI, model_uri)
            run_context.set_tag(TAG_LOGGED_MODEL_ARTIFACT_PATH, DEFAULT_MODEL_ARTIFACT_PATH)

    def log_user_artifacts(
        self,
        settings: _WorkflowSettings,
        run_context: IRunContext,
        result: TrainingResult,
    ) -> None:
        """Orchestrate logging of user-defined artifacts and params from settings.EXTRAS.

        Args:
            settings: Global settings
            run_context: Run context for logging
            result: Training result (not currently used but kept for extensibility)
        """
        accessor = ConfigAccessor(settings)
        extras = accessor.get_extras()
        if not extras:
            logger.debug("No EXTRAS configuration found")
            return

        try:
            self._log_user_params(accessor, run_context)
            self._log_user_file_artifacts(accessor, run_context)
            self._log_user_toml_artifacts(accessor, run_context)
        except Exception as e:
            logger.warning("Failed to log user-defined artifacts or params: {}", e)

    def _find_checkpoint_callback(self, trainer: Trainer) -> CheckpointCallbackLike | None:
        """Find ModelCheckpoint callback in trainer.

        Args:
            trainer: PyTorch Lightning trainer

        Returns:
            ModelCheckpoint callback or None
        """
        ckpt_cb = getattr(trainer, "checkpoint_callback", None)
        if ckpt_cb:
            return ckpt_cb

        callbacks = getattr(trainer, "callbacks", None)
        if not isinstance(callbacks, Sequence):
            return None

        try:
            from pytorch_lightning.callbacks import ModelCheckpoint
        except ImportError as e:
            logger.warning("Failed to import ModelCheckpoint: {}", e)
            return None

        candidates: list[CheckpointCallbackLike] = [
            c for c in callbacks if isinstance(c, ModelCheckpoint)
        ]
        return candidates[0] if candidates else None

    def _log_user_params(
        self,
        accessor: ConfigAccessor,
        run_context: IRunContext,
    ) -> None:
        """Log user-defined parameters from EXTRAS.mlflow_params.

        Args:
            accessor: Configuration accessor
            run_context: Run context for logging
        """
        params_dict = accessor.get_mlflow_params()
        if not params_dict:
            return

        safe_params: dict[str, ParamValue] = {}
        for key, value in params_dict.items():
            try:
                safe_params[key] = str(value) if value is not None else ""
            except Exception as e:
                logger.warning("Skipping non-serializable param '{}': {}", key, e)

        if safe_params:
            run_context.log_params(safe_params)
            logger.debug("Logged {} custom params from EXTRAS.mlflow_params", len(safe_params))

    def _log_user_file_artifacts(
        self,
        accessor: ConfigAccessor,
        run_context: IRunContext,
    ) -> None:
        """Log user-defined file artifacts from EXTRAS.mlflow_artifacts.

        Args:
            accessor: Configuration accessor
            run_context: Run context for logging
        """
        artifacts = accessor.get_mlflow_artifacts()
        if not artifacts:
            return

        for artifact_path in artifacts:
            try:
                path = Path(artifact_path)
                if path.exists() and path.is_file():
                    artifact_dir = str(path.parent) if path.parent != Path(".") else ""
                    run_context.log_artifact(path, artifact_dir=artifact_dir)
                    logger.debug("Logged artifact {}", artifact_path)
                else:
                    logger.warning("Artifact not found or not a file: {}", artifact_path)
            except Exception as e:
                logger.warning("Failed to log artifact '{}': {}", artifact_path, e)

    def _log_user_toml_artifacts(
        self,
        accessor: ConfigAccessor,
        run_context: IRunContext,
    ) -> None:
        """Log user-defined dicts as TOML artifacts from EXTRAS.mlflow_artifacts_toml.

        Converts dict values to TOML strings and logs them via ``log_text``
        (no temporary files).

        Args:
            accessor: Configuration accessor
            run_context: Run context for logging
        """
        artifacts_toml = accessor.get_mlflow_artifacts_toml()
        if not artifacts_toml:
            return

        from dlkit.infrastructure.io.config import serialize_config_to_string

        for name, data_dict in artifacts_toml.items():
            try:
                if not isinstance(data_dict, dict):
                    logger.warning("Skipping non-dict TOML artifact '{}'", name)
                    continue

                toml_str = serialize_config_to_string(data_dict, exclude_none=True)
                run_context.log_artifact_content(toml_str, f"config/{name}.toml")
                logger.debug("Logged TOML artifact {}.toml", name)

            except Exception as e:
                logger.warning("Failed to log TOML artifact '{}': {}", name, e)
