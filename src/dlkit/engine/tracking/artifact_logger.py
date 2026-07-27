"""Artifact logging service for MLflow tracking.

Single Responsibility: Log checkpoints, models, and user-defined artifacts to MLflow.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lightning.pytorch import Trainer
    from mlflow.models import ModelSignature

import numpy as np
from torch import nn

from dlkit.common import TrainingResult
from dlkit.common.hooks import ParamValue
from dlkit.engine.adapters.lightning.base import ProcessingLightningWrapper
from dlkit.engine.artifacts import (
    ArtifactPublisher,
    ContentArtifactPayload,
    FileArtifactPayload,
    ProducedArtifact,
    read_checkpoint_artifacts,
)
from dlkit.engine.training.checkpoint_utils import find_checkpoint_callback
from dlkit.engine.training.components import RuntimeComponents
from dlkit.infrastructure.config.job_config import JobConfig
from dlkit.infrastructure.utils.logging_config import get_logger

from .config_accessor import ConfigAccessor
from .interfaces import IExperimentTracker, IRunContext

type _WorkflowSettings = JobConfig
# MLflow pt2 requires ndarray/Tensor or a tuple of them — NOT a dict.
type _InputExample = np.ndarray | tuple[np.ndarray, ...]

logger = get_logger(__name__)

DEFAULT_MODEL_ARTIFACT_PATH = "model"
CHECKPOINT_ARTIFACT_DIR = "checkpoints"
TAG_LOGGED_MODEL_URI = "mlflow_logged_model_uri"
TAG_LOGGED_MODEL_ARTIFACT_PATH = "mlflow_logged_model_artifact_path"
TAG_MODEL_CLASS = "mlflow_model_class"


def _resolve_input_shapes(model: nn.Module) -> Mapping[str, tuple[int, ...]] | None:
    """Return checkpoint-declared input shapes, or None when unavailable.

    Args:
        model: Trained model, possibly carrying ``_checkpoint_metadata``.

    Returns:
        Mapping of input name to shape, or None when shape info is unavailable.
    """
    metadata = getattr(model, "_checkpoint_metadata", None)
    context = getattr(metadata, "context", None) if metadata is not None else None
    if context is None or not context.input_shapes:
        return None
    return context.input_shapes


def _build_input_example(model: nn.Module) -> _InputExample | None:
    """Build zero numpy arrays from checkpoint metadata input shapes.

    Returns a single ndarray for single-input models, or a tuple of ndarrays for
    multi-input models. MLflow pt2 validation iterates over the value and checks
    isinstance(v, (np.ndarray, Tensor)) — a dict fails this check because iteration
    yields string keys, not arrays.

    Args:
        model: Trained model, possibly carrying ``_checkpoint_metadata``.

    Returns:
        Single ndarray ``(1, *shape)`` or tuple thereof, or None when shape info
        is unavailable.
    """
    input_shapes = _resolve_input_shapes(model)
    if input_shapes is None:
        return None
    param = next(model.parameters(), None)
    np_dtype = np.float32 if param is None else param.detach().cpu().numpy().dtype
    arrays = [np.zeros((1, *shape), dtype=np_dtype) for shape in input_shapes.values()]
    return arrays[0] if len(arrays) == 1 else tuple(arrays)


def _build_pt2_signature(model: nn.Module) -> ModelSignature | None:
    """Build an MLflow ModelSignature with TensorSpec inputs for pt2 compatibility.

    pt2 serialization requires a TensorSpec-based signature in addition to
    input_example. Returns None when shape info is unavailable.

    Args:
        model: Trained model, possibly carrying ``_checkpoint_metadata``.

    Returns:
        ``ModelSignature`` with TensorSpec inputs, or None.
    """
    from mlflow.models import ModelSignature
    from mlflow.types.schema import Schema, TensorSpec

    input_shapes = _resolve_input_shapes(model)
    if input_shapes is None:
        return None
    param = next(model.parameters(), None)
    np_dtype = np.dtype("float32") if param is None else param.detach().cpu().numpy().dtype
    # Use static batch=1 — pt2 maps -1 to ExportDim("dynamic_dim") which fails
    # when the example is also batch=1 (torch.export specializes the constant).
    inputs = Schema(
        [
            TensorSpec(
                type=np_dtype,
                shape=(1, *shape),
                # Named schema forces MLflow's pyfunc predict into a dict path
                # that the pytorch flavor rejects; only disambiguate when >1 input.
                name=name if len(input_shapes) > 1 else None,
            )
            for name, shape in input_shapes.items()
        ]
    )
    return ModelSignature(inputs=inputs)


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
    *,
    remove_after_upload: bool,
) -> None:
    """Upload checkpoint to MLflow and optionally remove the local copy.

    The local file is only removed when the upload succeeds (exceptions from
    ``log_artifact`` propagate rather than being swallowed here) AND
    ``remove_after_upload`` is True — i.e. ``ArtifactPolicy.remove_uploaded_files``
    for the active run, which is only set for a genuinely remote, tracked
    backend. Local or untracked runs keep the on-disk checkpoint so
    ``TrainingResult.checkpoint_path`` always points at a file that still
    exists after ``api_train()`` returns.

    Args:
        run_context: Active ``IRunContext`` for logging.
        ckpt_path: Path to the checkpoint file.
        artifact_dir: Sub-path within the artifact store for uploaded files.
        remove_after_upload: Whether to delete the local file after a
            successful upload (``ArtifactPolicy.remove_uploaded_files``).
    """
    run_context.log_artifact(ckpt_path, artifact_dir)
    if not remove_after_upload:
        return
    try:
        ckpt_path.unlink()
        logger.debug("Removed local checkpoint after upload: {}", ckpt_path)
    except OSError as exc:
        logger.warning("Could not remove local checkpoint {}: {}", ckpt_path, exc)


def _log_trainer_checkpoints(
    trainer: Trainer | None,
    components: RuntimeComponents,
    run_context: IRunContext,
) -> None:
    """Log best/last checkpoints from a PyTorch Lightning trainer, if one ran.

    Args:
        trainer: The run's trainer, or None for a trainer-free (one-shot fit) run.
        components: Build components; used for the artifact-removal policy.
        run_context: Run context for logging.
    """
    if trainer is None:
        logger.debug("No trainer found in components")
        return

    ckpt_cb = find_checkpoint_callback(trainer)
    if not ckpt_cb:
        logger.debug("No ModelCheckpoint callback found")
        return

    best = getattr(ckpt_cb, "best_model_path", None)
    last = getattr(ckpt_cb, "last_model_path", None)
    if best is not None and not isinstance(best, str | Path):
        best = None
    if last is not None and not isinstance(last, str | Path):
        last = None

    remove_after_upload = components.artifacts.policy.remove_uploaded_files
    if best:
        _log_or_skip_checkpoint(
            run_context,
            Path(best),
            CHECKPOINT_ARTIFACT_DIR,
            remove_after_upload=remove_after_upload,
        )
        logger.debug("Logged best checkpoint {}", best)
    if last and last != best:
        _log_or_skip_checkpoint(
            run_context,
            Path(last),
            CHECKPOINT_ARTIFACT_DIR,
            remove_after_upload=remove_after_upload,
        )
        logger.debug("Logged last checkpoint {}", last)


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
        self._log_model_artifact(run_context=run_context, model=components.model, settings=settings)

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

        Logs best and last checkpoints from a PyTorch Lightning trainer when
        one ran, plus any checkpoint artifacts a trainer-free run (the
        one-shot fit path — see ``OneShotFitExecutor``) attached directly to
        the model via ``engine.artifacts.attach_checkpoint_artifacts``, since
        there is no ``Trainer``/``ModelCheckpoint`` callback to produce one
        for that path. Raises on failure so training aborts rather than
        silently missing the artifact.

        Args:
            components: Build components containing trainer
            run_context: Run context for logging
        """
        _log_trainer_checkpoints(getattr(components, "trainer", None), components, run_context)

        for artifact in read_checkpoint_artifacts(components.model):
            RunContextArtifactPublisher(run_context).publish(artifact)
            logger.debug("Logged one-shot-fit checkpoint artifact {}", artifact.artifact_path)

    def _log_model_artifact(
        self,
        *,
        run_context: IRunContext,
        model: nn.Module,
        settings: _WorkflowSettings,
    ) -> None:
        """Log the trained model as an MLflow artifact (no registry registration).

        Raises on failure so training aborts rather than silently missing the artifact.

        Args:
            run_context: Active run context for logging.
            model: Trained model to log.
            settings: Workflow settings controlling artifact serialization.
        """
        input_example = _build_input_example(model)
        signature = _build_pt2_signature(model)
        model_serialization_format = settings.tracking.model_serialization_format

        input_shapes = _resolve_input_shapes(model)
        if input_shapes is not None and len(input_shapes) > 1:
            logger.warning(
                "Model has {} inputs; MLflow's pytorch flavor does not support "
                "pyfunc/REST serving for multi-input models regardless of "
                "serialization_format. Load with mlflow.pytorch.load_model(uri), "
                "not mlflow.pyfunc.load_model(uri).predict().",
                len(input_shapes),
            )

        if model_serialization_format == "pt2" and input_example is None:
            raise ValueError(
                "PT2 model serialization requires checkpoint metadata with input shapes. "
                "Use model_serialization_format='pickle' or provide shape metadata."
            )

        model_uri = run_context.log_model(
            model=model,
            artifact_path=DEFAULT_MODEL_ARTIFACT_PATH,
            input_example=input_example,
            signature=signature,
            model_serialization_format=model_serialization_format,
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
