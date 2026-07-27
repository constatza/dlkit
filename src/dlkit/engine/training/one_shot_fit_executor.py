"""One-shot, non-gradient fit execution for ``Fittable`` models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch

from dlkit.common import ModelState, TrainingResult
from dlkit.common.errors import WorkflowError
from dlkit.common.protocols import IDataModule, ITrainableModule
from dlkit.engine.artifacts import (
    FileArtifactPayload,
    ProducedArtifact,
    attach_checkpoint_artifacts,
)
from dlkit.infrastructure.config.job_config import JobConfig
from dlkit.infrastructure.config.model_components import ModelComponentSettings, extract_init_kwargs
from dlkit.infrastructure.io.path_context import resolve_with_context
from dlkit.infrastructure.seeding.service import apply_global_seed
from dlkit.infrastructure.utils.logging_config import get_logger

from .components import RuntimeComponents
from .fittable import Fittable
from .interfaces import ITrainingExecutor

logger = get_logger(__name__)

# Mirrors dlkit.engine.workflows.factories.module_defaults._MODEL_MODULE. Not
# imported directly: engine.training sits below engine.workflows.factories in
# the package DAG (see tach.toml) and must not depend on it.
_DEFAULT_MODEL_MODULE = "dlkit.domain.nn"

_CHECKPOINT_FILENAME = "fitted.ckpt"


def _resolve_checkpoint_path() -> Path:
    """Resolve where this run's fitted checkpoint should be written locally.

    Uses the general-purpose ``PathResolver`` precedence (explicit path
    override context > ``DLKIT_ROOT_DIR`` > ``run.root_dir`` > cwd) rather
    than the ``training.trainer.default_root_dir`` convention the trainer
    path uses: ``FitJobConfig`` has no ``training`` section by design (that's
    the entire point of the job kind), so a checkpoint-write location must
    not depend on one being configured. Always resolves to something (falls
    back to cwd), so unlike the trainer path this never raises for a missing
    root.

    Returns:
        Local path the fitted checkpoint should be written to.
    """
    return resolve_with_context("checkpoints") / _CHECKPOINT_FILENAME


def _serialize_model_settings(model_settings: ModelComponentSettings) -> dict[str, Any]:
    """Build the minimal ``dlkit_metadata.model_settings`` DTO shape.

    Mirrors ``DLKitCheckpointSerializer._serialize_model_settings`` (
    ``engine.adapters.lightning.concerns.checkpoint_serializer``) closely
    enough for ``extract_model_settings()``/``build_model_from_checkpoint()``
    to reconstruct the model, but without that helper's ``ShapeContext``
    shape-kwarg stripping: ``FitBuildStrategy`` never builds a Fittable model
    with a ``ShapeContext`` (``build_model(..., context=None)``), so there
    are no shape kwargs to strip. Not imported directly because
    ``engine.training`` must not depend on ``engine.adapters.lightning``
    (see ``tach.toml``).

    Args:
        model_settings: Resolved model configuration settings.

    Returns:
        Dict matching ``ModelCheckpointDTO``'s shape (``name``,
        ``module_path``, ``hyper_kwargs``).
    """
    name = model_settings.name
    if isinstance(name, type):
        serialized_name = name.__qualname__
        serialized_module = name.__module__
    else:
        serialized_name = str(name)
        serialized_module = model_settings.module_path or _DEFAULT_MODEL_MODULE
    return {
        "name": serialized_name,
        "module_path": serialized_module,
        "hyper_kwargs": extract_init_kwargs(model_settings),
    }


def _build_checkpoint(model: Fittable, model_settings: ModelComponentSettings) -> dict[str, Any]:
    """Build a Lightning-checkpoint-shaped dict for a fitted model.

    Args:
        model: The fitted model (also an ``nn.Module``/``LightningModule``).
        model_settings: Resolved model configuration settings.

    Returns:
        Checkpoint dict with ``state_dict`` and ``dlkit_metadata`` keys, in
        the same shape ``build_model_from_checkpoint()``/
        ``extract_model_settings()`` already know how to read.
    """
    state_dict = cast("torch.nn.Module", model).state_dict()
    checkpoint: dict[str, Any] = {
        "state_dict": {f"model.{key}": value for key, value in state_dict.items()}
    }
    checkpoint["dlkit_metadata"] = {
        "wrapper_type": type(model).__name__,
        "model_settings": _serialize_model_settings(model_settings),
        "entry_configs": [],
        "feature_names": [],
        "forward_arg_map": {},
        "predict_target_key": "",
        "model_family": "external",
        "input_shapes": None,
        "output_shapes": None,
    }
    return checkpoint


class OneShotFitExecutor(ITrainingExecutor):
    """Execute a one-shot, non-gradient fit for a ``Fittable`` model.

    No epochs, optimizer, or loss — the entire "training" is a single
    deterministic ``model.fit(dataloader)`` call, with no Lightning
    ``Trainer`` involved. Mirrors ``VanillaExecutor``'s shape (settings
    guard, seeding chokepoint, ``WorkflowError`` wrapping) but replaces
    ``trainer.fit()`` with a direct call into the model's own fit method,
    and builds/writes the checkpoint itself since there is no ``Trainer``
    (and therefore no ``ModelCheckpoint`` callback) to produce one.
    """

    def execute(self, components: RuntimeComponents, settings: object) -> TrainingResult:
        """Execute the one-shot fit workflow.

        Args:
            components: Pre-built runtime components (model, datamodule; no trainer).
            settings: Global job settings (must be a JobConfig instance).

        Returns:
            TrainingResult with the fitted model state.

        Raises:
            TypeError: If settings is not a JobConfig instance.
            WorkflowError: If the model does not implement Fittable or the
                fit call raises.
        """
        if not isinstance(settings, JobConfig):
            raise TypeError(
                f"OneShotFitExecutor.execute requires a JobConfig, got {type(settings).__name__}"
            )
        # Reseed immediately before fit() for dataloader-worker determinism —
        # same chokepoint VanillaExecutor uses before trainer.fit().
        apply_global_seed(settings.run.resolve_seed(), workers=True)

        model = components.model
        datamodule = components.datamodule

        if not isinstance(model, Fittable):
            raise WorkflowError(
                f"Model {type(model).__name__} does not implement Fittable",
                {"stage": "execute"},
            )

        try:
            model.fit(datamodule.train_dataloader())
        except Exception as e:
            raise WorkflowError(f"One-shot fit failed: {e}", {"stage": "fit"}) from e

        model_settings = settings.model
        if model_settings is None:
            raise WorkflowError(
                "MODEL settings are required to write a checkpoint", {"stage": "fit"}
            )

        checkpoint = _build_checkpoint(model, model_settings)

        checkpoint_path = _resolve_checkpoint_path()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)

        artifact = ProducedArtifact(
            kind="checkpoint",
            artifact_path=_CHECKPOINT_FILENAME,
            payload=FileArtifactPayload(file_path=checkpoint_path),
        )
        # Attach to the live model instance (not the returned TrainingResult):
        # RuntimeComponents is a frozen snapshot TrackingDecorator takes
        # *before* this method runs, so a newly-built RuntimeComponents
        # returned from here would never be read by ArtifactLogger. See
        # attach_checkpoint_artifacts's docstring for the full rationale.
        attach_checkpoint_artifacts(model, (artifact,))

        return TrainingResult(
            model_state=ModelState(
                model=cast(ITrainableModule, model),
                datamodule=cast(IDataModule, datamodule),
                trainer=None,
                settings=settings,
            ),
            metrics={},
            artifacts={},
            duration_seconds=0.0,
            predictions=None,
        )
