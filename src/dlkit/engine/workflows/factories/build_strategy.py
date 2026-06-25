"""Core build strategies and shared workflow build helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from lightning.pytorch import LightningDataModule, LightningModule, Trainer

from dlkit.common.errors import WorkflowError
from dlkit.engine.adapters.lightning.factories import WrapperFactory
from dlkit.engine.artifacts import ProducedArtifact, RuntimeArtifactManifest
from dlkit.engine.training.components import RuntimeComponents
from dlkit.infrastructure.config.data_entries import DataEntry
from dlkit.infrastructure.config.enums import DatasetFamily
from dlkit.infrastructure.config.job_config import JobConfig
from dlkit.infrastructure.config.model_components import WrapperComponentSettings
from dlkit.infrastructure.config.trainer_settings import TrainerSettings
from dlkit.infrastructure.config.training_settings import TrainingSettings

from .component_builders import build_wrapper_components
from .dataset_builder import DatasetBuilder
from .module_defaults import with_runtime_module_defaults

type WorkflowSettings = JobConfig

# Dataset type constants for can_handle() checks
DATASET_TYPE_FLEXIBLE = "flexible"
DATASET_TYPE_GRAPH = "graph"


def _is_inference_mode(settings: JobConfig) -> bool:
    """Return True when the workflow is in inference mode.

    Args:
        settings: A JobConfig instance.

    Returns:
        True if the run type is ``"predict"``.
    """
    return settings.run.type == "predict"


def _get_training_settings(settings: JobConfig) -> TrainingSettings | None:
    """Extract training settings from JobConfig.

    Args:
        settings: A JobConfig instance.

    Returns:
        Training settings, or None when not configured.
    """
    return settings.training


def build_trainer(settings: JobConfig) -> Trainer | None:
    """Build the trainer when the workflow is in training mode.

    Args:
        settings: A JobConfig instance.

    Returns:
        Configured Trainer, or None for inference workflows.

    Raises:
        WorkflowError: If local-output trainer components require a root dir that is not set.
    """
    if _is_inference_mode(settings):
        return None

    training = settings.training
    if training is None:
        return None

    trainer_settings = training.trainer
    if trainer_settings is None:
        return None

    if _requires_explicit_local_root(trainer_settings):
        if getattr(trainer_settings, "default_root_dir", None) is None:
            raise WorkflowError(
                "TRAINING.trainer.default_root_dir is required when using local-output "
                "trainer components such as checkpointing, loggers, or "
                "ModelCheckpoint callbacks.",
                {"stage": "trainer_build", "component": "trainer.default_root_dir"},
            )
        trainer_settings = _pin_lightning_local_outputs(trainer_settings)
    return trainer_settings.build(session=None)


def _requires_explicit_local_root(trainer_settings: TrainerSettings) -> bool:
    """Return whether trainer components may emit local files that need pinning."""
    if getattr(trainer_settings, "enable_checkpointing", False):
        return True

    if getattr(trainer_settings.logger, "name", None):
        return True

    for callback in getattr(trainer_settings, "callbacks", ()):
        if getattr(callback, "name", None) == "ModelCheckpoint":
            return True

    return False


def _pin_lightning_local_outputs(trainer_settings: TrainerSettings) -> TrainerSettings:
    """Contain Lightning-owned local writes under one default root."""
    default_root_dir = trainer_settings.default_root_dir
    updates: dict[str, Any] = {}

    if getattr(trainer_settings.logger, "name", None):
        updates["logger"] = trainer_settings.logger.model_copy(
            update={"save_dir": default_root_dir}
        )

    pinned_callbacks = []
    callbacks_changed = False
    for callback in trainer_settings.callbacks:
        callback_name = getattr(callback, "name", None)
        dirpath = getattr(callback, "dirpath", None)
        if callback_name == "ModelCheckpoint" and dirpath is None and default_root_dir is not None:
            callback = callback.model_copy(
                update={"dirpath": Path(default_root_dir) / "checkpoints"}
            )
            callbacks_changed = True
        pinned_callbacks.append(callback)

    if callbacks_changed:
        updates["callbacks"] = tuple(pinned_callbacks)

    if not updates:
        return trainer_settings
    return trainer_settings.model_copy(update=updates)


def _build_datamodule(
    settings: WorkflowSettings,
    dataset_builder: DatasetBuilder,
    dataset: object,
    family: DatasetFamily | None = None,
) -> tuple[LightningDataModule, ProducedArtifact]:
    """Shared helper to build split and datamodule for strategies.

    Args:
        settings: The workflow settings.
        dataset_builder: The DatasetBuilder instance.
        dataset: The constructed dataset object.
        family: Optional DatasetFamily for datamodule defaults.

    Returns:
        Constructed LightningDataModule and the split artifact used by the run.
    """
    context = dataset_builder.build_context(settings)
    split_resolution = dataset_builder.build_split(settings, dataset)
    datamodule = dataset_builder.build_datamodule(
        settings,
        context,
        dataset,
        split_resolution,
        family=family,
    )
    split_artifact = dataset_builder.build_split_artifact(split_resolution)
    return datamodule, split_artifact


class IBuildStrategy(ABC):
    """Abstract base class for runtime component build strategies."""

    @abstractmethod
    def can_handle(self, settings: WorkflowSettings) -> bool:
        """Return True if this strategy can build components for the given settings."""

    def build(self, settings: WorkflowSettings) -> RuntimeComponents:
        """Build runtime components. Infrastructure context is applied by BuildFactory."""
        return self._build_core(settings)

    @abstractmethod
    def _build_core(self, settings: WorkflowSettings) -> RuntimeComponents:
        """Construct and return runtime components for the given settings."""


class GraphBuildStrategy(IBuildStrategy):
    """Build strategy for graph datasets and wrappers."""

    def __init__(
        self,
        dataset_builder: DatasetBuilder | None = None,
    ) -> None:
        self._dataset_builder = dataset_builder or DatasetBuilder()

    def can_handle(self, settings: WorkflowSettings) -> bool:
        from dlkit.engine.data.families import resolve_family

        return resolve_family(settings) is DatasetFamily.GRAPH

    def _build_core(self, settings: WorkflowSettings) -> RuntimeComponents:
        context = self._dataset_builder.build_context(settings)
        dataset = self._dataset_builder.build_dataset_with_tensor_entries(settings, context)
        datamodule, split_artifact = _build_datamodule(
            settings,
            self._dataset_builder,
            dataset,
            family=DatasetFamily.GRAPH,
        )

        entry_configs: tuple[DataEntry, ...] = ()
        training = _get_training_settings(settings)
        if training is None:
            raise ValueError("training settings are required but not configured")
        loss_fn = training.loss
        wrapper_kwargs: dict[str, Any] = {
            "optimizer": training.optimizer,
            "metrics": training.metrics,
        }
        if loss_fn is not None:
            wrapper_kwargs["loss_function"] = loss_fn
        wrapper_settings = with_runtime_module_defaults(WrapperComponentSettings(**wrapper_kwargs))

        components = build_wrapper_components(wrapper_settings, entry_configs)
        model_settings = with_runtime_module_defaults(settings.model)
        if model_settings is None:
            raise ValueError("model settings are required but not configured")
        model: LightningModule = WrapperFactory.create_graph_wrapper(
            model_settings=model_settings,
            settings=wrapper_settings,
            entry_configs=entry_configs,
            components=components,
        )
        return RuntimeComponents(
            model=model,
            datamodule=datamodule,
            trainer=build_trainer(settings),
            artifacts=RuntimeArtifactManifest(split_artifact=split_artifact),
            meta={"dataset_type": "graph"},
        )
