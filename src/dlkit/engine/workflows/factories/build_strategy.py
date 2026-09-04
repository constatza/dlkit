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
from dlkit.engine.training.transform_fitting import fit_transforms_if_needed
from dlkit.infrastructure.config.data_entries import DataEntry
from dlkit.infrastructure.config.enums import DatasetFamily
from dlkit.infrastructure.config.job_config import JobConfig
from dlkit.infrastructure.config.model_components import WrapperComponentSettings
from dlkit.infrastructure.config.trainer_settings import CallbackSettings, TrainerSettings
from dlkit.infrastructure.config.training_settings import StoppingSettings, TrainingSettings

from .component_builders import build_wrapper_components
from .dataset_builder import DatasetBuilder
from .module_defaults import with_runtime_module_defaults
from .run_output_paths import resolve_local_artifact_root

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

    trainer_settings = _inject_default_checkpoint_callback(trainer_settings, training.stopping)
    trainer_settings = _inject_early_stopping_callback(trainer_settings, training.stopping)

    if _requires_explicit_local_root(trainer_settings):
        default_root_dir = resolve_local_artifact_root(settings)
        if default_root_dir is None:
            raise WorkflowError(
                "TRAINING.trainer.default_root_dir is required when using local-output "
                "trainer components such as checkpointing, loggers, or "
                "ModelCheckpoint callbacks.",
                {"stage": "trainer_build", "component": "trainer.default_root_dir"},
            )
        trainer_settings = _pin_lightning_local_outputs(trainer_settings, default_root_dir)
    return trainer_settings.build(session=settings.run)


def _has_callback_named(trainer_settings: TrainerSettings, name: str) -> bool:
    """Return whether trainer_settings.callbacks already includes a callback with this name."""
    return any(getattr(cb, "name", None) == name for cb in trainer_settings.callbacks)


def _default_checkpoint_callback_settings(stopping: StoppingSettings) -> CallbackSettings:
    """Build CallbackSettings for a bounded-file-count default ModelCheckpoint.

    Reuses `stopping.monitor`/`stopping.direction` rather than inventing new
    config. `save_top_k=1` with a step/epoch-free filename bounds checkpoint
    files to at most 1: the current best.
    """
    # CallbackSettings (ComponentSettings) uses extra="allow": these kwargs are
    # accepted at runtime and forwarded to ModelCheckpoint's constructor, but
    # aren't declared fields, so ty can't verify them statically.
    return CallbackSettings(
        name="ModelCheckpoint",
        module_path="lightning.pytorch.callbacks",
        monitor=stopping.monitor,  # ty: ignore[unknown-argument]
        mode=stopping.direction,  # ty: ignore[unknown-argument]
        save_top_k=1,  # ty: ignore[unknown-argument]
        filename="best",  # ty: ignore[unknown-argument]
    )


def _inject_default_checkpoint_callback(
    trainer_settings: TrainerSettings, stopping: StoppingSettings
) -> TrainerSettings:
    """Auto-inject a monitored ModelCheckpoint when enabled and unconfigured.

    No-op unless `enable_checkpointing` is True and the user hasn't already
    supplied their own `ModelCheckpoint` callback.
    """
    if not trainer_settings.enable_checkpointing:
        return trainer_settings
    if _has_callback_named(trainer_settings, "ModelCheckpoint"):
        return trainer_settings

    default_callback = _default_checkpoint_callback_settings(stopping)
    return trainer_settings.model_copy(
        update={"callbacks": (*trainer_settings.callbacks, default_callback)}
    )


def _default_early_stopping_callback_settings(stopping: StoppingSettings) -> CallbackSettings:
    """Build CallbackSettings for EarlyStopping from StoppingSettings fields."""
    return CallbackSettings(
        name="EarlyStopping",
        module_path="lightning.pytorch.callbacks",
        monitor=stopping.monitor,  # ty: ignore[unknown-argument]
        patience=stopping.patience,  # ty: ignore[unknown-argument]
        mode=stopping.direction,  # ty: ignore[unknown-argument]
    )


def _inject_early_stopping_callback(
    trainer_settings: TrainerSettings, stopping: StoppingSettings
) -> TrainerSettings:
    """Auto-inject EarlyStopping when the user has opted in via `stopping.enabled`.

    No-op unless `stopping.enabled` is True and the user hasn't already
    supplied their own `EarlyStopping` callback. Never default-on: dlkit ran
    with no EarlyStopping at all until this field existed, and stopping
    training early is a training-semantics change (unlike checkpointing,
    which is purely additive).
    """
    if not stopping.enabled:
        return trainer_settings
    if _has_callback_named(trainer_settings, "EarlyStopping"):
        return trainer_settings

    default_callback = _default_early_stopping_callback_settings(stopping)
    return trainer_settings.model_copy(
        update={"callbacks": (*trainer_settings.callbacks, default_callback)}
    )


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


def _pin_lightning_local_outputs(
    trainer_settings: TrainerSettings, default_root_dir: Path
) -> TrainerSettings:
    """Contain Lightning-owned local writes under one default root.

    Args:
        trainer_settings: Trainer settings whose logger/callbacks may need pinning.
        default_root_dir: Resolved local artifact root (via
            ``resolve_local_artifact_root``) to pin local outputs under.

    Returns:
        TrainerSettings with logger/callback local paths pinned.
    """
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
        if callback_name == "ModelCheckpoint" and dirpath is None:
            callback = callback.model_copy(update={"dirpath": default_root_dir / "checkpoints"})
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
    split_resolution = dataset_builder.build_split_for_training(settings, dataset)
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
        """Build runtime components. Infrastructure context is applied by BuildFactory.

        Fits the model's batch transformer from the datamodule's train split
        immediately after construction — before any Trainer/Tuner exists. See
        ``engine.training.transform_fitting.fit_transforms_if_needed``.
        """
        components = self._build_core(settings)
        fit_transforms_if_needed(components.model, components.datamodule)
        return components

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
        from dlkit.engine.workflows.factories._dataset_helpers import graph_dataset_overrides

        data = settings.data
        if data is None:
            raise ValueError("DATASET settings are required but not configured")
        context = self._dataset_builder.build_context(settings)
        overrides = graph_dataset_overrides(data)
        dataset = self._dataset_builder.build_dataset(settings, context, overrides)
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
