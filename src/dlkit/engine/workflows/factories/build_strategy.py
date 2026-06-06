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
from dlkit.infrastructure.config.core.factories import FactoryProvider
from dlkit.infrastructure.config.data_entries import DataEntry
from dlkit.infrastructure.config.enums import DatasetFamily
from dlkit.infrastructure.config.model_components import WrapperComponentSettings
from dlkit.infrastructure.config.workflow_configs import (
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)

from .component_builders import build_wrapper_components
from .dataset_builder import DatasetBuilder
from .model_detection import should_skip_wrapper
from .module_defaults import with_runtime_module_defaults

type WorkflowSettings = TrainingWorkflowConfig | OptimizationWorkflowConfig

# Dataset type constants for can_handle() checks
DATASET_TYPE_FLEXIBLE = "flexible"
DATASET_TYPE_GRAPH = "graph"
DATASET_TYPE_TIMESERIES = "timeseries"


def build_trainer(settings: WorkflowSettings) -> Trainer | None:
    """Build the trainer when the workflow is in training mode."""
    if not settings.SESSION or settings.SESSION.is_inference_mode or not settings.TRAINING:
        return None

    trainer_settings = settings.TRAINING.trainer
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
    return trainer_settings.build(session=settings.SESSION)


def _requires_explicit_local_root(trainer_settings: Any) -> bool:
    """Return whether trainer components may emit local files that need pinning."""
    if getattr(trainer_settings, "enable_checkpointing", False):
        return True

    if getattr(trainer_settings.logger, "name", None):
        return True

    for callback in getattr(trainer_settings, "callbacks", ()):
        if getattr(callback, "name", None) == "ModelCheckpoint":
            return True

    return False


def _pin_lightning_local_outputs(trainer_settings: Any) -> Any:
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
        if callback_name == "ModelCheckpoint" and dirpath is None:
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
        if settings.TRAINING is None:
            raise ValueError("TRAINING settings are required but not configured")
        wrapper_kwargs: dict[str, Any] = {
            "optimizer": settings.TRAINING.optimizer,
            "loss_function": settings.TRAINING.loss_function,
            "metrics": settings.TRAINING.metrics,
        }
        if settings.TRAINING.scheduler is not None:
            wrapper_kwargs["scheduler"] = settings.TRAINING.scheduler
        wrapper_settings = with_runtime_module_defaults(WrapperComponentSettings(**wrapper_kwargs))

        components = build_wrapper_components(wrapper_settings, entry_configs)
        model_settings = with_runtime_module_defaults(settings.MODEL)
        if model_settings is None:
            raise ValueError("MODEL settings are required but not configured")
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


class TimeSeriesBuildStrategy(IBuildStrategy):
    """Build strategy for timeseries datasets and wrappers."""

    def __init__(
        self,
        dataset_builder: DatasetBuilder | None = None,
    ) -> None:
        self._dataset_builder = dataset_builder or DatasetBuilder()

    def can_handle(self, settings: WorkflowSettings) -> bool:
        from dlkit.engine.data.families import resolve_family

        return resolve_family(settings) is DatasetFamily.TIMESERIES

    def _build_core(self, settings: WorkflowSettings) -> RuntimeComponents:
        context = self._dataset_builder.build_context(settings)
        dataset = self._dataset_builder.build_dataset_with_tensor_entries(settings, context)
        datamodule, split_artifact = _build_datamodule(
            settings,
            self._dataset_builder,
            dataset,
            family=DatasetFamily.TIMESERIES,
        )

        entry_configs: tuple[DataEntry, ...] = ()
        model_settings = with_runtime_module_defaults(settings.MODEL)
        if model_settings is None:
            raise ValueError("MODEL settings are required but not configured")

        skip_wrapper = should_skip_wrapper(model_settings, dataset)

        if skip_wrapper:
            model = FactoryProvider.create_component(model_settings, context)
        else:
            if settings.TRAINING is None:
                raise ValueError("TRAINING settings are required but not configured")
            wrapper_kwargs: dict[str, Any] = {
                "optimizer": settings.TRAINING.optimizer,
                "loss_function": settings.TRAINING.loss_function,
                "metrics": settings.TRAINING.metrics,
            }
            if settings.TRAINING.scheduler is not None:
                wrapper_kwargs["scheduler"] = settings.TRAINING.scheduler
            wrapper_settings = with_runtime_module_defaults(
                WrapperComponentSettings(**wrapper_kwargs)
            )
            components = build_wrapper_components(wrapper_settings, entry_configs)
            model = WrapperFactory.create_timeseries_wrapper(
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
            meta={"dataset_type": "timeseries"},
        )
