"""Core build strategies and shared workflow build helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from lightning.pytorch import LightningDataModule, LightningModule, Trainer

from dlkit.common.shapes import ShapeSummary
from dlkit.engine.adapters.lightning.factories import WrapperFactory
from dlkit.engine.training.components import RuntimeComponents
from dlkit.infrastructure.config import GeneralSettings
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
from .model_detection import detect_model_type, requires_shape_spec, should_skip_wrapper
from .module_defaults import with_runtime_module_defaults
from .shape_inference_pipeline import ShapeInferencePipeline

type WorkflowSettings = GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig


def build_trainer(settings: WorkflowSettings) -> Trainer | None:
    """Build the trainer when the workflow is in training mode."""
    if (
        not settings.SESSION
        or getattr(settings.SESSION, "inference", False)
        or not settings.TRAINING
    ):
        return None

    trainer_settings = settings.TRAINING.trainer
    try:
        from dlkit.infrastructure.io import locations

        if getattr(trainer_settings, "default_root_dir", None) is None:
            trainer_settings = trainer_settings.model_copy(
                update={"default_root_dir": locations.lightning_work_dir()}
            )
    except Exception:
        pass
    return trainer_settings.build(session=settings.SESSION)


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
        index_split = self._dataset_builder.build_split(settings, dataset)
        datamodule: LightningDataModule = self._dataset_builder.build_datamodule(
            settings,
            context,
            dataset,
            index_split,
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
            shape_spec=None,
            meta={"dataset_type": "graph"},
        )


class TimeSeriesBuildStrategy(IBuildStrategy):
    """Build strategy for timeseries datasets and wrappers."""

    def __init__(
        self,
        dataset_builder: DatasetBuilder | None = None,
        shape_inference: ShapeInferencePipeline | None = None,
    ) -> None:
        self._dataset_builder = dataset_builder or DatasetBuilder()
        self._shape_inference = shape_inference or ShapeInferencePipeline()

    def can_handle(self, settings: WorkflowSettings) -> bool:
        from dlkit.engine.data.families import resolve_family

        return resolve_family(settings) is DatasetFamily.TIMESERIES

    def _build_core(self, settings: WorkflowSettings) -> RuntimeComponents:
        context = self._dataset_builder.build_context(settings)
        dataset = self._dataset_builder.build_dataset_with_tensor_entries(settings, context)
        index_split = self._dataset_builder.build_split(settings, dataset)
        datamodule = self._dataset_builder.build_datamodule(
            settings,
            context,
            dataset,
            index_split,
            family=DatasetFamily.TIMESERIES,
        )

        entry_configs: tuple[DataEntry, ...] = ()
        model_settings = with_runtime_module_defaults(settings.MODEL)
        if model_settings is None:
            raise ValueError("MODEL settings are required but not configured")
        model_type = detect_model_type(model_settings)
        shape_summary: ShapeSummary | None = None
        if requires_shape_spec(model_type):
            shape_summary = self._shape_inference.infer_timeseries(dataset)

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
                shape_summary=shape_summary,
                entry_configs=entry_configs,
                components=components,
            )

        return RuntimeComponents(
            model=model,
            datamodule=datamodule,
            trainer=build_trainer(settings),
            shape_spec=shape_summary,
            meta={"dataset_type": "timeseries"},
        )
