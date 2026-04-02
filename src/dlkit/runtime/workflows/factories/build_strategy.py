"""Core build strategies and shared workflow build helpers."""

from __future__ import annotations

import contextlib
from typing import Any

from lightning.pytorch import LightningDataModule, LightningModule, Trainer

from dlkit.runtime.execution.components import RuntimeComponents
from dlkit.shared.shapes import ShapeSummary
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.core.factories import FactoryProvider
from dlkit.tools.config.data_entries import DataEntry
from dlkit.tools.config.enums import DatasetFamily
from dlkit.tools.config.model_components import WrapperComponentSettings
from dlkit.tools.config.workflow_configs import (
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)

from .dataset_builder import DatasetBuilder
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
        from dlkit.tools.io import locations

        if getattr(trainer_settings, "default_root_dir", None) is None:
            trainer_settings = trainer_settings.model_copy(
                update={"default_root_dir": locations.lightning_work_dir()}
            )
    except Exception:
        pass
    return trainer_settings.build(session=settings.SESSION)


class IBuildStrategy:
    """Template-method base class for runtime component strategies."""

    def can_handle(self, settings: WorkflowSettings) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def build(self, settings: WorkflowSettings) -> RuntimeComponents:
        """Apply precision/path context setup before strategy-specific work."""
        from dlkit.tools.config.precision.context import precision_override
        from dlkit.tools.io.path_context import get_current_path_context, path_override_context
        from dlkit.tools.io.paths import coerce_root_dir_to_absolute

        precision_strategy = settings.SESSION.get_precision_strategy()
        context = get_current_path_context()
        session_root_dir = coerce_root_dir_to_absolute(settings.SESSION.root_dir)
        needs_path_context = (not context or not context.root_dir) and session_root_dir

        precision_ctx = (
            precision_override(precision_strategy)
            if precision_strategy is not None
            else contextlib.nullcontext()
        )
        with precision_ctx:
            if needs_path_context:
                with path_override_context({"root_dir": session_root_dir}):
                    return self._build_core(settings)
            return self._build_core(settings)

    def _build_core(self, settings: WorkflowSettings) -> RuntimeComponents:  # pragma: no cover
        raise NotImplementedError


class GraphBuildStrategy(IBuildStrategy):
    """Build strategy for graph datasets and wrappers."""

    def __init__(
        self,
        dataset_builder: DatasetBuilder | None = None,
    ) -> None:
        self._dataset_builder = dataset_builder or DatasetBuilder()

    def can_handle(self, settings: WorkflowSettings) -> bool:
        from dlkit.runtime.data.families import resolve_family

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

        from dlkit.runtime.workflows.factories.component_builders import build_wrapper_components

        from . import build_factory as build_factory_module

        components = build_wrapper_components(wrapper_settings, entry_configs)
        model_settings = with_runtime_module_defaults(settings.MODEL)
        if model_settings is None:
            raise ValueError("MODEL settings are required but not configured")
        model: LightningModule = build_factory_module.WrapperFactory.create_graph_wrapper(
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
        from dlkit.runtime.data.families import resolve_family

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
        from . import build_factory as build_factory_module

        model_type = build_factory_module.detect_model_type(model_settings)
        shape_summary: ShapeSummary | None = None
        if build_factory_module.requires_shape_spec(model_type):
            shape_summary = self._shape_inference.infer_timeseries(dataset)

        skip_wrapper = False
        try:
            from dlkit.runtime.data.datasets.timeseries import ForecastingDataset

            if isinstance(dataset, ForecastingDataset):
                skip_wrapper = True
        except Exception:
            pass
        try:
            model_ref = getattr(model_settings, "name", None)
            model_cls = None
            if isinstance(model_ref, str):
                from dlkit.tools.utils.general import import_object

                model_cls = import_object(
                    model_ref, fallback_module=model_settings.module_path or ""
                )
            elif isinstance(model_ref, type):
                model_cls = model_ref
            if isinstance(model_cls, type):
                from lightning.pytorch import LightningModule as BaseLightningModule

                if issubclass(model_cls, BaseLightningModule):
                    skip_wrapper = True
        except Exception:
            pass

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
            from dlkit.runtime.workflows.factories.component_builders import (
                build_wrapper_components,
            )

            from . import build_factory as build_factory_module

            components = build_wrapper_components(wrapper_settings, entry_configs)
            model = build_factory_module.WrapperFactory.create_timeseries_wrapper(
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
