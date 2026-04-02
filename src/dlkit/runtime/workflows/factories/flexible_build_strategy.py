"""Flexible-array workflow build strategy."""

from __future__ import annotations

from typing import Any

from loguru import logger

from dlkit.runtime.execution.components import RuntimeComponents
from dlkit.shared.shapes import ShapeSummary
from dlkit.tools.config.data_entries import DataEntry
from dlkit.tools.config.model_components import WrapperComponentSettings

from .build_strategy import IBuildStrategy, WorkflowSettings, build_trainer
from .dataset_builder import DatasetBuilder
from .feature_pipeline import FeaturePipeline
from .module_defaults import with_runtime_module_defaults
from .shape_inference_pipeline import ShapeInferencePipeline


class FlexibleBuildStrategy(IBuildStrategy):
    """Default build strategy for flexible array-like datasets."""

    def __init__(
        self,
        feature_pipeline: FeaturePipeline | None = None,
        dataset_builder: DatasetBuilder | None = None,
        shape_inference: ShapeInferencePipeline | None = None,
    ) -> None:
        self._feature_pipeline = feature_pipeline or FeaturePipeline()
        self._dataset_builder = dataset_builder or DatasetBuilder()
        self._shape_inference = shape_inference or ShapeInferencePipeline()

    def can_handle(self, settings: WorkflowSettings) -> bool:
        try:
            ds_type = getattr(settings.DATASET, "type", None)
            return ds_type is None or str(ds_type).lower() == "flexible"
        except Exception:
            return True

    def _build_core(self, settings: WorkflowSettings) -> RuntimeComponents:
        context = self._dataset_builder.build_context(settings)
        ds_settings = with_runtime_module_defaults(settings.DATASET)
        if ds_settings is None:
            raise ValueError("DATASET settings are required but not configured")

        configured_features: tuple[DataEntry, ...] = tuple(
            getattr(ds_settings, "features", ()) or ()
        )
        configured_targets: tuple[DataEntry, ...] = tuple(getattr(ds_settings, "targets", ()) or ())
        training_settings = settings.TRAINING
        if training_settings is None:
            raise ValueError("TRAINING settings are required but not configured")
        selection = self._feature_pipeline.select(
            configured_features,
            configured_targets,
            getattr(training_settings, "loss_function", None),
            tuple(getattr(training_settings, "metrics", ()) or ()),
        )

        selected_feature_names = sorted(
            {
                entry.name
                for entry in selection.features
                if isinstance(entry.name, str) and entry.name
            }
        )
        logger.debug(
            "Flexible feature selection: selected={} dropped={} reasons={}",
            selected_feature_names,
            selection.dropped_feature_names,
            selection.dependency_reasons,
        )

        dataset = self._dataset_builder.build_flexible_dataset(
            settings,
            context,
            selection.features,
            selection.targets,
        )
        index_split = self._dataset_builder.build_split(settings, dataset)
        datamodule = self._dataset_builder.build_datamodule(
            settings,
            context,
            dataset,
            index_split,
        )

        entry_configs: tuple[DataEntry, ...] = tuple([*selection.features, *selection.targets])
        if entry_configs:
            from dlkit.runtime.workflows.entry_registry import DataEntryRegistry

            DataEntryRegistry.get_instance().register_entries(
                {entry.name: entry for entry in entry_configs if entry.name is not None}
            )

        model_settings = with_runtime_module_defaults(settings.MODEL)
        if model_settings is None:
            raise ValueError("MODEL settings are required but not configured")
        from . import build_factory as build_factory_module

        model_type = build_factory_module.detect_model_type(model_settings)

        shape_summary: ShapeSummary | None = None
        if build_factory_module.requires_shape_spec(model_type):
            shape_summary = self._shape_inference.infer_flexible(
                model_settings.name,
                dataset,
                selection.features,
                selection.targets,
            )

        wrapper_kwargs: dict[str, Any] = {
            "optimizer": training_settings.optimizer,
            "loss_function": training_settings.loss_function,
            "metrics": training_settings.metrics,
        }
        if training_settings.scheduler is not None:
            wrapper_kwargs["scheduler"] = training_settings.scheduler
        wrapper_settings = with_runtime_module_defaults(WrapperComponentSettings(**wrapper_kwargs))

        from dlkit.runtime.workflows.factories.component_builders import build_wrapper_components

        from . import build_factory as build_factory_module

        components = build_wrapper_components(wrapper_settings, entry_configs)
        model = build_factory_module.WrapperFactory.create_standard_wrapper(
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
            meta={"dataset_type": "flexible"},
        )
