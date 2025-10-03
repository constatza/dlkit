"""Build factory for constructing runtime components (model, datamodule, trainer).

Phase 1 implementation: flexible dataset support with shape inference.
This replaces the legacy build_model_state() flow with a SOLID factory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from pathlib import Path

from lightning.pytorch import LightningDataModule, LightningModule, Trainer

from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.core.context import BuildContext
from dlkit.tools.config.core.factories import FactoryProvider
from dlkit.core.datatypes.split import IndexSplit
from dlkit.runtime.workflows.selectors.defaults import FamilyDefaults
from dlkit.core.shape_specs import IShapeSpec, ShapeInferenceEngine, ShapeSystemFactory, InferenceContext
from dlkit.runtime.workflows.entry_registry import DataEntryRegistry
from dlkit.tools.config.components.model_components import WrapperComponentSettings
from dlkit.core.models.wrappers.factories import WrapperFactory
from dlkit.tools.io.split_provider import get_or_create_split
from .model_detection import detect_model_type, ModelType, requires_shape_spec


@dataclass(frozen=True)
class BuildComponents:
    """Runtime components constructed from settings."""

    model: LightningModule
    datamodule: LightningDataModule
    trainer: Trainer | None
    shape_spec: IShapeSpec | None
    meta: dict[str, Any]


class IBuildStrategy:
    """Interface for build strategies (dataset/wrapper families)."""

    def can_handle(self, settings: GeneralSettings) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def build(self, settings: GeneralSettings) -> BuildComponents:  # pragma: no cover - interface
        raise NotImplementedError


class FlexibleBuildStrategy(IBuildStrategy):
    """Default build strategy for flexible array-like datasets."""

    def can_handle(self, settings: GeneralSettings) -> bool:
        # Default/fallback strategy; always true unless an explicit dataset type is provided
        try:
            ds = settings.DATASET
            ds_type = getattr(ds, "type", None)
            if ds_type is None:
                return True
            ds_type_str = str(ds_type).lower()
            return ds_type_str == "flexible"
        except Exception:
            return True

    def build(self, settings: GeneralSettings) -> BuildComponents:
        # Determine mode
        mode = (
            "inference"
            if (settings.SESSION and getattr(settings.SESSION, "inference", False))
            else "training"
        )
        # Use environment root as working directory for imports and relative paths
        try:
            from dlkit.tools.io.locations import root as _root

            cfg_dir = _root()
        except Exception:
            cfg_dir = Path.cwd()
        context = BuildContext(mode=mode, working_directory=cfg_dir)

        # Build dataset with legacy compatibility for SupervisedArrayDataset x/y
        ds_settings = settings.DATASET
        dataset = None
        if ds_settings is not None:
            ds_name = str(getattr(ds_settings, "name", "")).lower()
            # Translate legacy x/y -> flexible entries when helpful
            legacy_x = getattr(ds_settings, "x", None)
            legacy_y = getattr(ds_settings, "y", None)
            has_flexible_entries = bool(getattr(ds_settings, "features", ())) or bool(
                getattr(ds_settings, "targets", ())
            )
            if "supervisedarraydataset" in ds_name:
                from dlkit.core.datasets.flexible import FlexibleDataset

                if (legacy_x is not None) and not has_flexible_entries:
                    # Mirror x for targets if y is None to preserve old behavior
                    x_path = legacy_x
                    y_path = legacy_y or legacy_x
                    # Build FlexibleDataset directly to avoid pydantic FilePath constraints on-the-fly
                    dataset = FlexibleDataset(features={"x": x_path}, targets={"y": y_path})
                else:
                    # Build from flexible entries provided in settings
                    dataset = FlexibleDataset(
                        features=ds_settings.features, targets=ds_settings.targets
                    )
        # Default factory construction
        if dataset is None:
            dataset = FactoryProvider.create_component(ds_settings, context)

        # Get or create split (with caching)
        split_cfg = settings.DATASET.split
        index_split: IndexSplit = get_or_create_split(
            num_samples=len(dataset),
            test_ratio=split_cfg.test_ratio,
            val_ratio=split_cfg.val_ratio,
            session_name=settings.SESSION.name,
            explicit_filepath=split_cfg.filepath,
        )

        # Build datamodule with overrides (keep original settings object for test patching)
        # Apply dataloader overrides (e.g., batch_size, num_workers) via settings facade
        try:
            dm_loader_cfg = settings.DATAMODULE.get_dataloader_config()
        except Exception:
            dm_loader_cfg = getattr(settings.DATAMODULE, "dataloader", {})
        dm_context = context.with_overrides(
            dataset=dataset, split=index_split, dataloader=dm_loader_cfg
        )
        datamodule: LightningDataModule = FactoryProvider.create_component(
            settings.DATAMODULE, dm_context
        )

        # Build entry_configs (features + targets) if available in settings for transform-aware pipeline
        entry_configs: dict[str, Any] | None = None
        try:
            feats = getattr(settings.DATASET, "features", ()) or ()
            targs = getattr(settings.DATASET, "targets", ()) or ()
            if feats or targs:
                # Always provide entry configs so downstream steps know which entries
                # participate in loss/metrics, even without transforms.
                entry_configs = {**{e.name: e for e in feats}, **{e.name: e for e in targs}}
        except Exception:
            entry_configs = None

        # Register entry configs for user access
        if entry_configs:
            registry = DataEntryRegistry.get_instance()
            registry.register_entries(entry_configs)

        # Detect model type using ABC-based detection
        model_type = detect_model_type(settings.MODEL, settings)

        # Get appropriate shape specification if required
        shape_spec: IShapeSpec | None = None
        if requires_shape_spec(model_type):
            shape_factory = ShapeSystemFactory.create_production_system()
            inference_engine = ShapeInferenceEngine(shape_factory=shape_factory)
            shape_spec = inference_engine.infer_from_dataset(
                dataset=dataset,
                model_settings=settings.MODEL,
                entry_configs=entry_configs
            )

            # Validate shape inference for shape-aware models
            from dlkit.core.shape_specs import NullShapeSpec
            if isinstance(shape_spec, NullShapeSpec):
                raise ValueError(
                    f"Shape inference failed for shape-aware model '{settings.MODEL.name}'. "
                    f"Shape-aware models require shape information but shape inference returned no results. "
                    f"Please ensure your dataset provides shape information or configure shapes manually."
                )

        # Build wrapper settings with training optimizer/scheduler/loss/metrics configuration
        wrapper_kwargs = {
            "optimizer": settings.TRAINING.optimizer,
            "loss_function": settings.TRAINING.loss_function,
            "metrics": settings.TRAINING.metrics,
        }
        if settings.TRAINING.scheduler is not None:
            wrapper_kwargs["scheduler"] = settings.TRAINING.scheduler
        wrapper_settings = WrapperComponentSettings(**wrapper_kwargs)
        # Use family-compatible wrapper (factory kept for test monkeypatch on standard wrapper)
        # family resolved from dataset instance if needed; standard wrapper used for flexible
        model: LightningModule = WrapperFactory.create_standard_wrapper(
            model_settings=settings.MODEL,
            settings=wrapper_settings,
            shape_spec=shape_spec,
            entry_configs=entry_configs,
        )

        # Trainer (training mode only)
        trainer: Trainer | None = None
        if (
            settings.SESSION
            and (not getattr(settings.SESSION, "inference", False))
            and settings.TRAINING
        ):
            # Ensure default_root_dir is set to a standard location when unspecified
            trn_cfg = settings.TRAINING.trainer
            try:
                from dlkit.tools.io import locations

                if getattr(trn_cfg, "default_root_dir", None) is None:
                    trn_cfg = trn_cfg.model_copy(
                        update={"default_root_dir": locations.lightning_work_dir()}
                    )
            except Exception:
                pass
            trainer = trn_cfg.build()

        return BuildComponents(
            model=model,
            datamodule=datamodule,
            trainer=trainer,
            shape_spec=shape_spec,
            meta={"dataset_type": "flexible"},
        )


# Shape inference now handled by dlkit.runtime.workflows.shape_inference module




class BuildFactory:
    """Factory that selects an appropriate build strategy and constructs components."""

    def __init__(self, strategies: list[IBuildStrategy] | None = None) -> None:
        # Order: graph -> timeseries -> flexible (fallback)
        self._strategies = strategies or [
            GraphBuildStrategy(),
            TimeSeriesBuildStrategy(),
            FlexibleBuildStrategy(),
        ]

    def build_components(self, settings: GeneralSettings) -> BuildComponents:
        for strat in self._strategies:
            if strat.can_handle(settings):
                return strat.build(settings)
        # Fallback to flexible
        return FlexibleBuildStrategy().build(settings)


class GraphBuildStrategy(IBuildStrategy):
    """Build strategy for graph (PyG) datasets and models."""

    def can_handle(self, settings: GeneralSettings) -> bool:
        try:
            ds = settings.DATASET
            dm = settings.DATAMODULE
            explicit = getattr(ds, "type", None)
            if str(explicit).lower() == "graph":
                return True
            name_mod = f"{getattr(ds, 'name', '')} {getattr(ds, 'module_path', '')} {getattr(dm, 'name', '')} {getattr(dm, 'module_path', '')}".lower()
            return any(k in name_mod for k in ("graph", "pyg", "geometric"))
        except Exception:
            return False

    def build(self, settings: GeneralSettings) -> BuildComponents:
        mode = (
            "inference"
            if (settings.SESSION and getattr(settings.SESSION, "inference", False))
            else "training"
        )
        try:
            from dlkit.tools.config.environment import DLKitEnvironment

            cfg_dir = DLKitEnvironment().get_root_path()
        except Exception:
            cfg_dir = Path.cwd()
        context = BuildContext(mode=mode, working_directory=cfg_dir)

        # Allow passing a simple file path for features via an auxiliary key (features_path)
        # to avoid clashing with flexible DatasetSettings.features typing.
        ds_overrides: dict[str, Any] = {}
        try:
            ds_settings = settings.DATASET
            fp = getattr(ds_settings, "features_path", None)
            if fp is not None:
                ds_overrides["features"] = fp
        except Exception:
            pass

        dataset = FactoryProvider.create_component(
            settings.DATASET, context.with_overrides(**ds_overrides)
        )
        split_cfg = settings.DATASET.split
        index_split: IndexSplit = get_or_create_split(
            num_samples=len(dataset),
            test_ratio=split_cfg.test_ratio,
            val_ratio=split_cfg.val_ratio,
            session_name=settings.SESSION.name,
            explicit_filepath=split_cfg.filepath,
        )

        try:
            dm_loader_cfg = settings.DATAMODULE.get_dataloader_config()
        except Exception:
            dm_loader_cfg = getattr(settings.DATAMODULE, "dataloader", {})
        dm_context = context.with_overrides(
            dataset=dataset, split=index_split, dataloader=dm_loader_cfg
        )
        # Default to graph-aware datamodule when no explicit datamodule is provided
        dm_settings = settings.DATAMODULE
        try:
            name = getattr(dm_settings, "name", None)
            if not name or str(name) == "InMemoryModule":
                dm_class = FamilyDefaults.default_datamodule_class_for(settings)
                dm_settings = dm_settings.model_copy(update={"name": dm_class})  # type: ignore[attr-defined]
        except Exception:
            pass
        datamodule: LightningDataModule = FactoryProvider.create_component(dm_settings, dm_context)

        # Register entry configs for user access (graphs may not have explicit entry configs)
        entry_configs = None  # Graph strategy doesn't use flexible entry configs by default
        registry = DataEntryRegistry.get_instance()
        if entry_configs:
            registry.register_entries(entry_configs)

        # Use graph shape inference
        shape_factory = ShapeSystemFactory.create_production_system()
        inference_engine = ShapeInferenceEngine(shape_factory=shape_factory)
        shape_spec = inference_engine.infer_from_dataset(
            dataset=dataset,
            model_settings=settings.MODEL,
            entry_configs=entry_configs
        )

        # Build wrapper with shape hints and training optimizer/scheduler/loss/metrics configuration
        wrapper_kwargs = {
            "optimizer": settings.TRAINING.optimizer,
            "loss_function": settings.TRAINING.loss_function,
            "metrics": settings.TRAINING.metrics,
        }
        if settings.TRAINING.scheduler is not None:
            wrapper_kwargs["scheduler"] = settings.TRAINING.scheduler
        wrapper_settings = WrapperComponentSettings(**wrapper_kwargs)

        model: LightningModule = WrapperFactory.create_graph_wrapper(
            model_settings=settings.MODEL,
            settings=wrapper_settings,
            shape_spec=shape_spec,
            entry_configs=entry_configs,
        )

        trainer: Trainer | None = None
        if (
            settings.SESSION
            and (not getattr(settings.SESSION, "inference", False))
            and settings.TRAINING
        ):
            trn_cfg = settings.TRAINING.trainer
            try:
                from dlkit.tools.io import locations

                if getattr(trn_cfg, "default_root_dir", None) is None:
                    trn_cfg = trn_cfg.model_copy(
                        update={"default_root_dir": locations.lightning_work_dir()}
                    )
            except Exception:
                pass
            trainer = trn_cfg.build()

        return BuildComponents(
            model=model,
            datamodule=datamodule,
            trainer=trainer,
            shape_spec=shape_spec,
            meta={"dataset_type": "graph"},
        )


class TimeSeriesBuildStrategy(IBuildStrategy):
    """Build strategy for time series / forecasting datasets and models."""

    def can_handle(self, settings: GeneralSettings) -> bool:
        try:
            ds = settings.DATASET
            dm = settings.DATAMODULE
            explicit = getattr(ds, "type", None)
            if str(explicit).lower() == "timeseries":
                return True
            name_mod = f"{getattr(ds, 'name', '')} {getattr(ds, 'module_path', '')} {getattr(dm, 'name', '')} {getattr(dm, 'module_path', '')}".lower()
            return any(k in name_mod for k in ("timeseries", "forecast"))
        except Exception:
            return False

    def build(self, settings: GeneralSettings) -> BuildComponents:
        mode = (
            "inference"
            if (settings.SESSION and getattr(settings.SESSION, "inference", False))
            else "training"
        )
        try:
            from dlkit.tools.config.environment import DLKitEnvironment

            cfg_dir = DLKitEnvironment().get_root_path()
        except Exception:
            cfg_dir = Path.cwd()
        context = BuildContext(mode=mode, working_directory=cfg_dir)

        # Build dataset, allowing an auxiliary features_path to map to the constructor 'features'
        ds_overrides: dict[str, Any] = {}
        try:
            ds_settings = settings.DATASET
            fp = getattr(ds_settings, "features_path", None)
            if fp is not None:
                ds_overrides["features"] = fp
        except Exception:
            pass
        dataset = FactoryProvider.create_component(
            settings.DATASET, context.with_overrides(**ds_overrides)
        )
        split_cfg = settings.DATASET.split
        index_split: IndexSplit = get_or_create_split(
            num_samples=len(dataset),
            test_ratio=split_cfg.test_ratio,
            val_ratio=split_cfg.val_ratio,
            session_name=settings.SESSION.name,
            explicit_filepath=split_cfg.filepath,
        )

        try:
            dm_loader_cfg = settings.DATAMODULE.get_dataloader_config()
        except Exception:
            dm_loader_cfg = getattr(settings.DATAMODULE, "dataloader", {})
        dm_context = context.with_overrides(
            dataset=dataset, split=index_split, dataloader=dm_loader_cfg
        )
        # Default to time-series-aware datamodule when no explicit datamodule is provided
        dm_settings = settings.DATAMODULE
        try:
            name = getattr(dm_settings, "name", None)
            if not name or str(name) == "InMemoryModule":
                dm_class = FamilyDefaults.default_datamodule_class_for(settings)
                dm_settings = dm_settings.model_copy(update={"name": dm_class})  # type: ignore[attr-defined]
        except Exception:
            pass
        datamodule: LightningDataModule = FactoryProvider.create_component(dm_settings, dm_context)

        # Register entry configs for user access (timeseries may not have explicit entry configs)
        entry_configs = None  # Timeseries strategy doesn't use flexible entry configs by default
        registry = DataEntryRegistry.get_instance()
        if entry_configs:
            registry.register_entries(entry_configs)

        # Detect model type using ABC-based detection
        model_type = detect_model_type(settings.MODEL, settings)

        # Get appropriate shape specification if required
        shape_spec: IShapeSpec | None = None
        if requires_shape_spec(model_type):
            shape_factory = ShapeSystemFactory.create_production_system()
            inference_engine = ShapeInferenceEngine(shape_factory=shape_factory)
            shape_spec = inference_engine.infer_from_dataset(
                dataset=dataset,
                model_settings=settings.MODEL,
                entry_configs=entry_configs
            )

        # Decide whether to skip wrapper for PF-style datasets/models
        skip_wrapper = False
        try:
            # Skip when dataset is our PF wrapper
            from dlkit.core.datasets.timeseries import ForecastingDataset as _FDS  # type: ignore

            if isinstance(dataset, _FDS):
                skip_wrapper = True
        except Exception:
            pass
        try:
            # Skip when model is a LightningModule subclass (PF models inherit LightningModule)
            model_ref = getattr(settings.MODEL, "name", None)
            model_cls = None
            if isinstance(model_ref, str):
                from dlkit.tools.utils.general import import_object as _import

                model_cls = _import(model_ref, fallback_module=settings.MODEL.module_path)
            elif isinstance(model_ref, type):
                model_cls = model_ref
            if isinstance(model_cls, type):
                from lightning.pytorch import LightningModule as _LM

                if issubclass(model_cls, _LM):
                    skip_wrapper = True
        except Exception:
            pass

        if skip_wrapper:
            # Build model directly; do not inject pipeline wrapper
            model: LightningModule = FactoryProvider.create_component(settings.MODEL, context)
        else:
            # Otherwise, use the timeseries wrapper with inferred shape and training optimizer/scheduler/loss/metrics configuration
            wrapper_kwargs = {
                "optimizer": settings.TRAINING.optimizer,
                "loss_function": settings.TRAINING.loss_function,
                "metrics": settings.TRAINING.metrics,
            }
            if settings.TRAINING.scheduler is not None:
                wrapper_kwargs["scheduler"] = settings.TRAINING.scheduler
            wrapper_settings = WrapperComponentSettings(**wrapper_kwargs)

            model = WrapperFactory.create_timeseries_wrapper(
                model_settings=settings.MODEL,
                settings=wrapper_settings,
                shape_spec=shape_spec,
                entry_configs=entry_configs,
            )

        trainer: Trainer | None = None
        if (
            settings.SESSION
            and (not getattr(settings.SESSION, "inference", False))
            and settings.TRAINING
        ):
            trainer = settings.TRAINING.trainer.build()

        return BuildComponents(
            model=model,
            datamodule=datamodule,
            trainer=trainer,
            shape_spec=shape_spec,
            meta={"dataset_type": "timeseries"},
        )


# Graph shape inference now handled by dlkit.runtime.workflows.shape_inference module


# Timeseries shape inference now handled by dlkit.runtime.workflows.shape_inference module
