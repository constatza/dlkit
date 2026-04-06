"""Generative workflow build strategies."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from dlkit.engine.training.components import RuntimeComponents
from dlkit.infrastructure.config.core.context import BuildContext
from dlkit.infrastructure.config.core.factories import FactoryProvider
from dlkit.infrastructure.config.data_entries import DataEntry, FeatureType, TargetType
from dlkit.infrastructure.config.model_components import WrapperComponentSettings
from dlkit.infrastructure.types.split import IndexSplit

from .build_strategy import IBuildStrategy, WorkflowSettings, build_trainer
from .module_defaults import with_runtime_module_defaults


class GenerativeBuildStrategy(IBuildStrategy):
    """Base class for workflows with a GENERATIVE section."""

    def can_handle(self, settings: WorkflowSettings) -> bool:
        try:
            return getattr(settings, "GENERATIVE", None) is not None
        except Exception:
            return False


class FlowMatchingBuildStrategy(GenerativeBuildStrategy):
    """Build strategy for flow matching generative models."""

    def can_handle(self, settings: WorkflowSettings) -> bool:
        if not super().can_handle(settings):
            return False
        try:
            generative_settings = getattr(settings, "GENERATIVE", None)
            return getattr(generative_settings, "algorithm", None) == "flow_matching"
        except Exception:
            return False

    def _build_core(self, settings: WorkflowSettings) -> RuntimeComponents:
        from dlkit.domain.nn.generative.functions.solvers import euler_step, heun_step
        from dlkit.domain.nn.generative.samplers.noise import GaussianNoiseSampler
        from dlkit.domain.nn.generative.samplers.time import UniformTimeSampler
        from dlkit.domain.nn.generative.supervision import FlowMatchingSupervisionBuilder
        from dlkit.engine.adapters.lightning.base import _build_model_from_settings
        from dlkit.engine.adapters.lightning.flowmatching import FlowMatchingWrapper
        from dlkit.engine.adapters.lightning.generator_factories import (
            DeterministicGeneratorFactory,
        )
        from dlkit.engine.adapters.lightning.loss_routing import RoutedLossComputer
        from dlkit.engine.adapters.lightning.metrics_routing import RoutedMetricsUpdater
        from dlkit.engine.adapters.lightning.model_invoker import (
            ModelOutputSpec,
            TensorDictModelInvoker,
        )
        from dlkit.engine.adapters.lightning.prediction_strategies import ODEPredictionStrategy
        from dlkit.engine.adapters.lightning.transform_pipeline import NamedBatchTransformer
        from dlkit.engine.adapters.lightning.wrapper_types import WrapperCheckpointMetadata
        from dlkit.engine.data.datasets.flexible import FlexibleDataset
        from dlkit.engine.data.shape_inference import infer_post_transform_shapes
        from dlkit.infrastructure.io.locations import root as root_path
        from dlkit.infrastructure.io.split_provider import get_or_create_split

        gen_cfg = getattr(settings, "GENERATIVE", None)
        x1_key: str = getattr(gen_cfg, "x1_key", "x1")
        n_inference_steps: int = getattr(gen_cfg, "n_inference_steps", 100)
        solver_name: str = getattr(gen_cfg, "solver", "euler")
        val_seed: int = getattr(gen_cfg, "val_seed", 42)
        mode = (
            "inference"
            if (settings.SESSION and getattr(settings.SESSION, "inference", False))
            else "training"
        )
        try:
            cfg_dir = root_path()
        except Exception:
            cfg_dir = Path.cwd()
        context = BuildContext(mode=mode, working_directory=cfg_dir)

        ds_settings = with_runtime_module_defaults(settings.DATASET)
        configured_features: tuple[DataEntry, ...] = tuple(
            getattr(ds_settings, "features", ()) or ()
        )
        configured_targets: tuple[DataEntry, ...] = tuple(getattr(ds_settings, "targets", ()) or ())
        dataset = FlexibleDataset(
            features=cast(tuple[FeatureType, ...], configured_features),
            targets=cast(tuple[TargetType, ...], configured_targets),
            memmap_cache_dir=getattr(ds_settings, "resolved_memmap_cache_dir", None),
        )
        if settings.DATASET is None:
            raise ValueError("DATASET settings are required but not configured")
        split_cfg = settings.DATASET.split
        index_split: IndexSplit = get_or_create_split(
            num_samples=len(dataset),
            test_ratio=split_cfg.test_ratio,
            val_ratio=split_cfg.val_ratio,
            session_name=settings.SESSION.name,
            explicit_filepath=split_cfg.filepath,
        )
        datamodule_settings = with_runtime_module_defaults(settings.DATAMODULE)
        if datamodule_settings is None:
            raise ValueError("DATAMODULE settings are required but not configured")
        dm_context = context.with_overrides(
            dataset=dataset,
            split=index_split,
            dataloader=datamodule_settings.dataloader,
        )
        datamodule = FactoryProvider.create_component(datamodule_settings, dm_context)

        try:
            shape_summary = infer_post_transform_shapes(
                dataset, configured_features, configured_targets
            )
        except Exception:
            shape_summary = None

        model_settings = with_runtime_module_defaults(settings.MODEL)
        if model_settings is None:
            raise ValueError("MODEL settings are required but not configured")
        model = _build_model_from_settings(model_settings, shape_summary)
        supervision_builder = FlowMatchingSupervisionBuilder(
            x1_key=x1_key,
            time_sampler=UniformTimeSampler(),
            noise_sampler=GaussianNoiseSampler(),
        )
        output_spec = ModelOutputSpec()
        model_invoker = TensorDictModelInvoker(
            in_keys=[("features", "xt"), ("features", "t")],
            output_spec=output_spec,
        )
        batch_transformer = NamedBatchTransformer({}, {})
        if settings.TRAINING is None:
            raise ValueError("TRAINING settings are required but not configured")
        loss_fn = FactoryProvider.create_component(
            with_runtime_module_defaults(settings.TRAINING.loss_function),
            BuildContext(mode="training"),
        )
        loss_spec = settings.TRAINING.loss_function
        loss_computer = RoutedLossComputer(
            loss_fn=loss_fn,
            target_key=getattr(loss_spec, "target_key", None),
            default_target_key="ut",
            extra_inputs=tuple(getattr(loss_spec, "extra_inputs", ()) or ()),
        )
        metrics_updater = RoutedMetricsUpdater(val_routes=[], test_routes=[])
        solver_fn = euler_step if solver_name == "euler" else heun_step
        ode_strategy = ODEPredictionStrategy(
            x0_sampler=GaussianNoiseSampler(),
            solver=solver_fn,
            n_steps=n_inference_steps,
        )
        entry_configs: tuple[DataEntry, ...] = tuple([*configured_features, *configured_targets])
        wrapper_settings = with_runtime_module_defaults(
            WrapperComponentSettings(
                optimizer=settings.TRAINING.optimizer,
                loss_function=settings.TRAINING.loss_function,
            )
        )
        checkpoint_metadata = WrapperCheckpointMetadata(
            model_settings=model_settings,
            wrapper_settings=wrapper_settings,
            entry_configs=entry_configs,
            feature_names=(x1_key,),
            predict_target_key="ut",
            shape_summary=shape_summary,
            output_spec=output_spec,
        )
        model_wrapper = FlowMatchingWrapper(
            model=model,
            model_invoker=model_invoker,
            loss_computer=loss_computer,
            metrics_updater=metrics_updater,
            batch_transformer=batch_transformer,
            optimizer_settings=settings.TRAINING.optimizer,
            scheduler_settings=getattr(settings.TRAINING, "scheduler", None),
            predict_target_key="ut",
            checkpoint_metadata=checkpoint_metadata,
            ode_prediction_strategy=ode_strategy,
            supervision_builder=supervision_builder,
            val_generator_factory=DeterministicGeneratorFactory(base_seed=val_seed),
        )
        return RuntimeComponents(
            model=model_wrapper,
            datamodule=datamodule,
            trainer=build_trainer(settings),
            shape_spec=shape_summary,
            meta={"dataset_type": "flow_matching"},
        )
