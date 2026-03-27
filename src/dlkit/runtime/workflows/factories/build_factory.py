"""Build factory for constructing runtime components (model, datamodule, trainer).

Phase 1 implementation: flexible dataset support with shape inference.
This replaces the legacy build_model_state() flow with a SOLID factory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from loguru import logger

from dlkit.core.datatypes.split import IndexSplit
from dlkit.core.models.wrappers.factories import WrapperFactory
from dlkit.core.shape_specs.simple_inference import (
    ShapeSummary,
    infer_post_transform_shapes,
    infer_shapes_from_dataset,
)
from dlkit.runtime.workflows.entry_registry import DataEntryRegistry
from dlkit.runtime.workflows.selectors.defaults import FamilyDefaults
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.components.model_components import WrapperComponentSettings
from dlkit.tools.config.core.context import BuildContext
from dlkit.tools.config.core.factories import FactoryProvider
from dlkit.tools.config.data_entries import (
    DataEntry,
    Feature,
    FeatureType,
    Target,
    TargetType,
    convert_to_tensor_entries,
)
from dlkit.tools.config.workflow_configs import (
    InferenceWorkflowConfig,
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)
from dlkit.tools.io.paths import coerce_root_dir_to_absolute
from dlkit.tools.io.split_provider import get_or_create_split

from .feature_dependencies import (
    collect_feature_dependencies,
    select_required_features,
    validate_feature_selection,
)
from .model_detection import detect_model_type, requires_shape_spec

# Type alias for settings that can be passed to build strategies
WorkflowSettings = GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class BuildComponents:
    """Runtime components constructed from settings."""

    model: LightningModule
    datamodule: LightningDataModule
    trainer: Trainer | None
    shape_spec: ShapeSummary | None
    meta: dict[str, Any]


class IBuildStrategy:
    """Interface for build strategies (dataset/wrapper families).

    Implements Template Method pattern: shared build() sets up precision and
    path contexts, then delegates to _build_core() in each subclass.
    """

    def can_handle(self, settings: WorkflowSettings) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def build(self, settings: WorkflowSettings) -> BuildComponents:
        """Template method: set up precision + path contexts, then delegate.

        Shared across all build strategies — eliminates triplication.

        Args:
            settings: Workflow configuration settings.

        Returns:
            Constructed runtime components.
        """
        import contextlib

        from dlkit.tools.config.precision.context import precision_override
        from dlkit.tools.io.path_context import (
            get_current_path_context,
            path_override_context,
        )

        precision_strategy = settings.SESSION.get_precision_strategy()
        ctx = get_current_path_context()
        raw_root = settings.SESSION.root_dir
        session_root_dir = coerce_root_dir_to_absolute(raw_root)
        needs_path_context = (not ctx or not ctx.root_dir) and session_root_dir

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

    def _build_core(
        self, settings: WorkflowSettings
    ) -> BuildComponents:  # pragma: no cover - interface
        raise NotImplementedError


class FlexibleBuildStrategy(IBuildStrategy):
    """Default build strategy for flexible array-like datasets."""

    def can_handle(self, settings: WorkflowSettings) -> bool:
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

    def _build_core(self, settings: WorkflowSettings) -> BuildComponents:
        """Build flexible array dataset components.

        Args:
            settings: Workflow configuration settings.

        Returns:
            Constructed runtime components.
        """
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
        configured_features: tuple[DataEntry, ...] = tuple(
            getattr(ds_settings, "features", ()) or ()
        )
        configured_targets: tuple[DataEntry, ...] = tuple(getattr(ds_settings, "targets", ()) or ())
        feature_dependencies = collect_feature_dependencies(
            configured_features,
            configured_targets,
            getattr(settings.TRAINING, "loss_function", None),
            tuple(getattr(settings.TRAINING, "metrics", ()) or ()),
        )
        selected_features = select_required_features(configured_features, feature_dependencies)
        validate_feature_selection(selected_features, feature_dependencies)
        selected_targets = tuple(configured_targets)

        selected_feature_names = {
            entry.name for entry in selected_features if isinstance(entry.name, str) and entry.name
        }
        dropped_feature_names = [
            entry.name
            for entry in configured_features
            if isinstance(entry.name, str)
            and entry.name
            and entry.name not in selected_feature_names
        ]
        dependency_reasons = {
            name: sorted(reasons)
            for name, reasons in feature_dependencies.items()
            if name in selected_feature_names
        }
        logger.debug(
            "Flexible feature selection: selected={} dropped={} reasons={}",
            sorted(selected_feature_names),
            sorted(dropped_feature_names),
            dependency_reasons,
        )
        dataset = None
        if ds_settings is not None:
            ds_name = str(getattr(ds_settings, "name", "")).lower()
            # Translate legacy x/y -> flexible entries when helpful
            legacy_x = getattr(ds_settings, "x", None)
            legacy_y = getattr(ds_settings, "y", None)
            has_flexible_entries = bool(selected_features) or bool(selected_targets)
            if "supervisedarraydataset" in ds_name:
                from dlkit.core.datasets.flexible import FlexibleDataset

                if (legacy_x is not None) and not has_flexible_entries:
                    # Mirror x for targets if y is None to preserve old behavior
                    x_path = legacy_x
                    y_path = legacy_y or legacy_x
                    # Build FlexibleDataset - precision handled by context
                    dataset = FlexibleDataset(
                        features=[Feature(name="x", path=x_path)],
                        targets=[Target(name="y", path=y_path)],
                        memmap_cache_dir=getattr(ds_settings, "resolved_memmap_cache_dir", None),
                    )
                else:
                    # Build from flexible entries - precision handled by context
                    dataset = FlexibleDataset(
                        features=cast(tuple[FeatureType, ...], selected_features),
                        targets=cast(tuple[TargetType, ...], selected_targets),
                        memmap_cache_dir=getattr(ds_settings, "resolved_memmap_cache_dir", None),
                    )
        # Default factory construction
        if dataset is None:
            if ds_settings is None:
                raise ValueError("DATASET settings are required but not configured")
            ds_overrides = {
                "features": selected_features,
                "targets": selected_targets,
            }
            dataset = FactoryProvider.create_component(
                ds_settings, context.with_overrides(**ds_overrides)
            )

        # Get or create split (with caching)
        assert settings.DATASET is not None, "DATASET settings are required"
        split_cfg = settings.DATASET.split
        index_split: IndexSplit = get_or_create_split(
            num_samples=len(dataset),
            test_ratio=split_cfg.test_ratio,
            val_ratio=split_cfg.val_ratio,
            session_name=settings.SESSION.name,
            explicit_filepath=split_cfg.filepath,
        )

        # Build datamodule with overrides (keep original settings object for test patching)
        assert settings.DATAMODULE is not None, "DATAMODULE settings are required"
        dm_context = context.with_overrides(
            dataset=dataset, split=index_split, dataloader=settings.DATAMODULE.dataloader
        )
        datamodule: LightningDataModule = FactoryProvider.create_component(
            settings.DATAMODULE, dm_context
        )

        # Build entry_configs (features + targets) as tuple for wrapper, dict for registry
        entry_configs: tuple[DataEntry, ...] = ()
        try:
            if selected_features or selected_targets:
                entry_configs = tuple([*selected_features, *selected_targets])
        except Exception:
            entry_configs = ()

        # Register entry configs for user access (registry still uses dict)
        if entry_configs:
            registry = DataEntryRegistry.get_instance()
            registry.register_entries({e.name: e for e in entry_configs if e.name is not None})

        # Detect model type using ABC-based detection
        assert settings.MODEL is not None, "MODEL settings are required"
        model_type = detect_model_type(settings.MODEL)

        # Get appropriate shape specification if required
        shape_summary: ShapeSummary | None = None
        if requires_shape_spec(model_type):
            try:
                shape_summary = infer_post_transform_shapes(
                    dataset, selected_features, selected_targets
                )
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"Shape inference failed for '{settings.MODEL.name}'. "
                    "Ensure dataset.__getitem__ returns a nested TensorDict with "
                    "'features' and 'targets'. If transforms are applied, ensure all "
                    "transforms have registered shape inference functions or specify "
                    "explicit model init_kwargs (e.g., in_features=...)."
                ) from exc

        # Build wrapper settings with training optimizer/scheduler/loss/metrics configuration
        assert settings.TRAINING is not None, "TRAINING settings are required"
        wrapper_kwargs: dict[str, Any] = {
            "optimizer": settings.TRAINING.optimizer,
            "loss_function": settings.TRAINING.loss_function,
            "metrics": settings.TRAINING.metrics,
        }
        if settings.TRAINING.scheduler is not None:
            wrapper_kwargs["scheduler"] = settings.TRAINING.scheduler
        wrapper_settings = WrapperComponentSettings(**wrapper_kwargs)

        # Build pre-instantiated wrapper components (centralize FactoryProvider calls)
        from dlkit.runtime.workflows.factories.component_builders import build_wrapper_components

        components = build_wrapper_components(wrapper_settings, entry_configs)

        # Use family-compatible wrapper (factory kept for test monkeypatch on standard wrapper)
        # family resolved from dataset instance if needed; standard wrapper used for flexible
        model: LightningModule = WrapperFactory.create_standard_wrapper(
            model_settings=settings.MODEL,
            settings=wrapper_settings,
            shape_summary=shape_summary,
            entry_configs=entry_configs,
            components=components,
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
            trainer = trn_cfg.build(session=settings.SESSION)

        return BuildComponents(
            model=model,
            datamodule=datamodule,
            trainer=trainer,
            shape_spec=shape_summary,
            meta={"dataset_type": "flexible"},
        )


# Shape inference now handled by dlkit.runtime.workflows.shape_inference module


class GenerativeBuildStrategy(IBuildStrategy):
    """Abstract base for all generative build strategies.

    Handles settings that have a ``GENERATIVE`` section. Subclasses specialize
    per algorithm (flow matching, CNF, etc.) by overriding ``can_handle``
    and ``_build_core``.
    """

    def can_handle(self, settings: WorkflowSettings) -> bool:
        """Return True when GENERATIVE section is present.

        Args:
            settings: Workflow configuration settings.

        Returns:
            True if GENERATIVE settings are configured.
        """
        try:
            return getattr(settings, "GENERATIVE", None) is not None
        except Exception:
            return False


class FlowMatchingBuildStrategy(GenerativeBuildStrategy):
    """Build strategy for flow matching generative models.

    Constructs:
    - Dataset from ``x1`` entries (model_input=False)
    - ``FlowMatchingSupervisionBuilder`` injected as ``batch_transforms``
    - Model invoker reading ``(xt, t)`` from features
    - ``FlowMatchingWrapper`` with velocity MSE loss
    - ``ODEPredictionStrategy`` for generation

    Args:
        None
    """

    def can_handle(self, settings: WorkflowSettings) -> bool:
        """Return True when algorithm == 'flow_matching'.

        Args:
            settings: Workflow configuration settings.

        Returns:
            True if generative.algorithm is 'flow_matching'.
        """
        if not super().can_handle(settings):
            return False
        try:
            gen = getattr(settings, "GENERATIVE", None)
            return getattr(gen, "algorithm", None) == "flow_matching"
        except Exception:
            return False

    def _build_core(self, settings: WorkflowSettings) -> BuildComponents:
        """Build flow matching components.

        Args:
            settings: Workflow configuration settings with GENERATIVE section.

        Returns:
            Constructed runtime components.
        """
        from pathlib import Path

        from dlkit.core.datasets.flexible import FlexibleDataset
        from dlkit.core.models.nn.generative.functions.solvers import euler_step, heun_step
        from dlkit.core.models.nn.generative.samplers.noise import GaussianNoiseSampler
        from dlkit.core.models.nn.generative.samplers.time import UniformTimeSampler
        from dlkit.core.models.nn.generative.supervision import FlowMatchingSupervisionBuilder
        from dlkit.core.models.wrappers.components import (
            ModelOutputSpec,
            NamedBatchTransformer,
            RoutedLossComputer,
            RoutedMetricsUpdater,
            TensorDictModelInvoker,
            WrapperCheckpointMetadata,
        )
        from dlkit.core.models.wrappers.flowmatching import FlowMatchingWrapper
        from dlkit.core.models.wrappers.generator_factories import DeterministicGeneratorFactory
        from dlkit.core.models.wrappers.prediction_strategies import ODEPredictionStrategy
        from dlkit.core.shape_specs.simple_inference import infer_post_transform_shapes

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
            from dlkit.tools.io.locations import root as _root

            cfg_dir = _root()
        except Exception:
            cfg_dir = Path.cwd()
        context = BuildContext(mode=mode, working_directory=cfg_dir)

        # Build dataset (x1 entries have model_input=False, so they flow through normally)
        ds_settings = settings.DATASET
        configured_features: tuple[DataEntry, ...] = tuple(
            getattr(ds_settings, "features", ()) or ()
        )
        configured_targets: tuple[DataEntry, ...] = tuple(getattr(ds_settings, "targets", ()) or ())
        selected_features = configured_features
        selected_targets = configured_targets

        dataset = FlexibleDataset(
            features=cast(tuple[FeatureType, ...], selected_features),
            targets=cast(tuple[TargetType, ...], selected_targets),
            memmap_cache_dir=getattr(ds_settings, "resolved_memmap_cache_dir", None),
        )

        from dlkit.tools.io.split_provider import get_or_create_split

        assert settings.DATASET is not None, "DATASET settings are required"
        split_cfg = settings.DATASET.split
        index_split: IndexSplit = get_or_create_split(
            num_samples=len(dataset),
            test_ratio=split_cfg.test_ratio,
            val_ratio=split_cfg.val_ratio,
            session_name=settings.SESSION.name,
            explicit_filepath=split_cfg.filepath,
        )

        assert settings.DATAMODULE is not None, "DATAMODULE settings are required"
        dm_context = context.with_overrides(
            dataset=dataset,
            split=index_split,
            dataloader=settings.DATAMODULE.dataloader,
        )
        from lightning.pytorch import LightningDataModule as _LDM

        datamodule: _LDM = FactoryProvider.create_component(settings.DATAMODULE, dm_context)

        # Infer data shape for ODE strategy configuration
        shape_summary = None
        try:
            shape_summary = infer_post_transform_shapes(
                dataset, selected_features, selected_targets
            )
        except Exception:
            pass

        # Build model
        assert settings.MODEL is not None, "MODEL settings are required"
        model_settings = settings.MODEL
        from dlkit.core.models.wrappers.base import _build_model_from_settings

        model = _build_model_from_settings(model_settings, shape_summary)

        # Build supervision builder
        supervision_builder = FlowMatchingSupervisionBuilder(
            x1_key=x1_key,
            time_sampler=UniformTimeSampler(),
            noise_sampler=GaussianNoiseSampler(),
        )

        # Build model invoker for (xt, t) → model(xt, t)
        output_spec = ModelOutputSpec()
        model_invoker = TensorDictModelInvoker(
            in_keys=[("features", "xt"), ("features", "t")],
            output_spec=output_spec,
        )

        # Build empty batch transformer (no per-slot chains needed by default)
        batch_transformer = NamedBatchTransformer({}, {})

        # RoutedLossComputer routes predictions against batch["targets"]["ut"].
        # Loss function is configurable via TRAINING.loss_function (defaults to MSE).
        assert settings.TRAINING is not None, "TRAINING settings are required"
        loss_fn = FactoryProvider.create_component(
            settings.TRAINING.loss_function, BuildContext(mode="training")
        )
        loss_spec = settings.TRAINING.loss_function
        loss_computer = RoutedLossComputer(
            loss_fn=loss_fn,
            target_key=getattr(loss_spec, "target_key", None),
            default_target_key="ut",
            extra_inputs=tuple(getattr(loss_spec, "extra_inputs", ()) or ()),
        )
        metrics_updater = RoutedMetricsUpdater(val_routes=[], test_routes=[])

        # Build ODE prediction strategy
        solver_fn = euler_step if solver_name == "euler" else heun_step
        ode_strategy = ODEPredictionStrategy(
            x0_sampler=GaussianNoiseSampler(),
            solver=solver_fn,
            n_steps=n_inference_steps,
        )

        # Build checkpoint metadata
        assert settings.TRAINING is not None, "TRAINING settings are required"
        entry_configs: tuple[DataEntry, ...] = tuple([*selected_features, *selected_targets])
        from dlkit.tools.config.components.model_components import WrapperComponentSettings as _WCS

        wrapper_settings = _WCS(
            optimizer=settings.TRAINING.optimizer,
            loss_function=settings.TRAINING.loss_function,
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

        # Build generator factories
        val_gen_factory = DeterministicGeneratorFactory(base_seed=val_seed)

        # Build trainer
        trainer = None
        if (
            settings.SESSION
            and not getattr(settings.SESSION, "inference", False)
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
            trainer = trn_cfg.build(session=settings.SESSION)

        # Assemble FlowMatchingWrapper
        assert settings.TRAINING is not None, "TRAINING settings are required"
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
            val_generator_factory=val_gen_factory,
        )

        return BuildComponents(
            model=model_wrapper,
            datamodule=datamodule,
            trainer=trainer,
            shape_spec=shape_summary,
            meta={"dataset_type": "flow_matching"},
        )


class BuildFactory:
    """Factory that selects an appropriate build strategy and constructs components.

    **Pre-Build Validation**: This factory validates settings before building components.
    Settings are eagerly validated when loaded; this step ensures required sections
    are present before constructing expensive runtime components.
    """

    def __init__(self, strategies: list[IBuildStrategy] | None = None) -> None:
        # Order: flow_matching -> graph -> timeseries -> flexible (fallback)
        self._strategies = strategies or [
            FlowMatchingBuildStrategy(),
            GraphBuildStrategy(),
            TimeSeriesBuildStrategy(),
            FlexibleBuildStrategy(),
        ]

    def _validate_settings(self, settings: WorkflowSettings) -> None:
        """Validate settings completeness before building components.

        This uses workflow-specific completeness validators that check:
        - Required sections are present (not None)
        - Required fields within sections are populated
        - File paths exist
        - Cross-section validation (e.g., OPTUNA.enabled for optimization)

        Args:
            settings: Settings to validate

        Raises:
            ConfigValidationError: If settings are incomplete or invalid
        """
        from dlkit.tools.config.validators import validate_config_complete

        # Use the new completeness validators (only for concrete workflow configs)
        if isinstance(
            settings, (TrainingWorkflowConfig, InferenceWorkflowConfig, OptimizationWorkflowConfig)
        ):
            validate_config_complete(settings)

    def build_components(self, settings: WorkflowSettings) -> BuildComponents:
        # TODO: TYPE — settings.DATASET/MODEL/DATAMODULE typed as X | None but validate_config_complete
        # above guarantees non-None; consider TypeGuard or a validated-config type to convey this.
        """Build runtime components with pre-build validation.

        Validates settings completeness first (ensures all required sections present),
        then selects and executes appropriate build strategy.

        Args:
            settings: Workflow config (TrainingWorkflowConfig or OptimizationWorkflowConfig)

        Returns:
            BuildComponents: Validated and constructed runtime components

        Raises:
            ConfigValidationError: If settings are incomplete or invalid
        """
        # Validate settings completeness before building (fail-fast with clear errors)
        self._validate_settings(settings)

        # Build components using appropriate strategy
        for strat in self._strategies:
            if strat.can_handle(settings):
                return strat.build(settings)
        # Fallback to flexible
        return FlexibleBuildStrategy().build(settings)


class GraphBuildStrategy(IBuildStrategy):
    """Build strategy for graph (PyG) datasets and models."""

    def can_handle(self, settings: WorkflowSettings) -> bool:
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

    def _build_core(self, settings: WorkflowSettings) -> BuildComponents:
        """Build graph dataset components.

        Args:
            settings: Workflow configuration settings.

        Returns:
            Constructed runtime components.
        """
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
            resolved_features = convert_to_tensor_entries(
                getattr(ds_settings, "features", ()) or ()
            )
            resolved_targets = convert_to_tensor_entries(getattr(ds_settings, "targets", ()) or ())
            if resolved_features:
                ds_overrides["features"] = resolved_features
            elif fp is not None:
                ds_overrides["features"] = fp
            if resolved_targets:
                ds_overrides["targets"] = resolved_targets
        except Exception:
            pass

        assert settings.DATASET is not None, "DATASET settings are required"
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

        assert settings.DATAMODULE is not None, "DATAMODULE settings are required"
        dm_context = context.with_overrides(
            dataset=dataset, split=index_split, dataloader=settings.DATAMODULE.dataloader
        )
        # Default to graph-aware datamodule when no explicit datamodule is provided
        dm_settings = settings.DATAMODULE
        try:
            if dm_settings is not None:
                name = getattr(dm_settings, "name", None)
                if not name or str(name) == "InMemoryModule":
                    dm_class = FamilyDefaults.default_datamodule_class_for(settings)
                    dm_settings = dm_settings.model_copy(update={"name": dm_class})
        except Exception:
            pass
        datamodule: LightningDataModule = FactoryProvider.create_component(dm_settings, dm_context)

        # Graph strategy doesn't use flexible entry_configs — heuristics handle extraction
        entry_configs: tuple[DataEntry, ...] = ()

        # Build wrapper with training optimizer/scheduler/loss/metrics configuration
        assert settings.TRAINING is not None, "TRAINING settings are required"
        wrapper_kwargs: dict[str, Any] = {
            "optimizer": settings.TRAINING.optimizer,
            "loss_function": settings.TRAINING.loss_function,
            "metrics": settings.TRAINING.metrics,
        }
        if settings.TRAINING.scheduler is not None:
            wrapper_kwargs["scheduler"] = settings.TRAINING.scheduler
        wrapper_settings = WrapperComponentSettings(**wrapper_kwargs)

        # Build pre-instantiated wrapper components (centralize FactoryProvider calls)
        from dlkit.runtime.workflows.factories.component_builders import build_wrapper_components

        components = build_wrapper_components(wrapper_settings, entry_configs)

        assert settings.MODEL is not None, "MODEL settings are required"
        model: LightningModule = WrapperFactory.create_graph_wrapper(
            model_settings=settings.MODEL,
            settings=wrapper_settings,
            entry_configs=entry_configs,
            components=components,
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
            trainer = trn_cfg.build(session=settings.SESSION)

        return BuildComponents(
            model=model,
            datamodule=datamodule,
            trainer=trainer,
            shape_spec=None,
            meta={"dataset_type": "graph"},
        )


class TimeSeriesBuildStrategy(IBuildStrategy):
    """Build strategy for time series / forecasting datasets and models."""

    def can_handle(self, settings: WorkflowSettings) -> bool:
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

    def _build_core(self, settings: WorkflowSettings) -> BuildComponents:
        """Build time-series dataset components.

        Args:
            settings: Workflow configuration settings.

        Returns:
            Constructed runtime components.
        """
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
            resolved_features = convert_to_tensor_entries(
                getattr(ds_settings, "features", ()) or ()
            )
            resolved_targets = convert_to_tensor_entries(getattr(ds_settings, "targets", ()) or ())
            if resolved_features:
                ds_overrides["features"] = resolved_features
            elif fp is not None:
                ds_overrides["features"] = fp
            if resolved_targets:
                ds_overrides["targets"] = resolved_targets
        except Exception:
            pass
        assert settings.DATASET is not None, "DATASET settings are required"
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

        assert settings.DATAMODULE is not None, "DATAMODULE settings are required"
        dm_context = context.with_overrides(
            dataset=dataset, split=index_split, dataloader=settings.DATAMODULE.dataloader
        )
        # Default to time-series-aware datamodule when no explicit datamodule is provided
        dm_settings = settings.DATAMODULE
        try:
            if dm_settings is not None:
                name = getattr(dm_settings, "name", None)
                if not name or str(name) == "InMemoryModule":
                    dm_class = FamilyDefaults.default_datamodule_class_for(settings)
                    dm_settings = dm_settings.model_copy(update={"name": dm_class})
        except Exception:
            pass
        datamodule: LightningDataModule = FactoryProvider.create_component(dm_settings, dm_context)

        # Timeseries strategy doesn't use flexible entry configs by default
        entry_configs: tuple[DataEntry, ...] = ()

        # Detect model type using ABC-based detection
        assert settings.MODEL is not None, "MODEL settings are required"
        model_type = detect_model_type(settings.MODEL)

        # Get appropriate shape specification if required
        shape_summary: ShapeSummary | None = None
        if requires_shape_spec(model_type):
            try:
                shape_summary = infer_shapes_from_dataset(dataset)
            except ValueError, IndexError:
                pass  # Timeseries models may not require strict shape inference

        # Decide whether to skip wrapper for PF-style datasets/models
        skip_wrapper = False
        try:
            # Skip when dataset is our PF wrapper
            from dlkit.core.datasets.timeseries import ForecastingDataset as _FDS

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

                fallback_mod: str = settings.MODEL.module_path or ""
                model_cls = _import(model_ref, fallback_module=fallback_mod)
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
            assert settings.TRAINING is not None, "TRAINING settings are required"
            wrapper_kwargs: dict[str, Any] = {
                "optimizer": settings.TRAINING.optimizer,
                "loss_function": settings.TRAINING.loss_function,
                "metrics": settings.TRAINING.metrics,
            }
            if settings.TRAINING.scheduler is not None:
                wrapper_kwargs["scheduler"] = settings.TRAINING.scheduler
            wrapper_settings = WrapperComponentSettings(**wrapper_kwargs)

            # Build pre-instantiated wrapper components (centralize FactoryProvider calls)
            from dlkit.runtime.workflows.factories.component_builders import (
                build_wrapper_components,
            )

            components = build_wrapper_components(wrapper_settings, entry_configs)

            model = WrapperFactory.create_timeseries_wrapper(
                model_settings=settings.MODEL,
                settings=wrapper_settings,
                shape_summary=shape_summary,
                entry_configs=entry_configs,
                components=components,
            )

        trainer: Trainer | None = None
        if (
            settings.SESSION
            and (not getattr(settings.SESSION, "inference", False))
            and settings.TRAINING
        ):
            trainer = settings.TRAINING.trainer.build(session=settings.SESSION)

        return BuildComponents(
            model=model,
            datamodule=datamodule,
            trainer=trainer,
            shape_spec=shape_summary,
            meta={"dataset_type": "timeseries"},
        )


# Graph shape inference now handled by dlkit.runtime.workflows.shape_inference module


# Timeseries shape inference now handled by dlkit.runtime.workflows.shape_inference module
