"""Standard Lightning wrapper for tensor-based models.

This module provides a Lightning wrapper for standard neural networks that
work with tensor inputs, integrating with the dlkit processing pipeline.
"""

from typing import Any

import torch
from torch import Tensor
from torch.nn import ModuleDict
from loguru import logger

from dlkit.tools.config import ModelComponentSettings, WrapperComponentSettings
from dlkit.tools.config.data_entries import DataEntry, Target, is_target_entry
from dlkit.core.shape_specs import IShapeSpec
from dlkit.core.training.transforms.chain import TransformChain
from dlkit.core.training.transforms.base import InvertibleTransform
from dlkit.runtime.pipelines.pipeline import (
    ProcessingPipeline,
    DataExtractionStep,
    TransformApplicationStep,
    ModelInvocationStep,
    OutputClassificationStep,
    OutputNamingStep,
    ValidationDataStep,
)
from dlkit.runtime.pipelines.model_invokers import ModelInvokerFactory
from dlkit.runtime.pipelines.classifiers import NameBasedClassifier
from dlkit.runtime.pipelines.context import ProcessingContext
from .base import ProcessingLightningWrapper


class StandardLightningWrapper(ProcessingLightningWrapper):
    """Lightning wrapper for standard tensor-based neural networks with transforms.

    This wrapper handles models that accept tensor inputs and provides
    standard training, validation, and test steps using the processing pipeline
    with full dlkit transform support.

    Transforms are separated by data entry type (Feature vs Target) to maintain
    clear separation of concerns and eliminate runtime filtering.

    Attributes:
        fitted_feature_transforms (torch.nn.ModuleDict): Persisted feature transform chains.
        fitted_target_transforms (torch.nn.ModuleDict): Persisted target transform chains.
        apply_feature_transforms (bool): Toggle for feature transform application during inference.
        apply_inverse_target_transforms (bool): Toggle for inverse target transform application.

    Example:
        ```python
        wrapper = StandardLightningWrapper(
            settings=wrapper_settings,
            model_settings=model_settings,
            shape_spec=shape_spec,
            entry_configs=data_configs,
        )
        ```
    """

    def __init__(
        self,
        *,
        settings: WrapperComponentSettings,
        model_settings: ModelComponentSettings,
        entry_configs: dict[str, DataEntry] | None = None,
        shape_spec: IShapeSpec | None = None,
        **kwargs,
    ):
        """Initialize the standard Lightning wrapper.

        Args:
            settings: Wrapper configuration settings
            model_settings: Model configuration settings
            entry_configs: Data entry configurations for pipeline setup
            shape_spec: Shape specification for models
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(
            settings=settings,
            model_settings=model_settings,
            entry_configs=entry_configs,
            shape_spec=shape_spec,
            **kwargs,
        )

        # Initialize transform-specific attributes after super().__init__()
        # Separate Feature and Target transforms for clear separation of concerns
        self.fitted_feature_transforms: ModuleDict = ModuleDict()
        self.fitted_target_transforms: ModuleDict = ModuleDict()
        self.apply_feature_transforms: bool = True
        self.apply_inverse_target_transforms: bool = True

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the model with tensor input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Model output tensor.
        """
        return self.model(x)

    def _setup_processing_pipelines(self, entry_configs: dict[str, DataEntry]) -> None:
        """Set up processing pipelines for training/validation/test with transforms.

        Args:
            entry_configs (dict[str, DataEntry]): Mapping from entry name to configuration.
        """
        # Create model invoker
        model_invoker = ModelInvokerFactory.create_invoker(self.model, model_type="auto")

        # Create output classifier (can be overridden by subclasses)
        output_classifier = self._create_output_classifier()
        # Create output namer (post-classification key mapping)
        output_namer = self._create_output_namer()

        # Shared transform cache across phases (entry name -> TransformChain)
        self._shared_transform_cache = {}
        shared_transform_cache = self._shared_transform_cache

        # Build training steps
        self._extraction_step_train = DataExtractionStep(entry_configs)
        self._transform_step_train = TransformApplicationStep(
            entry_configs, cache=shared_transform_cache, fit_during_apply=True, wrapper=self
        )
        from dlkit.runtime.pipelines.pipeline import LossPairingStep

        invocation_train = ModelInvocationStep(
            model_invoker,
            OutputClassificationStep(
                output_classifier,
                OutputNamingStep(
                    output_namer,
                    LossPairingStep(
                        entry_configs,
                        is_autoencoder=getattr(self._wrapper_settings, "is_autoencoder", False),
                    ),
                ),
            ),
        )
        self._extraction_step_train.set_next(self._transform_step_train)
        self._transform_step_train.set_next(invocation_train)
        self.train_pipeline = ProcessingPipeline(self._extraction_step_train)

        # Build validation steps (share cache; disable fitting later in on_fit_start if needed)
        self._extraction_step_val = DataExtractionStep(entry_configs)
        self._transform_step_val = TransformApplicationStep(
            entry_configs, cache=shared_transform_cache, fit_during_apply=True, wrapper=self
        )
        invocation_val = ModelInvocationStep(
            model_invoker,
            OutputClassificationStep(
                output_classifier,
                OutputNamingStep(
                    output_namer,
                    LossPairingStep(
                        entry_configs,
                        is_autoencoder=getattr(self._wrapper_settings, "is_autoencoder", False),
                        next_step=ValidationDataStep(),
                    ),
                ),
            ),
        )
        self._extraction_step_val.set_next(self._transform_step_val)
        self._transform_step_val.set_next(invocation_val)
        self.val_pipeline = ProcessingPipeline(self._extraction_step_val)

        # Test pipeline mirrors validation
        self._extraction_step_test = DataExtractionStep(entry_configs)
        self._transform_step_test = TransformApplicationStep(
            entry_configs, cache=shared_transform_cache, fit_during_apply=True, wrapper=self
        )
        invocation_test = ModelInvocationStep(
            model_invoker,
            OutputClassificationStep(
                output_classifier,
                OutputNamingStep(
                    output_namer,
                    LossPairingStep(
                        entry_configs,
                        is_autoencoder=getattr(self._wrapper_settings, "is_autoencoder", False),
                        next_step=ValidationDataStep(),
                    ),
                ),
            ),
        )
        self._extraction_step_test.set_next(self._transform_step_test)
        self._transform_step_test.set_next(invocation_test)
        self.test_pipeline = ProcessingPipeline(self._extraction_step_test)

    def _setup_predict_pipeline(self, entry_configs: dict[str, DataEntry]) -> None:
        """Set up inference-only processing pipeline with transforms.

        This pipeline includes TransformApplicationStep but excludes LossPairingStep
        to make targets optional during inference.

        Args:
            entry_configs (dict[str, DataEntry]): Mapping from entry name to configuration.
        """
        # Create model invoker (reuse same factory as other pipelines)
        model_invoker = ModelInvokerFactory.create_invoker(self.model, model_type="auto")

        # Create output classifier and namer (consistent with other pipelines)
        output_classifier = self._create_output_classifier()
        output_namer = self._create_output_namer()

        # Build predict steps
        self._extraction_step_predict = DataExtractionStep(entry_configs)
        self._transform_step_predict = TransformApplicationStep(
            entry_configs, cache=self._shared_transform_cache, fit_during_apply=False, wrapper=self
        )

        # Chain: extraction → transform → invocation
        next_step = ModelInvocationStep(
            model_invoker,
            OutputClassificationStep(
                output_classifier,
                OutputNamingStep(output_namer)  # Terminates here - no loss pairing
            ),
        )
        self._extraction_step_predict.set_next(self._transform_step_predict)
        self._transform_step_predict.set_next(next_step)

        self.predict_pipeline = ProcessingPipeline(self._extraction_step_predict)

    def on_fit_start(self) -> None:
        """Fit all entry-based transforms using the entire training dataloader.

        - Aggregates full training dataflow per entry that has transforms and fits once with
          the stacked tensor (supports global transforms like PCA).
        - Populates the shared transform cache so train/val/test pipelines reuse the same
          fitted transform instances.
        - Disables any further in-apply fitting for all phases.
        """
        trainer = getattr(self, "trainer", None)
        if trainer is None or not hasattr(trainer, "datamodule"):
            return
        dm = trainer.datamodule
        if dm is None or not hasattr(dm, "train_dataloader"):
            return

        # Determine which entries have transforms configured
        entry_cfgs = getattr(self, "_entry_configs", {})  # type: ignore[attr-defined]
        names_with_transforms = [n for n, e in entry_cfgs.items() if getattr(e, "transforms", None)]
        if not names_with_transforms:
            # Nothing to fit globally
            return

        try:
            loader = dm.train_dataloader()
        except Exception:
            return

        # Aggregate tensors per entry name across the whole training set
        from collections import defaultdict
        import torch

        buffers: dict[str, list[torch.Tensor]] = defaultdict(list)
        for batch in loader:
            # Expect dict[str, Tensor]
            for name in names_with_transforms:
                tensor = batch.get(name)
                if tensor is not None:
                    buffers[name].append(tensor)

        # Build and fit transforms globally per entry

        for name in names_with_transforms:
            seq = buffers.get(name)
            if not seq:
                continue
            stacked = torch.cat(seq, dim=0)
            cfg = entry_cfgs[name]

            # Build a single chain per entry and fit globally
            # Shape will be inferred from stacked data during fit()
            chain = TransformChain(getattr(cfg, "transforms", ()))
            if hasattr(chain, "fit"):
                chain.fit(stacked)

            # Populate shared transform cache across phases
            self._transform_step_train.cache[name] = chain  # type: ignore[attr-defined]
            self._transform_step_val.cache[name] = chain  # type: ignore[attr-defined]
            self._transform_step_test.cache[name] = chain  # type: ignore[attr-defined]

            # Persist to appropriate module dict based on data entry type
            from dlkit.tools.config.data_entries import is_feature_entry, is_target_entry
            if is_target_entry(cfg):
                self.fitted_target_transforms[name] = chain
            elif is_feature_entry(cfg):
                self.fitted_feature_transforms[name] = chain

        # Disable any further fitting during application for all phases
        self._transform_step_train.set_fit_enabled(False)  # type: ignore[attr-defined]
        self._transform_step_val.set_fit_enabled(False)  # type: ignore[attr-defined]
        self._transform_step_test.set_fit_enabled(False)  # type: ignore[attr-defined]

    # --- Public helpers for manual transform application ---
    def feature_transforms(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply configured feature transforms to a features dict.

        Uses available transform chains from runtime caches (train/val/test) or from
        the persisted ``fitted_feature_transforms`` ModuleDict when caches are empty.
        Entries without a chain are returned unchanged.
        """
        out: dict[str, Tensor] = {}
        cache = None
        try:
            cache = getattr(self, "_transform_step_val").cache  # type: ignore[attr-defined]
        except Exception:
            cache = {}
        for name, tensor in features.items():
            chain = None
            try:
                chain = cache.get(name)
            except Exception:
                chain = None
            if chain is None and isinstance(self.fitted_feature_transforms, ModuleDict):
                fitted_chain = self.fitted_feature_transforms.get(name)  # type: ignore[misc]
                if fitted_chain is not None and callable(fitted_chain):
                    chain = fitted_chain
            if chain is not None and callable(chain):
                out[name] = chain(tensor)  # type: ignore[misc]
            else:
                out[name] = tensor
        return out

    def target_transforms_inverse(self, tensors: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply inverse transforms for targets/predictions by entry name.

        Looks up transform chains in caches or in ``fitted_target_transforms`` and applies
        ``inverse_transform`` when available using InvertibleTransform Protocol check.
        Entries without an invertible chain are returned unchanged.
        """
        out: dict[str, Tensor] = {}
        cache = None
        try:
            cache = getattr(self, "_transform_step_val").cache  # type: ignore[attr-defined]
        except Exception:
            cache = {}
        for name, tensor in tensors.items():
            chain = None
            try:
                chain = cache.get(name)
            except Exception:
                chain = None
            if chain is None and isinstance(self.fitted_target_transforms, ModuleDict):
                fitted_chain = self.fitted_target_transforms.get(name)  # type: ignore[misc]
                if fitted_chain is not None and isinstance(fitted_chain, InvertibleTransform):
                    chain = fitted_chain
            if chain is not None and isinstance(chain, InvertibleTransform):
                try:
                    out[name] = chain.inverse_transform(tensor)
                except Exception:
                    out[name] = tensor
            else:
                out[name] = tensor
        return out

    def _restore_fitted_transforms_from_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Manually restore fitted transforms ModuleDicts from checkpoint.

        Lightning's load_state_dict with strict=False skips transform keys
        when ModuleDicts are empty. This method reconstructs chains from
        checkpoint state_dict using the original transform configurations.

        Supports both legacy (fitted_transforms) and modern (fitted_feature_transforms/
        fitted_target_transforms) checkpoint formats.
        """
        from dlkit.core.training.transforms.chain import TransformChain
        from dlkit.tools.config.data_entries import is_feature_entry, is_target_entry

        logger.info("Starting manual restoration of fitted transforms from checkpoint")

        try:
            state_dict = checkpoint.get("state_dict", {})
            inference_metadata = checkpoint.get("inference_metadata", {})

            # Check for modern format (separate feature/target transforms)
            has_modern_format = any(k.startswith("fitted_feature_transforms.") or k.startswith("fitted_target_transforms.") for k in state_dict.keys())
            has_legacy_format = any(k.startswith("fitted_transforms.") for k in state_dict.keys())

            logger.info(f"Checkpoint format: modern={has_modern_format}, legacy={has_legacy_format}")

            # Get entry configs from metadata (contains original transform settings)
            entry_configs_data = inference_metadata.get("entry_configs", {})

            # Helper function to process transform keys for a given prefix and target ModuleDict
            def restore_transforms(prefix: str, target_module_dict: ModuleDict, entry_filter_func=None):
                """Restore transforms with a specific prefix into target ModuleDict."""
                transform_keys = [k for k in state_dict.keys() if k.startswith(f"{prefix}.")]

                if not transform_keys:
                    logger.debug(f"No {prefix} found in checkpoint")
                    return

                # Group keys by entry name
                entry_states: dict[str, dict[str, Any]] = {}
                for key in transform_keys:
                    parts = key.split(".", 2)  # ['prefix', 'entry_name', 'rest...']
                    if len(parts) >= 2:
                        entry_name = parts[1]
                        state_key = parts[2] if len(parts) > 2 else ""

                        # Apply filter if provided
                        entry_config = entry_configs_data.get(entry_name)
                        if entry_filter_func and not entry_filter_func(entry_config):
                            continue

                        if entry_name not in entry_states:
                            entry_states[entry_name] = {}

                        if state_key:
                            entry_states[entry_name][state_key] = state_dict[key]

                # Reconstruct each TransformChain
                for entry_name, entry_state in entry_states.items():
                    try:
                        entry_config = entry_configs_data.get(entry_name)
                        transform_settings = []
                        if entry_config and hasattr(entry_config, 'transforms'):
                            transform_settings = entry_config.transforms
                        elif entry_config and isinstance(entry_config, dict):
                            transform_settings = entry_config.get('transforms', [])

                        logger.info(f"Reconstructing {prefix} chain for '{entry_name}' with {len(transform_settings)} transforms")

                        # Create chain with original settings, then load fitted state
                        chain = TransformChain(transform_settings)
                        result = chain.load_state_dict(entry_state, strict=False)

                        # Manually register any unexpected keys as buffers
                        if result.unexpected_keys:
                            for unexpected_key in result.unexpected_keys:
                                try:
                                    parts = unexpected_key.split('.')
                                    module = chain
                                    for part in parts[:-1]:
                                        if part.isdigit():
                                            module = module[int(part)]
                                        else:
                                            module = getattr(module, part)
                                    buffer_name = parts[-1]
                                    buffer_value = entry_state[unexpected_key]
                                    module.register_buffer(buffer_name, buffer_value)
                                    logger.debug(f"Manually registered buffer: {unexpected_key}")
                                except Exception as e:
                                    logger.warning(f"Could not register unexpected key {unexpected_key}: {e}")

                        target_module_dict[entry_name] = chain
                        logger.info(f"Successfully restored {prefix} chain for '{entry_name}'")
                    except Exception as e:
                        logger.error(f"Failed to restore {prefix} chain for '{entry_name}': {e}")

            # Restore based on checkpoint format
            if has_modern_format:
                restore_transforms("fitted_feature_transforms", self.fitted_feature_transforms,
                                 entry_filter_func=is_feature_entry)
                restore_transforms("fitted_target_transforms", self.fitted_target_transforms,
                                 entry_filter_func=is_target_entry)
            elif has_legacy_format:
                # Legacy format - separate based on entry_configs
                logger.info("Loading legacy format, separating transforms by entry type")
                restore_transforms("fitted_transforms", self.fitted_feature_transforms,
                                 entry_filter_func=is_feature_entry)
                restore_transforms("fitted_transforms", self.fitted_target_transforms,
                                 entry_filter_func=is_target_entry)
            else:
                logger.debug("No fitted transforms found in checkpoint")

        except Exception as e:
            logger.warning(f"Failed to restore fitted_transforms from checkpoint: {e}")

    def _hydrate_transforms_from_module_dict(self) -> None:
        """Populate transform caches from persisted ModuleDicts and disable fitting.

        Enables inference-only runs with just a predict_dataloader by reusing transform
        chains loaded from checkpoint. Hydrates both feature and target transforms.
        """
        try:
            has_feature_transforms = (
                isinstance(self.fitted_feature_transforms, ModuleDict)
                and len(self.fitted_feature_transforms) > 0
            )
            has_target_transforms = (
                isinstance(self.fitted_target_transforms, ModuleDict)
                and len(self.fitted_target_transforms) > 0
            )

            if not has_feature_transforms and not has_target_transforms:
                return
        except Exception:
            return

        # Helper to copy transforms into caches
        def hydrate_cache(transforms_dict: ModuleDict):
            for name, chain in transforms_dict.items():
                if hasattr(self, '_transform_step_train'):
                    self._transform_step_train.cache[name] = chain  # type: ignore[attr-defined]
                if hasattr(self, '_transform_step_val'):
                    self._transform_step_val.cache[name] = chain  # type: ignore[attr-defined]
                if hasattr(self, '_transform_step_test'):
                    self._transform_step_test.cache[name] = chain  # type: ignore[attr-defined]
                # Include predict pipeline in transform chain hydration
                if hasattr(self, '_transform_step_predict'):
                    self._transform_step_predict.cache[name] = chain  # type: ignore[attr-defined]

        # Hydrate both feature and target transforms
        hydrate_cache(self.fitted_feature_transforms)
        hydrate_cache(self.fitted_target_transforms)

        # Disable fitting during application for all phases
        if hasattr(self, '_transform_step_train'):
            self._transform_step_train.set_fit_enabled(False)  # type: ignore[attr-defined]
        if hasattr(self, '_transform_step_val'):
            self._transform_step_val.set_fit_enabled(False)  # type: ignore[attr-defined]
        if hasattr(self, '_transform_step_test'):
            self._transform_step_test.set_fit_enabled(False)  # type: ignore[attr-defined]
        # Predict pipeline already has fit disabled by design

    def _persist_transform_caches(self) -> None:
        """Persist cached transform chains into ModuleDicts for checkpointing.

        Collects chains from train/val/test caches and registers them in the
        appropriate ModuleDict (feature vs target) based on entry_configs.
        """
        from dlkit.tools.config.data_entries import Feature, Target

        # Guard: No entry configs, can't separate
        if not self._entry_configs:
            return

        # Collect all caches
        caches = self._collect_transform_caches()
        if not caches:
            return

        # Persist each chain to appropriate ModuleDict
        for cache in caches:
            for name, chain in cache.items():
                self._persist_single_chain(name, chain)

    def _collect_transform_caches(self) -> list[dict]:
        """Collect transform caches from all pipeline steps."""
        caches = []
        for attr in ("_transform_step_train", "_transform_step_val", "_transform_step_test", "_transform_step_predict"):
            step = getattr(self, attr, None)
            if step is not None and hasattr(step, "cache"):
                caches.append(step.cache)
        return caches

    def _persist_single_chain(self, name: str, chain: Any) -> None:
        """Persist a single transform chain to appropriate ModuleDict."""
        from dlkit.tools.config.data_entries import is_feature_entry, is_target_entry

        try:
            entry_config = self._entry_configs.get(name)
            if not entry_config:
                return

            # Route to appropriate ModuleDict based on type
            if is_target_entry(entry_config):
                if name not in self.fitted_target_transforms:
                    self.fitted_target_transforms[name] = chain
            elif is_feature_entry(entry_config):
                if name not in self.fitted_feature_transforms:
                    self.fitted_feature_transforms[name] = chain
        except Exception:
            # Be resilient; this is best-effort registration
            pass

    def on_predict_start(self) -> None:
        """Hydrate caches when loading from checkpoint for inference.

        Ensures transform chains are available before prediction starts.
        """
        # Persist any runtime-built chains first, then hydrate
        self._persist_transform_caches()
        self._hydrate_transforms_from_module_dict()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Prepare checkpoint for saving with inference metadata.

        Ensures transform chains and inference configuration are persisted for
        standalone inference.
        """
        # Persist any cached transforms to module dict before saving
        self._persist_transform_caches()
        # The fitted_transforms ModuleDict will be automatically saved by Lightning

        # Call parent to save enhanced dlkit_metadata
        super().on_save_checkpoint(checkpoint)

        # Save inference metadata for inference
        feature_names = [name for name, cfg in self._entry_configs.items() if not is_target_entry(cfg)]
        target_names = [name for name, cfg in self._entry_configs.items() if is_target_entry(cfg)]
        feature_transform_names = list(self.fitted_feature_transforms.keys()) if self.fitted_feature_transforms else []
        target_transform_names = list(self.fitted_target_transforms.keys()) if self.fitted_target_transforms else []

        checkpoint["inference_metadata"] = {
            "entry_configs": self._entry_configs,
            "wrapper_settings": self._wrapper_settings.model_dump() if hasattr(self._wrapper_settings, 'model_dump') else dict(self._wrapper_settings),
            "feature_names": feature_names,
            "target_names": target_names,
            "feature_transform_names": feature_transform_names,
            "target_transform_names": target_transform_names,
            "model_shape": getattr(self, 'shape', None),
        }

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True, assign: bool = False) -> Any:
        """Override load_state_dict to ensure fitted_transforms are restored.

        Handles both Lightning checkpoint loading and direct torch.load scenarios.
        """
        # First, let PyTorch/Lightning load what it can
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)

        # Then manually restore fitted_transforms (which may have been skipped)
        # Create a minimal checkpoint dict for restoration
        # Use self._entry_configs if available (wrapper already has this from __init__)
        checkpoint = {
            "state_dict": state_dict,
            "inference_metadata": {
                "entry_configs": self._entry_configs if hasattr(self, '_entry_configs') else {}
            }
        }
        self._restore_fitted_transforms_from_checkpoint(checkpoint)

        # Hydrate caches from the restored transforms
        self._hydrate_transforms_from_module_dict()

        return result

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Handle checkpoint loading.

        Ensures transform chains are properly loaded and caches are hydrated.
        Also validates checkpoint version via base class.
        """
        # Call base class to validate version and restore metadata
        super().on_load_checkpoint(checkpoint)

        # Manually restore fitted_transforms from checkpoint
        # Lightning's load_state_dict with strict=False skips these keys if ModuleDict is empty
        self._restore_fitted_transforms_from_checkpoint(checkpoint)

        # Then hydrate caches from it
        self._hydrate_transforms_from_module_dict()

    def predict_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, dict[str, Tensor]]:
        """Prediction step with transform support.

        Uses the predict_pipeline which excludes LossPairingStep, making targets
        optional during inference. Applies inverse transforms to predictions and
        targets (if present) so the user receives values in the original dataflow space.

        Args:
            batch (dict[str, Tensor]): Raw batch dataflow from the dataset.
            batch_idx (int): Index of the batch.

        Returns:
            dict[str, dict[str, Tensor]]: Dictionary with ``predictions``, ``targets``, and ``latents``.
        """
        # Respect runtime toggles for transform application
        try:
            if hasattr(self, "_transform_step_predict") and hasattr(
                self._transform_step_predict, "set_enabled"
            ):
                self._transform_step_predict.set_enabled(bool(self.apply_feature_transforms))  # type: ignore[attr-defined]
        except Exception:
            pass

        # Use dedicated predict pipeline - no loss pairing required
        context = self.predict_pipeline.execute(batch)

        # Apply inverse transforms to predictions and optional targets
        inv_predictions = self._apply_inverse_transforms_to_predictions(context)
        inv_targets = self._apply_inverse_transforms_to_targets(context)

        return {"predictions": inv_predictions, "targets": inv_targets, "latents": context.latents}

    def _apply_inverse_transforms_to_predictions(self, context) -> dict[str, Tensor]:
        """Apply inverse transforms to predictions using target transform chains.

        Args:
            context: Processing context with predictions populated.

        Returns:
            dict[str, Tensor]: Inverse-transformed predictions.
        """
        if not self.apply_inverse_target_transforms:
            return dict(context.predictions)

        inv_predictions: dict[str, Tensor] = {}

        # Resolve target names from entry configs
        target_names = {
            name
            for name, cfg in self._entry_configs.items()  # type: ignore[attr-defined]
            if is_target_entry(cfg)
        }

        # Use cached transform chains from predict transform step
        cache = getattr(self, "_transform_step_predict").cache  # type: ignore[attr-defined]

        # Map predictions to targets: exact name match or single-target fallback
        single_target = next(iter(target_names)) if len(target_names) == 1 else None

        for pname, ptensor in context.predictions.items():
            target_for_pred = pname if pname in target_names else single_target
            chain = cache.get(target_for_pred) if target_for_pred is not None else None

            if chain:
                try:
                    inv_predictions[pname] = self._inverse_with_chain(ptensor, chain)
                except Exception:
                    inv_predictions[pname] = ptensor
            else:
                inv_predictions[pname] = ptensor

        return inv_predictions

    def _apply_inverse_transforms_to_targets(self, context) -> dict[str, Tensor]:
        """Apply inverse transforms to targets if they exist.

        Args:
            context: Processing context with optional targets populated.

        Returns:
            dict[str, Tensor]: Inverse-transformed targets (empty if no targets).
        """
        # Handle optional targets - they may not be present during inference
        if not context.targets:
            return {}

        if not self.apply_inverse_target_transforms:
            return dict(context.targets)

        inv_targets: dict[str, Tensor] = {}

        # Use cached transform chains from predict transform step
        cache = getattr(self, "_transform_step_predict").cache  # type: ignore[attr-defined]

        for tname, ttensor in context.targets.items():
            chain = cache.get(tname)
            if chain:
                try:
                    inv_targets[tname] = self._inverse_with_chain(ttensor, chain)
                except Exception:
                    inv_targets[tname] = ttensor
            else:
                inv_targets[tname] = ttensor

        return inv_targets

    def _inverse_with_chain(self, tensor: Tensor, chain) -> Tensor:
        """Apply inverse transform with a transform chain, handling failures gracefully.

        Args:
            tensor (Tensor): Tensor to inverse transform.
            chain: Transform chain with inverse_transform method.

        Returns:
            Tensor: Inverse-transformed tensor, or original if inverse fails.
        """
        try:
            return chain.inverse_transform(tensor)
        except Exception:
            return tensor

    def _create_output_classifier(self):
        """Create output classifier for standard models.

        Use a name-based classifier to robustly map outputs like "output"
        to predictions by default in simple setups. This is more permissive
        for quick-start scenarios while remaining compatible with entry configs.
        """
        return NameBasedClassifier()


class BareWrapper(StandardLightningWrapper):
    """Minimal Lightning wrapper with basic functionality.

    This is a simplified version of StandardLightningWrapper that provides
    minimal functionality without the full processing pipeline integration.
    Useful for simple models that don't need complex dataflow processing.
    """

    def __init__(self, model_settings: ModelComponentSettings, **kwargs):
        """Initialize the bare wrapper.

        Args:
            model_settings: Model configuration settings
            **kwargs: Additional arguments (most will be ignored)
        """
        # Create minimal settings if not provided
        from dlkit.tools.config import WrapperComponentSettings
        from dlkit.core.shape_specs import create_shape_spec

        minimal_settings = WrapperComponentSettings()
        # Provide minimal shape spec for compatibility
        from dlkit.core.shape_specs import ModelFamily
        minimal_shape_spec = create_shape_spec({"x": (1,), "y": (1,)}, model_family=ModelFamily.EXTERNAL)

        super().__init__(
            settings=minimal_settings,
            model_settings=model_settings,
            shape_spec=minimal_shape_spec,
            **kwargs
        )

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Any]:
        """Simplified training step without full pipeline processing.

        Args:
            batch: Raw batch dataflow
            batch_idx: Index of the batch

        Returns:
            Dictionary containing the training loss
        """
        # Extract input (assume first tensor is input)
        x = next(iter(batch.values()))

        # Forward pass
        output = self.forward(x)

        # Simple loss computation (assumes second tensor is target if available)
        batch_values = list(batch.values())
        if len(batch_values) >= 2:
            target = batch_values[1]
            loss = self.loss_function(output, target)
        else:
            # If no target, assume supervised learning isn't the goal
            loss = torch.tensor(0.0, requires_grad=True)

        self._log_stage_outputs("train", loss)
        return {"loss": loss}

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Any]:
        """Simplified validation step without full pipeline processing.

        Args:
            batch: Raw batch dataflow
            batch_idx: Index of the batch

        Returns:
            Dictionary containing validation metrics
        """
        # Extract input
        x = next(iter(batch.values()))

        # Forward pass
        output = self.forward(x)

        # Simple loss computation
        batch_values = list(batch.values())
        metrics = None
        if len(batch_values) >= 2:
            target = batch_values[1]
            val_loss = self.loss_function(output, target)

            # Compute metrics if available
            if self.val_metrics:
                metrics = self.val_metrics(output, target)
        else:
            val_loss = torch.tensor(0.0)

        self._log_stage_outputs("val", val_loss, metrics)
        return {"val_loss": val_loss}

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Any]:
        """Simplified test step without full pipeline processing.

        Args:
            batch: Raw batch dataflow
            batch_idx: Index of the batch

        Returns:
            Dictionary containing test metrics
        """
        # Same as validation step but with test metrics
        x = next(iter(batch.values()))
        output = self.forward(x)

        batch_values = list(batch.values())
        metrics = None
        if len(batch_values) >= 2:
            target = batch_values[1]
            test_loss = self.loss_function(output, target)

            if self.test_metrics:
                metrics = self.test_metrics(output, target)
        else:
            test_loss = torch.tensor(0.0)

        self._log_stage_outputs("test", test_loss, metrics)
        return {"test_loss": test_loss}
