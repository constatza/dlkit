"""Standard Lightning wrapper for tensor-based models with transform support.

This module provides a Lightning wrapper that extends the base wrapper with
transform capabilities for data preprocessing and postprocessing.
"""

from typing import Any

import torch
from torch import Tensor
from torch.nn import ModuleDict, ModuleList
from loguru import logger

from dlkit.tools.config import ModelComponentSettings, WrapperComponentSettings
from dlkit.tools.config.data_entries import DataEntry, is_feature_entry, is_target_entry
from dlkit.core.shape_specs import IShapeSpec
from dlkit.core.training.transforms.chain import TransformChain
from dlkit.core.training.transforms.base import FittableTransform, InvertibleTransform
from .base import ProcessingLightningWrapper


class StandardLightningWrapper(ProcessingLightningWrapper):
    """Lightning wrapper for standard tensor-based neural networks with transforms.

    This wrapper extends the base wrapper with transform capabilities:
    - Fitted transforms persisted as ModuleDicts (feature and target transforms separated)
    - Forward transform application to features before model invocation
    - Inverse transform application to predictions for denormalization
    - Full checkpoint support for transform state

    Attributes:
        fitted_feature_transforms (torch.nn.ModuleDict): Persisted feature transform chains.
        fitted_target_transforms (torch.nn.ModuleDict): Persisted target transform chains.
        apply_feature_transforms (bool): Toggle for feature transform application.
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
            entry_configs: Data entry configurations
            shape_spec: Shape specification for models
            **kwargs: Additional arguments passed to base class
        """
        # Call base class initialization first
        super().__init__(
            settings=settings,
            model_settings=model_settings,
            entry_configs=entry_configs,
            shape_spec=shape_spec,
            **kwargs,
        )

        # Initialize transform-specific attributes
        # Separate Feature and Target transforms for clear separation of concerns
        self.fitted_feature_transforms: ModuleDict = ModuleDict()
        self.fitted_target_transforms: ModuleDict = ModuleDict()
        self.apply_feature_transforms: bool = True
        self.apply_inverse_target_transforms: bool = True

        # Transform cache for runtime usage (entry name → TransformChain)
        self._transform_cache: dict[str, TransformChain] = {}

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the model with tensor input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Model output tensor.
        """
        return self.model(x)

    # =============================================================================
    # Transform Application Helpers
    # =============================================================================

    def _apply_transforms(self, data: dict[str, Tensor], transforms: ModuleDict) -> dict[str, Tensor]:
        """Apply forward transforms to data.

        Args:
            data (dict[str, Tensor]): Input data (features or targets).
            transforms (ModuleDict): Transform chains to apply.

        Returns:
            dict[str, Tensor]: Transformed data.
        """
        if not self.apply_feature_transforms:
            return data

        transformed = {}
        for name, tensor in data.items():
            chain = self._transform_cache.get(name) or (transforms[name] if name in transforms else None)
            if chain is not None and callable(chain):
                try:
                    transformed[name] = chain(tensor)
                except Exception as e:
                    logger.warning(f"Transform application failed for '{name}': {e}")
                    transformed[name] = tensor
            else:
                transformed[name] = tensor
        return transformed

    def _apply_inverse_transforms(self, data: dict[str, Tensor], transforms: ModuleDict) -> dict[str, Tensor]:
        """Apply inverse transforms to data (predictions or targets).

        Args:
            data (dict[str, Tensor]): Predictions or targets to inverse transform.
            transforms (ModuleDict): Transform chains with inverse_transform methods.

        Returns:
            dict[str, Tensor]: Inverse-transformed data.
        """
        if not self.apply_inverse_target_transforms:
            return data

        inverse_transformed = {}
        for name, tensor in data.items():
            chain = self._transform_cache.get(name) or (transforms[name] if name in transforms else None)
            if chain is not None and isinstance(chain, InvertibleTransform):
                try:
                    inverse_transformed[name] = chain.inverse_transform(tensor)
                except Exception as e:
                    logger.warning(f"Inverse transform failed for '{name}': {e}")
                    inverse_transformed[name] = tensor
            else:
                inverse_transformed[name] = tensor
        return inverse_transformed

    # =============================================================================
    # Transform Application Helpers
    # =============================================================================

    def _apply_forward_feature_transforms(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply forward feature transforms if available.

        Args:
            features: Feature tensors to transform.

        Returns:
            Transformed features.
        """
        # Guard: Early return if no transforms
        if not self.fitted_feature_transforms:
            return features

        return self._apply_transforms(features, self.fitted_feature_transforms)

    def _apply_forward_target_transforms(self, targets: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply forward target transforms if available.

        Transforms targets to normalized space for loss computation.

        Args:
            targets: Target tensors to transform.

        Returns:
            Transformed targets.
        """
        # Guard: Early return if no transforms
        if not self.fitted_target_transforms:
            return targets

        return self._apply_transforms(targets, self.fitted_target_transforms)

    def _apply_inverse_target_transforms_to_predictions(
        self, predictions: dict[str, Tensor] | Tensor
    ) -> dict[str, Tensor] | Tensor:
        """Apply inverse target transforms to predictions.

        This handles both dict and single tensor predictions with proper error handling.

        Args:
            predictions: Model predictions (dict or single tensor).

        Returns:
            Predictions with inverse transforms applied.
        """
        # Guard: Early return if no transforms
        if not self.fitted_target_transforms:
            return predictions

        # Use match-case for type dispatch
        match predictions:
            case dict():
                return self._apply_inverse_transforms(predictions, self.fitted_target_transforms)
            case Tensor():
                return self._apply_inverse_to_single_tensor(predictions)
            case _:
                return predictions

    def _apply_inverse_to_single_tensor(self, prediction: Tensor) -> Tensor:
        """Apply inverse transform to single tensor prediction.

        Args:
            prediction: Single tensor prediction.

        Returns:
            Prediction with inverse transform applied (or original if not applicable).
        """
        # Guard: Only works with single target transform
        if len(self.fitted_target_transforms) != 1:
            return prediction

        # Get the single transform chain
        name = next(iter(self.fitted_target_transforms.keys()))
        chain = self.fitted_target_transforms[name]

        # Guard: Chain must be invertible
        if not isinstance(chain, InvertibleTransform):
            return prediction

        # Apply inverse with error handling
        try:
            return chain.inverse_transform(prediction)
        except Exception as e:
            logger.warning(f"Inverse transform failed: {e}")
            return prediction

    # =============================================================================
    # Overridden Step Methods (Add Transform Application)
    # =============================================================================

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Any]:
        """Training step with transform application.

        Transforms both features and targets to normalized space before computing loss.
        This ensures the model trains in normalized space for better convergence.

        Args:
            batch (dict[str, Tensor]): Raw batch from dataset.
            batch_idx (int): Index of the batch.

        Returns:
            dict[str, Any]: Dictionary containing the training loss.
        """
        # 1. Extract features and targets (inherited from base)
        features, targets = self._extract_features_targets(batch)

        # 2. Apply forward transforms to features (raw → normalized)
        features = self._apply_forward_feature_transforms(features)

        # 3. Apply forward transforms to targets (raw → normalized)
        targets = self._apply_forward_target_transforms(targets)

        # 4. Model forward (predicts in normalized space)
        predictions = self._invoke_model(features)

        # 5. Compute loss in normalized space (both predictions and targets normalized)
        loss = self._compute_loss(predictions, targets)

        # 6. Log metrics (inherited from base)
        self._log_stage_outputs("train", loss)

        return {"loss": loss}

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Any]:
        """Validation step with transform application.

        Transforms both features and targets to normalized space before computing loss.
        Metrics are also computed in normalized space for consistency.

        Args:
            batch (dict[str, Tensor]): Raw batch from dataset.
            batch_idx (int): Index of the batch.

        Returns:
            dict[str, Any]: Dictionary containing validation metrics.
        """
        # 1. Extract features and targets
        features, targets = self._extract_features_targets(batch)

        # 2. Apply forward transforms to features (raw → normalized)
        features = self._apply_forward_feature_transforms(features)

        # 3. Apply forward transforms to targets (raw → normalized)
        targets = self._apply_forward_target_transforms(targets)

        # 4. Model forward (predicts in normalized space)
        predictions = self._invoke_model(features)

        # 5. Compute loss in normalized space
        val_loss = self._compute_loss(predictions, targets)

        # 6. Compute metrics in normalized space
        metrics = self._update_metrics(predictions, targets, stage="val")

        # 7. Log metrics
        self._log_stage_outputs("val", val_loss, metrics)

        return {"val_loss": val_loss}

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Any]:
        """Test step with transform application.

        Transforms both features and targets to normalized space before computing loss.
        Metrics are also computed in normalized space for consistency.

        Args:
            batch (dict[str, Tensor]): Raw batch from dataset.
            batch_idx (int): Index of the batch.

        Returns:
            dict[str, Any]: Dictionary containing test metrics.
        """
        # 1. Extract features and targets
        features, targets = self._extract_features_targets(batch)

        # 2. Apply forward transforms to features (raw → normalized)
        features = self._apply_forward_feature_transforms(features)

        # 3. Apply forward transforms to targets (raw → normalized)
        targets = self._apply_forward_target_transforms(targets)

        # 4. Model forward (predicts in normalized space)
        predictions = self._invoke_model(features)

        # 5. Compute loss in normalized space
        test_loss = self._compute_loss(predictions, targets)

        # 6. Compute metrics in normalized space
        metrics = self._update_metrics(predictions, targets, stage="test")

        # 7. Log metrics
        self._log_stage_outputs("test", test_loss, metrics)

        return {"test_loss": test_loss}

    def predict_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, dict[str, Tensor]]:
        """Prediction step with transform application.

        Applies forward transforms to features and inverse transforms to both
        predictions and targets (if present) so outputs are in original data space.

        Args:
            batch (dict[str, Tensor]): Raw batch from dataset.
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary with ``predictions``, ``targets``, and ``latents``.
        """
        # 1. Extract features (targets are optional)
        features, targets = self._extract_features_targets(batch)

        # 2. Apply forward transforms to features
        if self.fitted_feature_transforms and self.apply_feature_transforms:
            features = self._apply_transforms(features, self.fitted_feature_transforms)

        # 3. Model forward
        predictions = self._invoke_model(features)

        # 4. Normalize predictions to dict format
        # Use target name for single-tensor predictions if available
        if isinstance(predictions, Tensor):
            target_configs = self.get_target_configs()
            target_name = next(iter(target_configs.keys())) if len(target_configs) == 1 else None
            predictions = {target_name: predictions} if target_name else {"output": predictions}

        # 5. Apply inverse transforms to predictions to get back to original space
        if self.fitted_target_transforms and self.apply_inverse_target_transforms:
            predictions = self._apply_inverse_transforms(predictions, self.fitted_target_transforms)

        # 6. Targets from dataloader are already in original space (raw data)
        # Do NOT apply inverse transforms to them

        return {
            "predictions": predictions,
            "targets": targets if targets else {},
            "latents": {}
        }

    # =============================================================================
    # Transform Management (Fitting, Checkpoint Persistence)
    # =============================================================================

    def on_fit_start(self) -> None:
        """Fit all entry-based transforms using the entire training dataloader.

        - Aggregates full training data per entry that has transforms and fits once
        - Populates fitted_feature_transforms and fitted_target_transforms ModuleDicts
        - Caches transforms for runtime usage
        """
        trainer = getattr(self, "trainer", None)
        if trainer is None or not hasattr(trainer, "datamodule"):
            return
        dm = trainer.datamodule
        if dm is None or not hasattr(dm, "train_dataloader"):
            return

        # Determine which entries have transforms configured
        entry_cfgs = getattr(self, "_entry_configs", {})
        names_with_transforms = [n for n, e in entry_cfgs.items() if getattr(e, "transforms", None)]
        if not names_with_transforms:
            return

        try:
            loader = dm.train_dataloader()
        except Exception:
            return

        # Aggregate tensors per entry name across the whole training set
        from collections import defaultdict

        buffers: dict[str, list[torch.Tensor]] = defaultdict(list)
        for batch in loader:
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
            chain = TransformChain(getattr(cfg, "transforms", ()))
            if isinstance(chain, FittableTransform):
                chain.fit(stacked)

            # Populate cache for runtime
            self._transform_cache[name] = chain

            # Persist to appropriate module dict based on data entry type
            if is_target_entry(cfg):
                self.fitted_target_transforms[name] = chain
            elif is_feature_entry(cfg):
                self.fitted_feature_transforms[name] = chain

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Prepare checkpoint for saving with transform and inference metadata.

        Ensures transform chains and inference configuration are persisted.
        """
        # Call parent to save enhanced dlkit_metadata
        super().on_save_checkpoint(checkpoint)

        # The fitted_transforms ModuleDicts are automatically saved by Lightning
        # Add inference metadata for standalone inference
        feature_names = [name for name, cfg in self._entry_configs.items() if is_feature_entry(cfg)]
        target_names = [name for name, cfg in self._entry_configs.items() if is_target_entry(cfg)]
        feature_transform_names = list(self.fitted_feature_transforms.keys())
        target_transform_names = list(self.fitted_target_transforms.keys())

        checkpoint["inference_metadata"] = {
            "entry_configs": self._entry_configs,
            "wrapper_settings": self._wrapper_settings.model_dump() if hasattr(self._wrapper_settings, 'model_dump') else dict(self._wrapper_settings),
            "feature_names": feature_names,
            "target_names": target_names,
            "feature_transform_names": feature_transform_names,
            "target_transform_names": target_transform_names,
            "model_shape": getattr(self, 'shape', None),
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Handle checkpoint loading with transform restoration.

        Ensures transform chains are properly loaded and caches are hydrated.
        Also validates checkpoint version via base class.
        """
        # Call base class to validate version and restore metadata
        super().on_load_checkpoint(checkpoint)

        # Manually restore fitted_transforms from checkpoint
        # Lightning's load_state_dict with strict=False may skip these if ModuleDict is empty
        self._restore_fitted_transforms_from_checkpoint(checkpoint)

        # Hydrate runtime cache from restored transforms
        self._hydrate_transform_cache()

    def on_predict_start(self) -> None:
        """Hydrate transform cache when loading from checkpoint for inference."""
        self._hydrate_transform_cache()

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True, assign: bool = False) -> Any:
        """Override load_state_dict to ensure fitted_transforms are restored.

        Handles both Lightning checkpoint loading and direct torch.load scenarios.
        """
        # First, let PyTorch/Lightning load what it can
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)

        # Then manually restore fitted_transforms (which may have been skipped)
        checkpoint = {
            "state_dict": state_dict,
            "inference_metadata": {
                "entry_configs": self._entry_configs if hasattr(self, '_entry_configs') else {}
            }
        }
        self._restore_fitted_transforms_from_checkpoint(checkpoint)

        # Hydrate caches from the restored transforms
        self._hydrate_transform_cache()

        return result

    # =============================================================================
    # Transform Restoration and Cache Hydration
    # =============================================================================

    def _restore_fitted_transforms_from_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Manually restore fitted transforms ModuleDicts from checkpoint.

        Lightning's load_state_dict with strict=False skips transform keys
        when ModuleDicts are empty. This method reconstructs chains from
        checkpoint state_dict using the original transform configurations.

        Supports both legacy (fitted_transforms) and modern (fitted_feature_transforms/
        fitted_target_transforms) checkpoint formats.
        """
        logger.info("Starting manual restoration of fitted transforms from checkpoint")

        try:
            state_dict = checkpoint.get("state_dict", {})
            inference_metadata = checkpoint.get("inference_metadata", {})

            # Check for modern format (separate feature/target transforms)
            has_modern_format = any(
                k.startswith("fitted_feature_transforms.") or k.startswith("fitted_target_transforms.")
                for k in state_dict.keys()
            )
            has_legacy_format = any(k.startswith("fitted_transforms.") for k in state_dict.keys())

            logger.info(f"Checkpoint format: modern={has_modern_format}, legacy={has_legacy_format}")

            # Get entry configs: prefer instance's entry_configs over checkpoint metadata
            # This ensures transforms are properly restored when loading with entry_configs parameter
            entry_configs_data = self._entry_configs or inference_metadata.get("entry_configs", {})

            # Helper function to process transform keys
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
                                            # Handle ModuleList indexing
                                            if isinstance(module, ModuleList):
                                                module = module[int(part)]
                                            else:
                                                module = module.transforms[int(part)]
                                        else:
                                            module = getattr(module, part)
                                    buffer_name = parts[-1]
                                    buffer_value = entry_state[unexpected_key]
                                    module.register_buffer(buffer_name, buffer_value)
                                    logger.info(f"Manually registered buffer: {unexpected_key} = {buffer_value}")
                                except Exception as e:
                                    logger.warning(f"Could not register unexpected key {unexpected_key}: {e}")

                        target_module_dict[entry_name] = chain
                        logger.info(f"Successfully restored {prefix} chain for '{entry_name}'")
                    except Exception as e:
                        logger.error(f"Failed to restore {prefix} chain for '{entry_name}': {e}")

            # Restore based on checkpoint format
            if has_modern_format:
                restore_transforms("fitted_feature_transforms", self.fitted_feature_transforms, is_feature_entry)
                restore_transforms("fitted_target_transforms", self.fitted_target_transforms, is_target_entry)
            elif has_legacy_format:
                # Legacy format - separate based on entry_configs
                logger.info("Loading legacy format, separating transforms by entry type")
                restore_transforms("fitted_transforms", self.fitted_feature_transforms, is_feature_entry)
                restore_transforms("fitted_transforms", self.fitted_target_transforms, is_target_entry)
            else:
                logger.debug("No fitted transforms found in checkpoint")

        except Exception as e:
            logger.warning(f"Failed to restore fitted_transforms from checkpoint: {e}")

    def _hydrate_transform_cache(self) -> None:
        """Populate transform cache from persisted ModuleDicts.

        Enables inference-only runs with just a predict_dataloader by reusing
        transform chains loaded from checkpoint.
        """
        try:
            has_feature_transforms = isinstance(self.fitted_feature_transforms, ModuleDict) and len(self.fitted_feature_transforms) > 0
            has_target_transforms = isinstance(self.fitted_target_transforms, ModuleDict) and len(self.fitted_target_transforms) > 0

            if not has_feature_transforms and not has_target_transforms:
                return
        except Exception:
            return

        # Hydrate cache from both feature and target transforms
        for name, chain in self.fitted_feature_transforms.items():
            self._transform_cache[name] = chain

        for name, chain in self.fitted_target_transforms.items():
            self._transform_cache[name] = chain

    # =============================================================================
    # Public Helper Methods
    # =============================================================================

    def feature_transforms(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply configured feature transforms to a features dict.

        Uses transform cache or persisted fitted_feature_transforms.
        Entries without a chain are returned unchanged.

        Args:
            features (dict[str, Tensor]): Features to transform.

        Returns:
            dict[str, Tensor]: Transformed features.
        """
        return self._apply_transforms(features, self.fitted_feature_transforms)

    def target_transforms_inverse(self, tensors: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply inverse transforms for targets/predictions by entry name.

        Looks up transform chains in cache or fitted_target_transforms and applies
        inverse_transform when available. Entries without an invertible chain are
        returned unchanged.

        Args:
            tensors (dict[str, Tensor]): Tensors to inverse transform.

        Returns:
            dict[str, Tensor]: Inverse-transformed tensors.
        """
        return self._apply_inverse_transforms(tensors, self.fitted_target_transforms)


class BareWrapper(StandardLightningWrapper):
    """Minimal Lightning wrapper with basic functionality.

    This is a simplified version of StandardLightningWrapper that provides
    minimal functionality without the full transform integration.
    Useful for simple models that don't need complex data processing.
    """

    def __init__(self, model_settings: ModelComponentSettings, **kwargs):
        """Initialize the bare wrapper.

        Args:
            model_settings: Model configuration settings
            **kwargs: Additional arguments (most will be ignored)
        """
        # Create minimal settings if not provided
        from dlkit.tools.config import WrapperComponentSettings
        from dlkit.core.shape_specs import create_shape_spec, ModelFamily

        minimal_settings = WrapperComponentSettings()
        # Provide minimal shape spec for compatibility
        minimal_shape_spec = create_shape_spec({"x": (1,), "y": (1,)}, model_family=ModelFamily.EXTERNAL)

        super().__init__(
            settings=minimal_settings,
            model_settings=model_settings,
            shape_spec=minimal_shape_spec,
            **kwargs
        )

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Any]:
        """Simplified training step without transform processing.

        Args:
            batch: Raw batch data
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
            loss = torch.tensor(0.0, requires_grad=True)

        self._log_stage_outputs("train", loss)
        return {"loss": loss}

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Any]:
        """Simplified validation step without transform processing.

        Args:
            batch: Raw batch data
            batch_idx: Index of the batch

        Returns:
            Dictionary containing validation metrics
        """
        x = next(iter(batch.values()))
        output = self.forward(x)

        batch_values = list(batch.values())
        metrics = None
        if len(batch_values) >= 2:
            target = batch_values[1]
            val_loss = self.loss_function(output, target)

            if self.val_metrics:
                metrics = self.val_metrics(output, target)
        else:
            val_loss = torch.tensor(0.0)

        self._log_stage_outputs("val", val_loss, metrics)
        return {"val_loss": val_loss}

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Any]:
        """Simplified test step without transform processing.

        Args:
            batch: Raw batch data
            batch_idx: Index of the batch

        Returns:
            Dictionary containing test metrics
        """
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
