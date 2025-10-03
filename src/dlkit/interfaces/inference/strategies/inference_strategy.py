"""Inference strategy with direct model execution.

This module provides a production-optimized inference strategy that executes
models directly using model.forward() calls, bypassing Lightning overhead
for minimal latency and maximum throughput.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..config.inference_config import InferenceConfig
from ..inputs.inference_input import InferenceInput
from ..transforms.executor import TransformChainExecutor
from dlkit.interfaces.api.domain.models import InferenceResult, ModelState


class InferenceStrategy:
    """Inference strategy with direct model execution.

    This strategy provides minimal-overhead inference by:
    1. Loading model directly from checkpoint
    2. Applying transforms via standalone executor
    3. Executing inference via direct model.forward() calls
    4. Processing results with inverse transforms

    The strategy operates completely independently from Lightning and
    training configurations, requiring only the model checkpoint.
    """

    def __init__(self) -> None:
        """Initialize inference strategy."""
        self._model: torch.nn.Module | None = None
        self._transform_executor: TransformChainExecutor | None = None
        self._device: torch.device = torch.device("cpu")

    def load_model_from_checkpoint(
        self,
        config: InferenceConfig
    ) -> None:
        """Load model and transforms from checkpoint.

        Args:
            config: Inference configuration

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If checkpoint loading fails
        """
        checkpoint_path = config.model_checkpoint_path

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        # Resolve target device
        self._device = config.resolve_device()

        try:
            # Load checkpoint
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self._device,
                weights_only=False
            )

            # Load model from checkpoint
            self._model = self._load_model_from_checkpoint_data(checkpoint, config)

            # Load transform executor
            if config.has_transforms():
                self._transform_executor = TransformChainExecutor.from_checkpoint(checkpoint_path)

        except Exception as e:
            raise ValueError(f"Failed to load model from checkpoint: {e}") from e

    def _load_model_from_checkpoint_data(
        self,
        checkpoint: dict[str, Any],
        config: InferenceConfig
    ) -> torch.nn.Module:
        """Load model from checkpoint data.

        Args:
            checkpoint: Loaded checkpoint dictionary
            config: Inference configuration

        Returns:
            Loaded model instance

        Raises:
            ValueError: If model loading fails
        """
        if "state_dict" not in checkpoint:
            raise ValueError("Invalid checkpoint: missing state_dict")

        # Try to reconstruct model from inference metadata
        if "inference_metadata" in checkpoint:
            model = self._reconstruct_model_from_metadata(
                checkpoint["inference_metadata"],
                config
            )
        else:
            raise ValueError(
                "Cannot reconstruct model: inference metadata missing. "
                "This checkpoint may not support inference."
            )

        # Load model weights
        try:
            # Handle Lightning wrapper state dict format
            state_dict = checkpoint["state_dict"]

            # Remove wrapper prefixes if present (e.g., "model.layer.weight" -> "layer.weight")
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("model."):
                    cleaned_key = key[6:]  # Remove "model." prefix
                    cleaned_state_dict[cleaned_key] = value
                elif not key.startswith("fitted_transforms."):
                    # Skip transform parameters, keep other model parameters
                    cleaned_state_dict[key] = value

            # Load state dict with strict=False to handle potential mismatches
            missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)

            if missing_keys:
                print(f"Warning: Missing keys in model state dict: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in model state dict: {unexpected_keys}")

        except Exception as e:
            raise ValueError(f"Failed to load model weights: {e}") from e

        # Move model to target device and set to eval mode
        model = model.to(self._device)
        model.eval()

        return model

    def _reconstruct_model_from_metadata(
        self,
        metadata: dict[str, Any],
        config: InferenceConfig
    ) -> torch.nn.Module:
        """Reconstruct model from inference metadata.

        Args:
            metadata: Inference metadata from checkpoint
            config: Inference configuration

        Returns:
            Model instance ready for weight loading

        Raises:
            ValueError: If model reconstruction fails
        """
        # Get wrapper settings and model shape
        wrapper_settings = metadata.get("wrapper_settings", {})
        model_shape = metadata.get("model_shape")

        if not wrapper_settings:
            raise ValueError("Wrapper settings missing from inference metadata")

        # Try to reconstruct model using factory
        try:
            from dlkit.tools.config import FactoryProvider, BuildContext, ModelComponentSettings

            # Reconstruct model settings
            if isinstance(wrapper_settings, dict):
                # Convert dict back to settings if needed
                from dlkit.tools.config import WrapperComponentSettings

                # Create minimal model component settings
                # This is a simplified approach - in practice, you might need
                # more sophisticated model reconstruction
                model_settings = ModelComponentSettings(
                    name="auto",  # Will be determined from metadata
                    module_path="auto"
                )

                # Build context with shape information
                overrides = {}
                if model_shape:
                    overrides["shape"] = self._convert_shape_for_model(model_shape)

                build_context = BuildContext(mode="inference", overrides=overrides)

                # Create model via factory
                model = FactoryProvider.create_component(model_settings, build_context)

                return model

            else:
                raise ValueError("Invalid wrapper settings format")

        except Exception as e:
            raise ValueError(f"Failed to reconstruct model from metadata: {e}") from e

    def _convert_shape_for_model(self, shape: Any) -> dict[str, tuple[int, ...]]:
        """Convert shape metadata for model constructor.

        Args:
            shape: Shape information from metadata

        Returns:
            Shape dictionary for model construction
        """
        # Simple conversion - in practice this might need to be more sophisticated
        if hasattr(shape, 'x') and shape.x:
            return {"x": tuple(shape.x)}
        else:
            return {}

    def _extract_model_shape_from_wrapper_settings(self, wrapper_settings: Any) -> dict[str, tuple[int, ...]]:
        """Extract model shape from wrapper settings.

        Args:
            wrapper_settings: Wrapper settings object with shape attributes

        Returns:
            Dictionary mapping feature names to shape tuples
        """
        shape: dict[str, tuple[int, ...]] = {}

        # Extract shape for common attributes
        for attr_name in ['x', 'y', 'features', 'targets']:
            if hasattr(wrapper_settings, attr_name):
                attr_value = getattr(wrapper_settings, attr_name)
                if attr_value is not None:
                    try:
                        # Convert to tuple if it's a list or other iterable
                        if hasattr(attr_value, '__iter__') and not isinstance(attr_value, (str, bytes)):
                            shape[attr_name] = tuple(int(d) for d in attr_value)
                        else:
                            # Single dimension
                            shape[attr_name] = (int(attr_value),)
                    except (ValueError, TypeError):
                        continue

        return shape

    def _extract_input_tensors(self, inputs: InferenceInput) -> dict[str, Tensor]:
        """Extract input tensors from InferenceInput.

        Args:
            inputs: Input data in flexible format

        Returns:
            Dictionary of input tensors
        """
        # This is essentially a wrapper around the to_tensor_dict method
        # with default parameters
        return inputs.to_tensor_dict()

    def infer(
        self,
        inputs: InferenceInput,
        config: InferenceConfig,
        batch_size: int | None = None
    ) -> InferenceResult:
        """Execute inference.

        Args:
            inputs: Input data in flexible format
            config: Inference configuration
            batch_size: Batch size for processing (None = use config default)

        Returns:
            InferenceResult: Inference execution result

        Raises:
            ValueError: If model not loaded or inference fails
        """
        if self._model is None:
            raise ValueError("Model not loaded. Call load_model_from_checkpoint() first.")

        batch_size = batch_size or config.batch_size

        try:
            # Convert inputs to tensor dictionary
            input_tensors = inputs.to_tensor_dict(
                feature_names=config.get_feature_names(),
                device=self._device,
                dtype=config.dtype
            )

            # Validate inputs
            validation_errors = config.validate_inputs(input_tensors)
            if validation_errors:
                raise ValueError(f"Input validation failed: {validation_errors}")

            # Apply feature transforms if available
            if self._transform_executor and config.has_transforms():
                input_tensors = self._transform_executor.apply_feature_transforms(
                    input_tensors,
                    feature_names=config.get_feature_names()
                )

            # Execute inference in batches
            predictions = self._execute_batched_inference(input_tensors, batch_size)

            # Apply inverse transforms to predictions if available
            if self._transform_executor and config.has_transforms():
                predictions = self._transform_executor.apply_inverse_target_transforms(
                    predictions,
                    target_names=config.get_target_names()
                )

            # Create minimal model state for result
            # In inference, we only need the basic model reference
            if self._model is None:
                raise RuntimeError("Model not loaded")

            if not hasattr(self._model, 'forward'):
                raise RuntimeError("Model must have a forward method")

            from dlkit.tools.config import GeneralSettings
            from lightning.pytorch import LightningDataModule, LightningModule

            # Create a minimal datamodule placeholder for type compatibility
            class MinimalDataModule(LightningDataModule):
                def __init__(self):
                    super().__init__()

            # For inference, we create a wrapper to make the model Lightning-compatible
            # if it's not already a LightningModule
            if not isinstance(self._model, LightningModule):
                # Wrap the model in a minimal LightningModule for compatibility
                class InferenceWrapper(LightningModule):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model

                    def forward(self, *args, **kwargs):
                        return self.model(*args, **kwargs)

                wrapped_model = InferenceWrapper(self._model)
            else:
                wrapped_model = self._model

            model_state = ModelState(
                model=wrapped_model,  # Now properly typed as LightningModule
                datamodule=MinimalDataModule(),  # Minimal placeholder
                trainer=None,     # Not using Lightning trainer
                settings=GeneralSettings()  # Minimal settings
            )

            # Create inference result
            result = InferenceResult(
                model_state=model_state,
                predictions=predictions,
                metrics=None,     # No metrics in inference
                duration_seconds=0.0  # Could add timing if needed
            )

            return result

        except Exception as e:
            raise ValueError(f"Inference execution failed: {e}") from e

    def _execute_batched_inference(
        self,
        input_tensors: dict[str, Tensor],
        batch_size: int
    ) -> dict[str, Tensor]:
        """Execute inference in batches to handle large inputs.

        Args:
            input_tensors: Dictionary of input tensors
            batch_size: Batch size for processing

        Returns:
            Dictionary of prediction tensors
        """
        # Determine total number of samples
        first_tensor = next(iter(input_tensors.values()))
        total_samples = first_tensor.shape[0]

        if total_samples <= batch_size:
            # Single batch processing
            return self._execute_single_batch(input_tensors)

        # Multi-batch processing
        all_predictions = {}

        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)

            # Extract batch
            batch_inputs = {
                name: tensor[start_idx:end_idx]
                for name, tensor in input_tensors.items()
            }

            # Execute batch
            batch_predictions = self._execute_single_batch(batch_inputs)

            # Accumulate predictions
            for name, pred_tensor in batch_predictions.items():
                if name not in all_predictions:
                    all_predictions[name] = []
                all_predictions[name].append(pred_tensor)

        # Concatenate batch results
        final_predictions = {}
        for name, pred_list in all_predictions.items():
            final_predictions[name] = torch.cat(pred_list, dim=0)

        return final_predictions

    def _execute_single_batch(self, input_tensors: dict[str, Tensor]) -> dict[str, Tensor]:
        """Execute inference on a single batch.

        Args:
            input_tensors: Dictionary of input tensors

        Returns:
            Dictionary of prediction tensors
        """
        with torch.no_grad():
            # Ensure model is loaded
            if self._model is None:
                raise RuntimeError("Model not loaded. Call load_model_from_checkpoint() first.")

            # Prepare model inputs
            model_inputs = self._prepare_model_inputs(input_tensors)

            # Execute model forward pass
            model_outputs = self._model(**model_inputs)

            # Normalize outputs to dictionary format
            predictions = self._normalize_model_outputs(model_outputs)

            return predictions

    def _prepare_model_inputs(self, input_tensors: dict[str, Tensor]) -> dict[str, Tensor]:
        """Prepare inputs for model forward pass.

        Args:
            input_tensors: Dictionary of input tensors

        Returns:
            Dictionary ready for model(**inputs)
        """
        # For most models, inputs can be passed directly
        # Some models might need special input preparation
        return input_tensors

    def _normalize_model_outputs(self, model_outputs: Any) -> dict[str, Tensor]:
        """Normalize model outputs to dictionary format.

        Args:
            model_outputs: Raw model outputs

        Returns:
            Dictionary of prediction tensors
        """
        if isinstance(model_outputs, dict):
            return model_outputs
        elif isinstance(model_outputs, Tensor):
            return {"output": model_outputs}
        elif isinstance(model_outputs, (list, tuple)):
            return {f"output_{i}": output for i, output in enumerate(model_outputs)}
        else:
            raise ValueError(f"Unsupported model output type: {type(model_outputs)}")

    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference.

        Returns:
            True if model is loaded
        """
        return self._model is not None

    def get_device(self) -> torch.device:
        """Get the device where model is loaded.

        Returns:
            Device where model is running
        """
        return self._device

    def unload_model(self) -> None:
        """Unload model and free memory."""
        self._model = None
        self._transform_executor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()