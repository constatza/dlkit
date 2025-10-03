"""Infrastructure adapters for inference operations.

Concrete implementations of domain interfaces,
handling PyTorch-specific operations and external dependencies.
"""

from __future__ import annotations

import torch
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger

from dlkit.core.shape_specs import ShapeSpec
from dlkit.tools.config.components.model_components import ModelComponentSettings
from dlkit.tools.config.core.context import BuildContext
from dlkit.tools.config.core.factories import FactoryProvider
from dlkit.interfaces.api.domain.errors import WorkflowError

from ..domain.ports import (
    IModelLoader,
    IModelStateManager,
    IInferenceExecutor,
    ICheckpointReconstructor
)
from ..domain.models import (
    ModelState,
    ModelStateType,
    InferenceRequest,
    BatchProcessingConfig
)


class CheckpointReconstructor:
    """Adapter for checkpoint metadata extraction and model settings reconstruction."""

    def reconstruct_model_settings(
        self,
        checkpoint_path: Path
    ) -> Optional[ModelComponentSettings]:
        """Reconstruct model settings from checkpoint metadata."""
        try:
            if not checkpoint_path.exists():
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                return None

            # Load checkpoint metadata
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            if not isinstance(checkpoint, dict):
                logger.warning(f"Invalid checkpoint format: expected dict, got {type(checkpoint)}")
                return None

            # Try enhanced metadata first
            metadata = checkpoint.get('dlkit_metadata', {})
            if metadata and 'model_settings' in metadata:
                settings_data = metadata['model_settings']
                return self._deserialize_model_settings(settings_data)

            # Legacy checkpoint - cannot reconstruct model settings
            logger.warning(
                "Legacy checkpoint format detected without model settings. "
                "Automatic reconstruction not supported for this checkpoint format."
            )
            return None

        except Exception as e:
            logger.warning(f"Failed to reconstruct model settings from {checkpoint_path}: {e}")
            return None

    def extract_checkpoint_metadata(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Extract metadata from checkpoint file."""
        try:
            if not checkpoint_path.exists():
                raise WorkflowError(
                    f"Checkpoint not found: {checkpoint_path}",
                    {"component": "CheckpointReconstructor", "operation": "extract_metadata"}
                )

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            if not isinstance(checkpoint, dict):
                raise WorkflowError(
                    f"Invalid checkpoint format: expected dict, got {type(checkpoint)}",
                    {"component": "CheckpointReconstructor", "operation": "extract_metadata"}
                )

            return checkpoint

        except WorkflowError:
            raise
        except Exception as e:
            raise WorkflowError(
                f"Failed to extract checkpoint metadata: {str(e)}",
                {"component": "CheckpointReconstructor", "operation": "extract_metadata"}
            ) from e

    def _deserialize_model_settings(self, settings_data: Dict[str, Any]) -> Optional[ModelComponentSettings]:
        """Deserialize model settings from checkpoint metadata."""
        try:
            name = settings_data.get('name')
            module_path = settings_data.get('module_path')
            params = settings_data.get('params', {}) or {}

            if name is None:
                return None

            # Merge params with any additional top-level fields
            extra_fields = {
                key: value
                for key, value in settings_data.items()
                if key not in {"name", "module_path", "params", "class_name"}
            }

            init_kwargs = {**params, **extra_fields}

            return ModelComponentSettings(
                name=name,
                module_path=module_path,
                **init_kwargs,
            )

        except Exception as e:
            logger.warning(f"Failed to deserialize model settings: {e}")
            return None


class TorchModelStateManager:
    """PyTorch-specific model state management.

    This is where eval() is called exactly once during state transitions.
    """

    def create_state(self, model: Any, metadata: Dict[str, Any]) -> ModelState:
        """Create managed model state from raw PyTorch model."""
        return ModelState(
            model=model,
            state_type=ModelStateType.LOADED,
            device="cpu",  # Default device
            metadata=metadata
        )

    def transition_to_inference(self, model_state: ModelState) -> ModelState:
        """Transition model to inference mode.

        This is the SINGLE PLACE where eval() is called.
        """
        if model_state.state_type == ModelStateType.INFERENCE_READY:
            # Already in inference mode, return as-is
            logger.debug("Model already in inference mode, skipping eval()")
            return model_state

        try:
            # This is where eval() gets called exactly once
            model_state.model.eval()
            logger.info("Model transitioned to inference mode (eval() called)")

            return model_state.with_inference_ready()

        except Exception as e:
            error_info = {
                "operation": "transition_to_inference",
                "error": str(e)
            }
            return model_state.with_error(error_info)

    def is_inference_ready(self, model_state: ModelState) -> bool:
        """Check if model state is ready for inference."""
        return model_state.is_inference_ready()


class PyTorchModelLoader:
    """PyTorch-specific model loader implementation."""

    def __init__(self, model_state_manager: TorchModelStateManager):
        """Initialize with model state manager."""
        self._model_state_manager = model_state_manager

    def load_from_checkpoint(
        self,
        checkpoint_path: Path,
        model_settings: ModelComponentSettings,
        shape_spec: ShapeSpec
    ) -> ModelState:
        """Load PyTorch model from checkpoint with proper state management."""
        try:
            logger.info(f"Loading model from checkpoint: {checkpoint_path}")

            # Create build context for model instantiation
            build_context = BuildContext(mode="inference")

            # Apply shape overrides for model creation
            if shape_spec and not shape_spec.is_empty():
                # Provide unified shape specification directly to shape-aware models
                canonical_spec = shape_spec.with_canonical_aliases()
                build_context = build_context.with_overrides(unified_shape=canonical_spec)
                logger.info("Applied shape specification overrides via unified_shape")

            # Create model using factory
            model = FactoryProvider.create_component(model_settings, build_context)
            logger.info(f"Model created: {type(model).__name__}")

            # Load state dict
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = self._extract_state_dict(checkpoint)

            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            # Log key mismatches for debugging
            if missing_keys:
                logger.warning(f"Missing keys when loading checkpoint: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")

            # Create initial model state
            metadata = {
                "checkpoint_path": str(checkpoint_path),
                "model_settings": model_settings,
                "shape_spec": shape_spec,
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys
            }

            return self._model_state_manager.create_state(model, metadata)

        except Exception as e:
            raise WorkflowError(
                f"Failed to load model from checkpoint: {str(e)}",
                {"component": "PyTorchModelLoader", "checkpoint": str(checkpoint_path)}
            ) from e

    def prepare_for_inference(self, model_state: ModelState, device: str) -> ModelState:
        """Prepare model for inference with device placement and eval mode."""
        try:
            # Handle device placement
            if device == "auto":
                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"

            # Move model to device
            model_state.model.to(device)
            logger.info(f"Model moved to device: {device}")

            # Update state with device
            device_updated_state = model_state.with_device(device)

            # Transition to inference mode (this calls eval() exactly once)
            inference_ready_state = self._model_state_manager.transition_to_inference(
                device_updated_state
            )

            logger.info("Model prepared for inference")
            return inference_ready_state

        except Exception as e:
            raise WorkflowError(
                f"Failed to prepare model for inference: {str(e)}",
                {"component": "PyTorchModelLoader", "device": device}
            ) from e

    def _extract_state_dict(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Extract state dict from checkpoint with flexible key handling."""
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            # Fallback: assume entire checkpoint is state dict
            state_dict = checkpoint

        # Handle common key prefix issues
        if isinstance(state_dict, dict) and all(k.startswith("model.") for k in state_dict.keys()):
            logger.info("Stripping 'model.' prefix from state dict keys")
            state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}

        return state_dict


class DirectInferenceExecutor:
    """Direct inference executor using PyTorch models.

    Executes inference directly on loaded models with batch processing.
    """

    def execute_inference(
        self,
        model_state: ModelState,
        request: InferenceRequest
    ) -> Dict[str, Any]:
        """Execute inference with batch processing and transform handling."""
        if not model_state.is_inference_ready():
            raise WorkflowError(
                "Model is not in inference-ready state",
                {"component": "DirectInferenceExecutor", "model_state": str(model_state)}
            )

        try:
            # Process inference with torch.no_grad() for efficiency
            with torch.no_grad():
                return self._run_inference_batched(model_state, request)

        except Exception as e:
            raise WorkflowError(
                f"Inference execution failed: {str(e)}",
                {"component": "DirectInferenceExecutor", "request": str(request)}
            ) from e

    def _run_inference_batched(
        self,
        model_state: ModelState,
        request: InferenceRequest
    ) -> Dict[str, Any]:
        """Run inference with batch processing."""
        model = model_state.model
        inputs = request.inputs
        batch_size = request.batch_size

        target_device = self._resolve_target_device(model_state, request)

        if target_device != model_state.device:
            # Align model with requested device if it has changed post-preparation
            model.to(target_device)
            model_state = model_state.with_device(target_device)

        inputs = self._move_inputs_to_device(inputs, target_device)

        # For now, implement simple direct inference
        # This can be extended with proper batch processing logic
        if hasattr(model, 'predict_step'):
            # Use Lightning predict_step if available
            predictions = model.predict_step(inputs, 0)
            if isinstance(predictions, dict) and "predictions" in predictions:
                return predictions["predictions"]
            return predictions
        else:
            # Direct model forward pass
            if isinstance(inputs, dict):
                # Handle dict inputs - use first tensor for simple case
                input_tensor = next(iter(inputs.values()))
                output = model(input_tensor)
                # Return with appropriate key
                return {"predictions": output}
            else:
                # Handle single tensor input
                output = model(inputs)
                return {"predictions": output}

    def _resolve_target_device(
        self,
        model_state: ModelState,
        request: InferenceRequest
    ) -> str:
        """Determine the device to use for inference execution."""
        if request.device and request.device != "auto":
            return request.device
        if model_state.device:
            return model_state.device
        return "cpu"

    def _move_inputs_to_device(
        self,
        inputs: Any,
        device: str
    ) -> Any:
        """Recursively move inputs to the specified device when possible."""
        if torch.is_tensor(inputs):
            return inputs.to(device)
        if isinstance(inputs, dict):
            return {key: self._move_inputs_to_device(value, device) for key, value in inputs.items()}
        if isinstance(inputs, (list, tuple)):
            moved = [self._move_inputs_to_device(item, device) for item in inputs]
            return type(inputs)(moved)
        return inputs
