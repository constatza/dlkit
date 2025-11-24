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
from dlkit.interfaces.api.domain.errors import WorkflowError, ModelLoadingError
from dlkit.interfaces.api.services.precision_service import get_precision_service
from dlkit.interfaces.inference.checkpoint_utils import extract_state_dict

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
        """Reconstruct model settings from checkpoint metadata.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            ModelComponentSettings if found, None otherwise

        Note:
            Only supports v2.0+ checkpoints with 'dlkit_metadata'. Legacy checkpoints
            without this metadata format are no longer supported.
        """
        try:
            if not checkpoint_path.exists():
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                return None

            # Load checkpoint metadata
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            if not isinstance(checkpoint, dict):
                logger.warning(f"Invalid checkpoint format: expected dict, got {type(checkpoint)}")
                return None

            # Only support dlkit_metadata format (v2.0+)
            if 'dlkit_metadata' not in checkpoint:
                logger.error(
                    "Checkpoint missing 'dlkit_metadata'. This checkpoint uses a legacy format "
                    "that is no longer supported. Please re-train your model with the current "
                    "version of dlkit to generate a compatible checkpoint."
                )
                return None

            metadata = checkpoint['dlkit_metadata']

            # Validate version
            version = metadata.get('version')
            if version != '2.0':
                logger.error(
                    f"Unsupported checkpoint version '{version}'. Only version '2.0' is supported. "
                    "Please re-train your model with the current version of dlkit."
                )
                return None

            if 'model_settings' not in metadata:
                logger.warning("Checkpoint metadata missing 'model_settings' field")
                return None

            settings_data = metadata['model_settings']
            return self._deserialize_model_settings(settings_data)

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

    def create_state(
        self,
        model: Any,
        metadata: Dict[str, Any],
        feature_transforms: Optional[Dict[str, Any]] = None,
        target_transforms: Optional[Dict[str, Any]] = None
    ) -> ModelState:
        """Create managed model state from raw PyTorch model.

        Args:
            model: PyTorch model instance
            metadata: Model metadata dictionary
            feature_transforms: Optional dictionary of feature transform chains
            target_transforms: Optional dictionary of target transform chains

        Returns:
            ModelState with loaded model and separated transforms
        """
        return ModelState(
            model=model,
            state_type=ModelStateType.LOADED,
            device="cpu",  # Default device
            feature_transforms=feature_transforms,
            target_transforms=target_transforms,
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
            # Validate checkpoint exists and warn if not
            if not checkpoint_path.exists():
                logger.warning(
                    f"Checkpoint file not found: {checkpoint_path}. "
                    f"This will likely cause inference to fail."
                )
                raise WorkflowError(
                    f"Checkpoint file not found: {checkpoint_path}",
                    {"component": "PyTorchModelLoader", "checkpoint": str(checkpoint_path)}
                )

            logger.info(f"Loading model from checkpoint: {checkpoint_path}")

            # Load checkpoint FIRST to detect dtype
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = self._extract_state_dict(checkpoint)

            # Detect checkpoint dtype BEFORE creating model
            checkpoint_dtype = self._detect_checkpoint_dtype(state_dict)
            if checkpoint_dtype is not None:
                logger.info(f"Detected checkpoint dtype: {checkpoint_dtype}")
            else:
                logger.warning("Could not detect checkpoint dtype, using default")
                checkpoint_dtype = torch.float32

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

            # CRITICAL: Convert model to checkpoint's dtype BEFORE loading state_dict
            # This prevents precision loss from dtype conversion during load_state_dict
            model = model.to(dtype=checkpoint_dtype)
            logger.info(f"Converted model to checkpoint dtype: {checkpoint_dtype}")

            # Now load state dict - no dtype conversion needed since dtypes match
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            # Log key mismatches for debugging
            if missing_keys:
                logger.warning(f"Missing keys when loading checkpoint: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")

            # CRITICAL VALIDATION: Check if model weights actually loaded
            # Filter out non-critical keys (transforms, optimizer, scheduler, etc.)
            model_missing_keys = [
                k for k in missing_keys
                if not any(k.startswith(prefix) for prefix in [
                    'fitted_transforms.', 'fitted_feature_transforms.', 'fitted_target_transforms.',
                    'optimizer', 'lr_scheduler', '_metadata'
                ])
            ]

            if model_missing_keys:
                # Get total number of model parameters
                total_model_params = sum(1 for _ in model.state_dict().keys())
                missing_model_params = len(model_missing_keys)
                missing_percentage = (missing_model_params / total_model_params * 100) if total_model_params > 0 else 0

                # Raise error if >80% of model parameters are missing (likely key mismatch)
                if missing_percentage > 80:
                    raise ModelLoadingError(
                        f"Failed to load model weights from checkpoint: {missing_percentage:.1f}% of model "
                        f"parameters are missing ({missing_model_params}/{total_model_params}). "
                        f"This usually indicates a state dict key mismatch. "
                        f"Missing keys: {model_missing_keys[:10]}{'...' if len(model_missing_keys) > 10 else ''}. "
                        f"Check that checkpoint format matches model architecture.",
                        {"checkpoint_path": str(checkpoint_path),
                         "missing_percentage": missing_percentage,
                         "missing_count": missing_model_params,
                         "total_count": total_model_params}
                    )

            # Extract entry_configs from checkpoint for transform type information
            entry_configs = None
            if isinstance(checkpoint, dict):
                inference_metadata = checkpoint.get("inference_metadata", {})
                if inference_metadata:
                    entry_configs = inference_metadata.get("entry_configs", {})
                    if entry_configs:
                        logger.info(f"Loaded entry_configs from checkpoint: {list(entry_configs.keys())}")

            # Create initial model state
            metadata = {
                "checkpoint_path": str(checkpoint_path),
                "model_settings": model_settings,
                "shape_spec": shape_spec,
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "entry_configs": entry_configs  # Add entry configs for transform type detection
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

    def _detect_checkpoint_dtype(self, state_dict: Dict[str, Any]) -> Optional[torch.dtype]:
        """Detect the dtype of tensors in the checkpoint state dict.

        Strategy: Use the HIGHEST precision dtype found. If any weight is float64,
        load the entire model as float64 to preserve precision.

        Args:
            state_dict: State dictionary from checkpoint

        Returns:
            The highest precision dtype in the state dict, or None if no tensors found
        """
        dtypes_found = self._find_tensor_dtypes(state_dict)

        # Guard: No floating point tensors found
        if not dtypes_found:
            return None

        # Use highest precision: float64 > float32 > float16 > bfloat16
        highest_precision_dtype = self._select_highest_precision_dtype(dtypes_found)
        logger.info(f"Detected checkpoint dtype: {highest_precision_dtype} (found: {dtypes_found})")
        return highest_precision_dtype

    def _select_highest_precision_dtype(self, dtypes: set[torch.dtype]) -> torch.dtype:
        """Select the highest precision dtype from a set of dtypes.

        Precision hierarchy: float64 > float32 > float16 > bfloat16

        Args:
            dtypes: Set of dtypes to choose from

        Returns:
            The highest precision dtype
        """
        # Define precision order (highest to lowest)
        precision_order = [
            torch.float64,
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ]

        # Return first dtype from precision order that exists in the set
        for dtype in precision_order:
            if dtype in dtypes:
                return dtype

        # Fallback: return any dtype from the set
        return next(iter(dtypes))

    def _find_tensor_dtypes(self, state_dict: Dict[str, Any]) -> set[torch.dtype]:
        """Find all unique dtypes present in state dict tensors.

        Args:
            state_dict: State dictionary from checkpoint

        Returns:
            Set of unique dtypes found in floating-point tensors
        """
        dtypes_found: set[torch.dtype] = set()

        for value in state_dict.values():
            # Guard: Skip non-tensor values
            if not torch.is_tensor(value):
                continue

            # Guard: Skip non-floating-point tensors
            if not value.is_floating_point():
                continue

            dtypes_found.add(value.dtype)

        return dtypes_found

    def _extract_state_dict(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Extract state dict from checkpoint with automatic prefix stripping."""
        return extract_state_dict(checkpoint)


class DirectInferenceExecutor:
    """Direct inference executor using PyTorch models.

    Executes inference directly on loaded models with batch processing.
    Includes defensive dtype validation to prevent dtype mismatch errors.
    """

    def __init__(self):
        """Initialize executor with precision service."""
        self._precision_service = get_precision_service()

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
        """Run inference with batch processing and transform application.

        Transform Application Flow:
        1. If apply_transforms=True and fitted_transforms exist:
           - Apply forward transforms to inputs (normalize/scale)
           - Run model forward pass on transformed inputs
           - Apply inverse transforms to outputs (denormalize/descale)
        2. If apply_transforms=False or no fitted_transforms:
           - Run model forward pass directly on raw inputs
        """
        model = model_state.model
        inputs = request.inputs
        batch_size = request.batch_size

        target_device = self._resolve_target_device(model_state, request)

        if target_device != model_state.device:
            # Align model with requested device if it has changed post-preparation
            model.to(target_device)
            model_state = model_state.with_device(target_device)

        # DEFENSIVE DTYPE VALIDATION: Ensure inputs match model dtype before device placement
        inputs = self._validate_and_cast_inputs(inputs, model, model_state)

        inputs = self._move_inputs_to_device(inputs, target_device)

        # TRANSFORM APPLICATION: Apply forward transforms to inputs if requested
        if request.apply_transforms and model_state.feature_transforms:
            inputs = self._apply_feature_transforms(
                inputs,
                model_state.feature_transforms,
                target_device
            )
            logger.debug("Applied feature transforms to inputs")

        # Log input dtype for debugging
        if torch.is_tensor(inputs):
            logger.debug(f"Input dtype before model forward: {inputs.dtype}")
        elif isinstance(inputs, dict):
            dtypes = {k: v.dtype for k, v in inputs.items() if torch.is_tensor(v)}
            logger.debug(f"Input dtypes before model forward: {dtypes}")

        # Run model inference
        if hasattr(model, 'predict_step'):
            # Use Lightning predict_step if available
            predictions = model.predict_step(inputs, 0)
            if isinstance(predictions, dict) and "predictions" in predictions:
                raw_predictions = predictions["predictions"]
            else:
                raw_predictions = predictions
        else:
            # Direct model forward pass
            if isinstance(inputs, dict):
                # Handle dict inputs - use first tensor for simple case
                input_tensor = next(iter(inputs.values()))
                raw_predictions = model(input_tensor)
            else:
                # Handle single tensor input
                raw_predictions = model(inputs)

        # TRANSFORM APPLICATION: Apply inverse transforms to outputs if requested
        if request.apply_transforms and model_state.target_transforms:
            predictions = self._apply_inverse_transforms(
                raw_predictions,
                model_state.target_transforms,
                target_device
            )
            logger.debug("Applied inverse transforms to predictions")
        else:
            predictions = raw_predictions

        return {"predictions": predictions}

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

    def _get_model_dtype(self, model: torch.nn.Module) -> Optional[torch.dtype]:
        """Get the dtype of model parameters.

        Args:
            model: PyTorch model

        Returns:
            Model parameter dtype or None if no parameters found
        """
        try:
            first_param = next(model.parameters(), None)
            return first_param.dtype if first_param is not None else None
        except (StopIteration, AttributeError):
            return None

    def _validate_and_cast_inputs(
        self,
        inputs: Any,
        model: torch.nn.Module,
        model_state: ModelState
    ) -> Any:
        """Validate input dtypes match model and auto-cast if needed.

        This defensive layer prevents dtype mismatch errors by:
        1. Checking if input dtypes match model parameter dtypes
        2. Auto-casting with warning if mismatch detected
        3. Logging precision inference information

        Args:
            inputs: Input tensors (dict, tensor, or nested structure)
            model: PyTorch model
            model_state: Model state with metadata

        Returns:
            Inputs with validated/corrected dtypes
        """
        # Get model's expected dtype
        model_dtype = self._get_model_dtype(model)
        if model_dtype is None:
            # Model has no parameters, no validation needed
            return inputs

        # Validate and cast inputs recursively
        return self._cast_inputs_recursive(inputs, model_dtype)

    def _cast_inputs_recursive(
        self,
        inputs: Any,
        target_dtype: torch.dtype
    ) -> Any:
        """Recursively cast inputs to target dtype with warnings.

        Args:
            inputs: Input structure (dict, tensor, list, tuple, etc.)
            target_dtype: Target dtype to cast to

        Returns:
            Inputs cast to target dtype
        """
        if torch.is_tensor(inputs):
            if inputs.dtype != target_dtype:
                logger.warning(
                    f"Input dtype mismatch detected: input has {inputs.dtype}, "
                    f"model expects {target_dtype}. Auto-casting input to match model."
                )
                return inputs.to(dtype=target_dtype)
            return inputs

        if isinstance(inputs, dict):
            return {
                key: self._cast_inputs_recursive(value, target_dtype)
                for key, value in inputs.items()
            }

        if isinstance(inputs, (list, tuple)):
            casted = [self._cast_inputs_recursive(item, target_dtype) for item in inputs]
            return type(inputs)(casted)

        # Non-tensor input, pass through
        return inputs

    def _apply_feature_transforms(
        self,
        inputs: Dict[str, Any],
        fitted_transforms: Dict[str, Any],
        device: str
    ) -> Dict[str, Any]:
        """Apply forward transforms to feature inputs.

        Args:
            inputs: Dictionary of input tensors (e.g., {"x": tensor})
            fitted_transforms: Dictionary of fitted transform chains
            device: Target device for transforms

        Returns:
            Dictionary with transformed inputs
        """
        transformed_inputs = {}

        for entry_name, input_tensor in inputs.items():
            # Guard: No transform for this entry, pass through unchanged
            if entry_name not in fitted_transforms:
                transformed_inputs[entry_name] = input_tensor
                logger.debug(f"No transform found for '{entry_name}', passing through")
                continue

            # Apply transform
            transform_chain = fitted_transforms[entry_name].to(device)
            transformed_inputs[entry_name] = transform_chain(input_tensor)
            logger.debug(f"Applied forward transform to '{entry_name}'")

        return transformed_inputs

    def _apply_inverse_transforms(
        self,
        predictions: Any,
        target_transforms: Dict[str, Any],
        device: str
    ) -> Any:
        """Apply inverse transforms to model predictions.

        Args:
            predictions: Raw model predictions (tensor or dict)
            target_transforms: Dictionary of ONLY target transform chains
            device: Target device for transforms

        Returns:
            Inverse-transformed predictions in original space
        """
        if isinstance(predictions, dict):
            return self._apply_inverse_dict_transforms(predictions, target_transforms, device)

        return self._apply_inverse_tensor_transform(predictions, target_transforms, device)

    def _apply_inverse_dict_transforms(
        self,
        predictions: Dict[str, Any],
        target_transforms: Dict[str, Any],
        device: str
    ) -> Dict[str, Any]:
        """Apply inverse transforms to dictionary of predictions (SRP)."""
        inverse_predictions = {}

        for key, pred_tensor in predictions.items():
            # Guard: No transform for this prediction key
            if key not in target_transforms:
                inverse_predictions[key] = pred_tensor
                logger.debug(f"No transform found for prediction '{key}', passing through")
                continue

            # Apply inverse transform
            transform_chain = target_transforms[key].to(device)
            inverse_predictions[key] = transform_chain.inverse_transform(pred_tensor)
            logger.debug(f"Applied inverse transform to prediction '{key}'")

        return inverse_predictions

    def _apply_inverse_tensor_transform(
        self,
        predictions: torch.Tensor,
        target_transforms: Dict[str, Any],
        device: str
    ) -> torch.Tensor:
        """Apply inverse transform to single tensor prediction.

        Transforms are separated at the source (Feature vs Target) so this function
        only receives target transforms, eliminating ambiguity and complex filtering logic.

        Uses guard clauses for simplicity: handles trivial unambiguous cases first,
        fails fast on ambiguity.

        Args:
            predictions: Model predictions (single tensor)
            target_transforms: Dict of ONLY target transform chains
            device: Target device for transforms

        Returns:
            Inverse-transformed predictions

        Raises:
            TransformAmbiguityError: When multiple target transforms exist (ambiguous)
        """
        from dlkit.core.training.transforms.errors import TransformAmbiguityError

        # GUARD 1: No target transforms
        if not target_transforms:
            logger.debug("No target transforms, returning raw predictions")
            return predictions

        # GUARD 2: Single target transform (TRIVIAL UNAMBIGUOUS CASE!)
        if len(target_transforms) == 1:
            name = next(iter(target_transforms.keys()))
            logger.info(f"Applying inverse transform '{name}' (single target)")
            transform_chain = target_transforms[name].to(device)
            return transform_chain.inverse_transform(predictions)

        # GUARD 3: Multiple target transforms (AMBIGUOUS - FAIL FAST!)
        raise TransformAmbiguityError(
            list(target_transforms.keys()),
            context="Model returned single tensor but multiple target transforms exist. "
                   "Model must return dict with keys matching target names"
        )
