"""Use cases for inference workflows - REFACTORED.

This module provides focused use cases following Single Responsibility Principle:
- ModelLoadingUseCase: Load model from checkpoint (expensive, call once)
- InferenceExecutionUseCase: Execute inference on loaded model (fast, call many times)
- ShapeInferenceUseCase: Infer shape specifications (unchanged)
"""

from __future__ import annotations

import time
import torch
from pathlib import Path
from typing import Any, Dict, Optional

from dlkit.interfaces.api.domain.errors import WorkflowError
from dlkit.interfaces.api.domain.models import InferenceResult
from dlkit.interfaces.inference.checkpoint_utils import extract_state_dict

from ..domain.ports import (
    IModelLoader,
    IShapeInferenceChain,
    IInferenceExecutor,
    IModelStateManager,
    ICheckpointReconstructor
)
from ..transforms.checkpoint_loader import CheckpointTransformLoader
from ..domain.models import (
    ModelState,
    InferenceRequest,
    InferenceContext,
    ModelStateType
)
from ..inputs.inference_input import InferenceInput
from dlkit.core.shape_specs import ShapeSpec
from dlkit.tools.config.components.model_components import ModelComponentSettings


class ShapeInferenceUseCase:
    """Use case for inferring shapes from available context.

    Orchestrates the shape inference process using the chain of responsibility pattern.
    """

    def __init__(self, shape_inference_chain: IShapeInferenceChain):
        """Initialize with shape inference chain."""
        self._shape_inference_chain = shape_inference_chain

    def infer_shape(self, context: InferenceContext) -> ShapeSpec:
        """Execute shape inference with fallback strategies.

        Args:
            context: Inference context with available information

        Returns:
            ShapeSpec: Inferred shape specification

        Raises:
            WorkflowError: If all inference strategies fail
        """
        shape_spec = self._shape_inference_chain.infer_shape_with_fallbacks(context)

        if shape_spec is None:
            raise WorkflowError(
                "Shape inference failed with all available strategies. "
                "Please ensure the checkpoint contains valid shape information or provide a dataset.",
                {"use_case": "ShapeInferenceUseCase", "context": str(context)}
            )

        return shape_spec


class ModelLoadingUseCase:
    """Use case for loading models from checkpoints.

    SINGLE RESPONSIBILITY: Load model from checkpoint and prepare for inference.
    Does NOT execute inference - that's handled by InferenceExecutionUseCase.

    This use case performs all expensive operations ONCE:
    - Load checkpoint from disk (ONCE, not 3 times!)
    - Parse metadata
    - Reconstruct model settings
    - Infer shape specification
    - Instantiate model
    - Load weights
    - Place on device
    - Set to eval mode

    Usage:
        >>> model_state = model_loading_use_case.load_model(checkpoint_path, device="cuda")
        >>> # Model now loaded and ready - reuse for many inferences!
        >>> result1 = inference_execution_use_case.execute_inference(model_state, request1)
        >>> result2 = inference_execution_use_case.execute_inference(model_state, request2)
    """

    def __init__(
        self,
        checkpoint_reconstructor: ICheckpointReconstructor,
        shape_inference_use_case: ShapeInferenceUseCase,
        model_loader: IModelLoader,
        model_state_manager: IModelStateManager,
        transform_loader: CheckpointTransformLoader
    ):
        """Initialize with required dependencies.

        Args:
            checkpoint_reconstructor: Reconstructs model settings from checkpoints
            shape_inference_use_case: Infers shape specifications
            model_loader: Loads PyTorch models from checkpoints
            model_state_manager: Manages model state transitions
            transform_loader: Loads fitted transform chains from checkpoints
        """
        self._checkpoint_reconstructor = checkpoint_reconstructor
        self._shape_inference_use_case = shape_inference_use_case
        self._model_loader = model_loader
        self._model_state_manager = model_state_manager
        self._transform_loader = transform_loader

    def load_model(
        self,
        checkpoint_path: Path,
        device: str = "auto",
        dataset: Optional[Any] = None
    ) -> ModelState:
        """Load model from checkpoint and prepare for inference (expensive - call ONCE).

        This method loads the checkpoint from disk ONCE and reuses the loaded data
        throughout the loading process. Previous implementation loaded the checkpoint
        3 separate times!

        Args:
            checkpoint_path: Path to checkpoint file
            device: Target device for inference
            dataset: Optional dataset for shape inference fallback

        Returns:
            ModelState: Model ready for inference (eval mode, on device)

        Raises:
            WorkflowError: If loading fails at any step
        """
        try:
            # STEP 1: Load checkpoint ONCE (previously this happened 3 times!)
            checkpoint_data = self._load_checkpoint_once(checkpoint_path)

            # STEP 2: Extract model settings from ALREADY-LOADED checkpoint data
            model_settings = self._extract_model_settings_from_checkpoint_data(checkpoint_data)

            if model_settings is None:
                raise WorkflowError(
                    "Could not reconstruct model settings from checkpoint metadata. "
                    "This checkpoint may be incompatible with automatic inference.",
                    {"use_case": "ModelLoadingUseCase", "step": "model_settings_extraction"}
                )

            # STEP 3: Create inference context with checkpoint data
            context = InferenceContext(
                checkpoint_path=checkpoint_path,
                model_settings=model_settings,
                dataset=dataset,
                checkpoint_metadata=checkpoint_data
            )

            # STEP 4: Infer shape specification
            shape_spec = self._shape_inference_use_case.infer_shape(context)

            # STEP 5: Load fitted transforms from checkpoint (separated by type)
            feature_transforms, target_transforms = self._load_transforms_from_checkpoint_data(checkpoint_data)

            # STEP 6: Load model using ALREADY-LOADED checkpoint data
            model_state = self._load_model_from_checkpoint_data(
                checkpoint_data=checkpoint_data,
                checkpoint_path=checkpoint_path,
                model_settings=model_settings,
                shape_spec=shape_spec,
                feature_transforms=feature_transforms,
                target_transforms=target_transforms
            )

            # STEP 7: Prepare for inference (device placement + eval mode)
            inference_ready_state = self._model_loader.prepare_for_inference(
                model_state, device
            )

            return inference_ready_state

        except WorkflowError:
            raise
        except Exception as e:
            raise WorkflowError(
                f"Model loading failed: {str(e)}",
                {"use_case": "ModelLoadingUseCase", "error": str(e)}
            ) from e

    def _load_checkpoint_once(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Load checkpoint from disk ONCE and return the data.

        This is the ONLY place where torch.load() should be called during model loading.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Dictionary containing checkpoint data

        Raises:
            WorkflowError: If checkpoint cannot be loaded
        """
        try:
            if not checkpoint_path.exists():
                raise WorkflowError(
                    f"Checkpoint not found: {checkpoint_path}",
                    {"use_case": "ModelLoadingUseCase", "operation": "load_checkpoint"}
                )

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            if not isinstance(checkpoint, dict):
                raise WorkflowError(
                    f"Invalid checkpoint format: expected dict, got {type(checkpoint)}",
                    {"use_case": "ModelLoadingUseCase", "operation": "load_checkpoint"}
                )

            return checkpoint

        except WorkflowError:
            raise
        except Exception as e:
            raise WorkflowError(
                f"Failed to load checkpoint: {str(e)}",
                {"use_case": "ModelLoadingUseCase", "operation": "load_checkpoint"}
            ) from e

    def _extract_model_settings_from_checkpoint_data(
        self,
        checkpoint_data: Dict[str, Any]
    ) -> Optional[ModelComponentSettings]:
        """Extract model settings from already-loaded checkpoint data.

        Args:
            checkpoint_data: Checkpoint data dict (already loaded)

        Returns:
            ModelComponentSettings if found, None otherwise
        """
        # Check for v2.0+ metadata format
        if 'dlkit_metadata' not in checkpoint_data:
            return None

        metadata = checkpoint_data['dlkit_metadata']

        # Validate version
        version = metadata.get('version')
        if version != '2.0':
            return None

        if 'model_settings' not in metadata:
            return None

        settings_data = metadata['model_settings']

        # Deserialize settings
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

        except Exception:
            return None

    def _load_transforms_from_checkpoint_data(
        self,
        checkpoint_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Load fitted transform chains from checkpoint data.

        Args:
            checkpoint_data: Already-loaded checkpoint dictionary

        Returns:
            Dictionary of fitted transforms or None if not present

        Note:
            Uses CheckpointTransformLoader's internal extraction logic directly
            on the checkpoint data to avoid redundant disk I/O.

            Returns tuple of (feature_transforms, target_transforms).
        """
        try:
            # Extract transforms using the loader's internal method (returns tuple)
            feature_transforms, target_transforms = self._transform_loader._extract_fitted_transforms(checkpoint_data)

            from loguru import logger
            feature_count = len(feature_transforms) if feature_transforms else 0
            target_count = len(target_transforms) if target_transforms else 0

            if feature_count > 0 or target_count > 0:
                logger.info(f"Loaded {feature_count} feature and {target_count} target transform chains from checkpoint")
                logger.debug(f"Feature transforms: {list(feature_transforms.keys()) if feature_transforms else []}")
                logger.debug(f"Target transforms: {list(target_transforms.keys()) if target_transforms else []}")
                return feature_transforms, target_transforms
            else:
                logger.debug("No fitted transforms found in checkpoint (expected for graph/timeseries models)")
                return None, None

        except Exception as e:
            from loguru import logger
            logger.warning(f"Failed to load transforms from checkpoint: {e}")
            return None, None

    def _load_model_from_checkpoint_data(
        self,
        checkpoint_data: Dict[str, Any],
        checkpoint_path: Path,
        model_settings: ModelComponentSettings,
        shape_spec: ShapeSpec,
        feature_transforms: Optional[Dict[str, Any]] = None,
        target_transforms: Optional[Dict[str, Any]] = None
    ) -> ModelState:
        """Load model using already-loaded checkpoint data (no redundant disk I/O).

        CHECKPOINT LOADING ARCHITECTURE:
        --------------------------------
        Lightning checkpoints contain wrapper state (trained model structure):
        - `state_dict['model.*']`: Actual model weights (nested under wrapper)
        - `state_dict['fitted_transforms.*']`: Transform state (ModuleDict)
        - `state_dict['optimizer_states.*']`: Optimizer state (not needed for inference)
        - `state_dict['lr_schedulers.*']`: Scheduler state (not needed for inference)

        For inference, we load into a BARE MODEL (not wrapped), so:
        1. We extract state_dict from checkpoint
        2. We detect checkpoint dtype to preserve precision
        3. We create bare model using factory (no wrapper)
        4. We convert model to checkpoint dtype BEFORE loading weights
        5. We load weights with strict=False (intentional structural mismatch)
        6. Missing keys are expected (bare model != wrapped model)
        7. Unexpected keys (transforms, optimizer) are expected and handled separately

        The fitted_transforms.* keys are extracted separately by CheckpointTransformLoader
        and used to reconstruct transform pipelines for inference.

        Args:
            checkpoint_data: Already-loaded checkpoint dictionary
            checkpoint_path: Path to checkpoint (for metadata only)
            model_settings: Model component settings
            shape_spec: Shape specification

        Returns:
            ModelState with loaded model

        Raises:
            WorkflowError: If model loading fails
        """
        try:
            from dlkit.tools.config.core.context import BuildContext
            from dlkit.tools.config.core.factories import FactoryProvider
            from loguru import logger

            logger.info(f"Loading model from checkpoint: {checkpoint_path}")

            # Extract state dict from ALREADY-LOADED checkpoint data FIRST
            state_dict = self._extract_state_dict(checkpoint_data)

            # Detect checkpoint dtype BEFORE creating model to preserve precision
            checkpoint_dtype = self._detect_checkpoint_dtype(state_dict)
            if checkpoint_dtype is not None:
                logger.info(f"Detected checkpoint dtype: {checkpoint_dtype}")
            else:
                logger.warning("Could not detect checkpoint dtype, using default float32")
                checkpoint_dtype = torch.float32

            # Create build context for model instantiation
            build_context = BuildContext(mode="inference")

            # Apply shape overrides for model creation
            if shape_spec and not shape_spec.is_empty():
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

            # Load weights - dtypes now match, no conversion needed
            # Note: strict=False is intentional - Lightning checkpoints contain wrapper structure
            # that doesn't match the bare model (e.g., fitted_transforms.*, optimizer_states.*)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            # Filter out expected mismatches:
            # - fitted_transforms.* keys are used separately by CheckpointTransformLoader
            # - optimizer_states.* and other Lightning metadata are not needed for inference
            # - model.* prefix indicates wrapper structure (should already be stripped)
            unexpected_keys_filtered = [
                k for k in unexpected_keys
                if not k.startswith(('fitted_transforms.', 'optimizer_states.', 'lr_schedulers.', '_metadata'))
            ]

            # Only log genuine unexpected keys (not the expected transform/optimizer state)
            if missing_keys:
                logger.debug(f"Missing keys when loading checkpoint (expected for bare model): {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            if unexpected_keys_filtered:
                logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys_filtered[:5]}{'...' if len(unexpected_keys_filtered) > 5 else ''}")
            elif unexpected_keys:
                logger.debug(f"Checkpoint contains expected metadata keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")

            # Create model state with separated feature/target transforms
            metadata = {
                "checkpoint_path": str(checkpoint_path),
                "model_settings": model_settings,
                "shape_spec": shape_spec,
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys
            }

            return self._model_state_manager.create_state(
                model=model,
                metadata=metadata,
                feature_transforms=feature_transforms,
                target_transforms=target_transforms
            )

        except Exception as e:
            raise WorkflowError(
                f"Failed to load model from checkpoint data: {str(e)}",
                {"use_case": "ModelLoadingUseCase", "operation": "load_model"}
            ) from e

    def _extract_state_dict(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Extract state dict from checkpoint with automatic prefix stripping."""
        return extract_state_dict(checkpoint)

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
        from loguru import logger
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


class InferenceExecutionUseCase:
    """Use case for executing inference on ALREADY-LOADED models.

    SINGLE RESPONSIBILITY: Execute inference on loaded ModelState.
    Does NOT load models - that's handled by ModelLoadingUseCase.

    This use case is designed to be FAST - it only performs the forward pass,
    no checkpoint loading, no model reconstruction. Call this hundreds of times
    on the same loaded model.

    Usage:
        >>> # Load model ONCE
        >>> model_state = model_loading_use_case.load_model(checkpoint_path)
        >>>
        >>> # Execute inference MANY times (fast!)
        >>> result1 = inference_execution_use_case.execute_inference(model_state, request1)
        >>> result2 = inference_execution_use_case.execute_inference(model_state, request2)
        >>> result3 = inference_execution_use_case.execute_inference(model_state, request3)
    """

    def __init__(
        self,
        inference_executor: IInferenceExecutor,
        model_state_manager: IModelStateManager
    ):
        """Initialize with required dependencies."""
        self._inference_executor = inference_executor
        self._model_state_manager = model_state_manager

    def execute_inference(
        self,
        model_state: ModelState,
        request: InferenceRequest
    ) -> InferenceResult:
        """Execute inference on already-loaded model (fast operation).

        This method is FAST - it only does the forward pass through the model.
        No checkpoint loading, no model reconstruction, no weight loading.

        Args:
            model_state: ALREADY LOADED model state (from ModelLoadingUseCase)
            request: Inference request with inputs and parameters

        Returns:
            InferenceResult with predictions

        Raises:
            WorkflowError: If model not ready or inference fails
        """
        start_time = time.time()

        try:
            # Validate model is ready for inference
            if not self._model_state_manager.is_inference_ready(model_state):
                raise WorkflowError(
                    "Model state is not ready for inference execution",
                    {"use_case": "InferenceExecutionUseCase", "model_state_type": model_state.state_type}
                )

            # Execute inference (just forward pass, no loading!)
            predictions = self._inference_executor.execute_inference(model_state, request)

            # Create result
            duration = time.time() - start_time
            return InferenceResult(
                model_state=None,  # Don't pass domain model state to API layer
                predictions=predictions,
                metrics=None,  # Could be extended to include metrics
                duration_seconds=duration
            )

        except WorkflowError:
            raise
        except Exception as e:
            raise WorkflowError(
                f"Inference execution failed: {str(e)}",
                {"use_case": "InferenceExecutionUseCase", "error": str(e)}
            ) from e
