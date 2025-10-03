"""Use cases for inference workflows.

Pure business logic for inference operations,
following Single Responsibility Principle with focused use cases.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

from dlkit.interfaces.api.domain.errors import WorkflowError
from dlkit.interfaces.api.domain.models import InferenceResult

from ..domain.ports import (
    IModelLoader,
    IShapeInferenceChain,
    IInferenceExecutor,
    IModelStateManager,
    ICheckpointReconstructor
)
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


class ModelReconstructionUseCase:
    """Use case for reconstructing models from checkpoint metadata.

    Handles the complex process of model reconstruction with proper validation.
    """

    def __init__(
        self,
        checkpoint_reconstructor: ICheckpointReconstructor,
        shape_inference_use_case: ShapeInferenceUseCase,
        model_loader: IModelLoader,
        model_state_manager: IModelStateManager
    ):
        """Initialize with required dependencies."""
        self._checkpoint_reconstructor = checkpoint_reconstructor
        self._shape_inference_use_case = shape_inference_use_case
        self._model_loader = model_loader
        self._model_state_manager = model_state_manager

    def reconstruct_model_from_checkpoint(
        self,
        checkpoint_path: Path,
        device: str = "auto",
        dataset: Optional[Any] = None
    ) -> ModelState:
        """Reconstruct model from checkpoint with full validation.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Target device for model
            dataset: Optional dataset for shape inference

        Returns:
            ModelState: Fully reconstructed and validated model state

        Raises:
            WorkflowError: If reconstruction fails at any step
        """
        try:
            # Step 1: Extract checkpoint metadata
            checkpoint_metadata = self._checkpoint_reconstructor.extract_checkpoint_metadata(
                checkpoint_path
            )

            # Step 2: Reconstruct model settings
            model_settings = self._checkpoint_reconstructor.reconstruct_model_settings(
                checkpoint_path
            )

            if model_settings is None:
                raise WorkflowError(
                    "Could not reconstruct model settings from checkpoint metadata. "
                    "This checkpoint may be incompatible with automatic inference.",
                    {"use_case": "ModelReconstructionUseCase", "step": "model_settings_reconstruction"}
                )

            # Step 3: Create inference context
            context = InferenceContext(
                checkpoint_path=checkpoint_path,
                model_settings=model_settings,
                dataset=dataset,
                checkpoint_metadata=checkpoint_metadata
            )

            # Step 4: Infer shape specification
            shape_spec = self._shape_inference_use_case.infer_shape(context)

            # Step 5: Load model using the loader
            model_state = self._model_loader.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                model_settings=model_settings,
                shape_spec=shape_spec
            )

            # Step 6: Prepare for inference (including device placement)
            inference_ready_state = self._model_loader.prepare_for_inference(
                model_state, device
            )

            return inference_ready_state

        except WorkflowError:
            raise
        except Exception as e:
            raise WorkflowError(
                f"Model reconstruction failed: {str(e)}",
                {"use_case": "ModelReconstructionUseCase", "error": str(e)}
            ) from e


class InferenceUseCase:
    """Main use case for executing inference operations.

    Orchestrates the complete inference workflow from input processing to result generation.
    """

    def __init__(
        self,
        model_reconstruction_use_case: ModelReconstructionUseCase,
        inference_executor: IInferenceExecutor,
        model_state_manager: IModelStateManager
    ):
        """Initialize with required dependencies."""
        self._model_reconstruction_use_case = model_reconstruction_use_case
        self._inference_executor = inference_executor
        self._model_state_manager = model_state_manager

    def execute_inference(
        self,
        checkpoint_path: Path,
        inputs: InferenceInput,
        device: str = "auto",
        batch_size: int = 32,
        apply_transforms: bool = True,
        dataset: Optional[Any] = None
    ) -> InferenceResult:
        """Execute complete inference workflow.

        Args:
            checkpoint_path: Path to model checkpoint
            inputs: Input data for inference
            device: Device for inference execution
            batch_size: Batch size for processing
            apply_transforms: Whether to apply fitted transforms
            dataset: Optional dataset for shape inference

        Returns:
            InferenceResult: Complete inference result

        Raises:
            WorkflowError: If inference fails at any step
        """
        start_time = time.time()

        try:
            # Step 1: Reconstruct model from checkpoint
            model_state = self._model_reconstruction_use_case.reconstruct_model_from_checkpoint(
                checkpoint_path=checkpoint_path,
                device=device,
                dataset=dataset
            )

            # Step 2: Validate model state is inference ready
            if not self._model_state_manager.is_inference_ready(model_state):
                raise WorkflowError(
                    "Model state is not ready for inference execution",
                    {"use_case": "InferenceUseCase", "model_state": str(model_state)}
                )

            # Step 3: Create inference request
            # Convert inputs to tensor dictionary format
            if hasattr(inputs, 'to_tensor_dict'):
                input_data = inputs.to_tensor_dict()
            else:
                # Fallback for raw dictionary inputs
                if isinstance(inputs, dict):
                    input_data = inputs
                else:
                    raise WorkflowError(
                        "Input data must be convertible to dictionary format",
                        {"use_case": "InferenceUseCase", "input_type": type(inputs).__name__}
                    )

            request = InferenceRequest(
                inputs=input_data,
                batch_size=batch_size,
                apply_transforms=apply_transforms,
                device=device
            )

            # Step 4: Execute inference
            predictions = self._inference_executor.execute_inference(model_state, request)

            # Step 5: Create result
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
                {"use_case": "InferenceUseCase", "error": str(e)}
            ) from e