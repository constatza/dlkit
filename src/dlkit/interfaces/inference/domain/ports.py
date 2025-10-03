"""Domain interfaces (ports) for inference system.

These interfaces define the contracts for the inference domain,
following Interface Segregation Principle with focused, cohesive interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from dlkit.core.shape_specs import ShapeSpec
from dlkit.tools.config.components.model_components import ModelComponentSettings
from .models import ModelState, InferenceRequest, InferenceContext


class IModelLoader(Protocol):
    """Interface for loading and managing PyTorch models.

    Separates model loading concerns from business logic,
    following Dependency Inversion Principle.
    """

    def load_from_checkpoint(
        self,
        checkpoint_path: Path,
        model_settings: ModelComponentSettings,
        shape_spec: ShapeSpec
    ) -> ModelState:
        """Load model from checkpoint with specified configuration.

        Args:
            checkpoint_path: Path to checkpoint file
            model_settings: Model configuration settings
            shape_spec: Shape specification for model

        Returns:
            ModelState: Loaded model in managed state

        Raises:
            ModelLoadingError: If loading fails
        """
        ...

    def prepare_for_inference(self, model_state: ModelState, device: str) -> ModelState:
        """Prepare model for inference execution.

        Args:
            model_state: Current model state
            device: Target device for inference

        Returns:
            ModelState: Model prepared for inference
        """
        ...


class IShapeInferrer(Protocol):
    """Interface for inferring model shapes from various sources.

    Focused interface for shape inference strategies,
    following Interface Segregation Principle.
    """

    def infer_shape(self, context: InferenceContext) -> Optional[ShapeSpec]:
        """Infer shape specification from available context.

        Args:
            context: Inference context with available information

        Returns:
            ShapeSpec if successful, None if this strategy cannot infer
        """
        ...

    def can_infer(self, context: InferenceContext) -> bool:
        """Check if this inferrer can handle the given context.

        Args:
            context: Inference context to evaluate

        Returns:
            True if this inferrer can handle the context
        """
        ...


class IInferenceExecutor(Protocol):
    """Interface for executing inference on prepared models.

    Separates inference execution logic from orchestration.
    """

    def execute_inference(
        self,
        model_state: ModelState,
        request: InferenceRequest
    ) -> Dict[str, Any]:
        """Execute inference with the prepared model.

        Args:
            model_state: Prepared model state
            request: Inference request with inputs and parameters

        Returns:
            Dictionary of predictions

        Raises:
            InferenceExecutionError: If execution fails
        """
        ...


class IModelStateManager(Protocol):
    """Interface for managing model state transitions.

    Handles model lifecycle and state consistency.
    """

    def create_state(self, model: Any, metadata: Dict[str, Any]) -> ModelState:
        """Create managed model state from raw model.

        Args:
            model: Raw PyTorch model
            metadata: Model metadata

        Returns:
            ModelState: Managed model state
        """
        ...

    def transition_to_inference(self, model_state: ModelState) -> ModelState:
        """Transition model state to inference mode.

        This is where eval() should be called exactly once.

        Args:
            model_state: Current model state

        Returns:
            ModelState: Model in inference mode
        """
        ...

    def is_inference_ready(self, model_state: ModelState) -> bool:
        """Check if model state is ready for inference.

        Args:
            model_state: Model state to check

        Returns:
            True if ready for inference
        """
        ...


class ICheckpointReconstructor(Protocol):
    """Interface for reconstructing models from checkpoint metadata.

    Handles the complex process of model reconstruction
    with proper error handling and validation.
    """

    def reconstruct_model_settings(
        self,
        checkpoint_path: Path
    ) -> Optional[ModelComponentSettings]:
        """Reconstruct model settings from checkpoint metadata.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            ModelComponentSettings if successful, None if unable to reconstruct
        """
        ...

    def extract_checkpoint_metadata(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Extract metadata from checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Dictionary of checkpoint metadata

        Raises:
            CheckpointError: If checkpoint is invalid or corrupt
        """
        ...


class IShapeInferenceChain(Protocol):
    """Interface for shape inference chain coordination.

    Orchestrates multiple shape inference strategies
    following Chain of Responsibility pattern.
    """

    def infer_shape_with_fallbacks(
        self,
        context: InferenceContext
    ) -> Optional[ShapeSpec]:
        """Execute shape inference chain with fallback strategies.

        Args:
            context: Inference context

        Returns:
            ShapeSpec from first successful strategy, None if all fail
        """
        ...

    def add_strategy(self, strategy: IShapeInferrer) -> None:
        """Add a new shape inference strategy to the chain.

        Args:
            strategy: Shape inference strategy to add
        """
        ...