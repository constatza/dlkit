"""Dependency Injection container for inference system.

Provides a centralized way to wire up dependencies following
Dependency Inversion Principle with proper abstraction management.
"""

from __future__ import annotations

from typing import Optional

from .domain.ports import (
    IModelLoader,
    IShapeInferenceChain,
    IInferenceExecutor,
    IModelStateManager,
    ICheckpointReconstructor
)

from .application.use_cases import (
    InferenceUseCase,
    ModelReconstructionUseCase,
    ShapeInferenceUseCase
)

from .application.orchestrators import InferenceOrchestrator

from .infrastructure.adapters import (
    PyTorchModelLoader,
    CheckpointReconstructor,
    TorchModelStateManager,
    DirectInferenceExecutor
)

from dlkit.core.shape_specs import ShapeInferenceEngine, ShapeSystemFactory
from .domain.models import InferenceContext
from .domain.ports import IShapeInferenceChain


class ShapeInferenceEngineAdapter(IShapeInferenceChain):
    """Adapter to bridge ShapeInferenceEngine to IShapeInferenceChain interface."""

    def __init__(self, engine: ShapeInferenceEngine):
        self._engine = engine

    def infer_shape_with_fallbacks(self, context: InferenceContext):
        """Adapt modern engine to legacy interface."""
        try:
            # Try comprehensive inference with all available context
            return self._engine.infer_comprehensive(
                checkpoint_path=context.checkpoint_path,
                dataset=context.dataset,
                model_settings=context.model_settings,
                entry_configs=getattr(context, 'entry_configs', None)
            )
        except Exception:
            return None


class InferenceContainer:
    """Dependency injection container for inference components.

    Provides centralized dependency management with lazy initialization
    and proper separation of concerns.
    """

    def __init__(self):
        """Initialize container with None values for lazy loading."""
        self._model_state_manager: Optional[IModelStateManager] = None
        self._model_loader: Optional[IModelLoader] = None
        self._checkpoint_reconstructor: Optional[ICheckpointReconstructor] = None
        self._shape_inference_chain: Optional[IShapeInferenceChain] = None
        self._inference_executor: Optional[IInferenceExecutor] = None

        # Use cases
        self._shape_inference_use_case: Optional[ShapeInferenceUseCase] = None
        self._model_reconstruction_use_case: Optional[ModelReconstructionUseCase] = None
        self._inference_use_case: Optional[InferenceUseCase] = None

        # Orchestrators
        self._inference_orchestrator: Optional[InferenceOrchestrator] = None

    def get_model_state_manager(self) -> IModelStateManager:
        """Get or create model state manager."""
        if self._model_state_manager is None:
            self._model_state_manager = TorchModelStateManager()
        return self._model_state_manager

    def get_model_loader(self) -> IModelLoader:
        """Get or create model loader."""
        if self._model_loader is None:
            model_state_manager = self.get_model_state_manager()
            if not isinstance(model_state_manager, TorchModelStateManager):
                raise TypeError(
                    f"PyTorchModelLoader requires TorchModelStateManager, "
                    f"got {type(model_state_manager).__name__}"
                )
            self._model_loader = PyTorchModelLoader(
                model_state_manager=model_state_manager
            )
        return self._model_loader

    def get_checkpoint_reconstructor(self) -> ICheckpointReconstructor:
        """Get or create checkpoint reconstructor."""
        if self._checkpoint_reconstructor is None:
            self._checkpoint_reconstructor = CheckpointReconstructor()
        return self._checkpoint_reconstructor

    def get_shape_inference_chain(self) -> IShapeInferenceChain:
        """Get or create shape inference chain."""
        if self._shape_inference_chain is None:
            shape_factory = ShapeSystemFactory.create_production_system()
            engine = ShapeInferenceEngine(shape_factory=shape_factory)
            self._shape_inference_chain = ShapeInferenceEngineAdapter(engine)
        return self._shape_inference_chain

    def get_inference_executor(self) -> IInferenceExecutor:
        """Get or create inference executor."""
        if self._inference_executor is None:
            self._inference_executor = DirectInferenceExecutor()
        return self._inference_executor

    def get_shape_inference_use_case(self) -> ShapeInferenceUseCase:
        """Get or create shape inference use case."""
        if self._shape_inference_use_case is None:
            self._shape_inference_use_case = ShapeInferenceUseCase(
                shape_inference_chain=self.get_shape_inference_chain()
            )
        return self._shape_inference_use_case

    def get_model_reconstruction_use_case(self) -> ModelReconstructionUseCase:
        """Get or create model reconstruction use case."""
        if self._model_reconstruction_use_case is None:
            self._model_reconstruction_use_case = ModelReconstructionUseCase(
                checkpoint_reconstructor=self.get_checkpoint_reconstructor(),
                shape_inference_use_case=self.get_shape_inference_use_case(),
                model_loader=self.get_model_loader(),
                model_state_manager=self.get_model_state_manager()
            )
        return self._model_reconstruction_use_case

    def get_inference_use_case(self) -> InferenceUseCase:
        """Get or create inference use case."""
        if self._inference_use_case is None:
            self._inference_use_case = InferenceUseCase(
                model_reconstruction_use_case=self.get_model_reconstruction_use_case(),
                inference_executor=self.get_inference_executor(),
                model_state_manager=self.get_model_state_manager()
            )
        return self._inference_use_case

    def get_inference_orchestrator(self) -> InferenceOrchestrator:
        """Get or create inference orchestrator."""
        if self._inference_orchestrator is None:
            self._inference_orchestrator = InferenceOrchestrator(
                inference_use_case=self.get_inference_use_case()
            )
        return self._inference_orchestrator


# Global container instance
_container = InferenceContainer()


def get_inference_orchestrator() -> InferenceOrchestrator:
    """Get the main inference orchestrator.

    This is the primary entry point for the new architecture.
    """
    return _container.get_inference_orchestrator()


def reset_container() -> None:
    """Reset the container (useful for testing).

    Creates a new container instance to ensure clean state.
    """
    global _container
    _container = InferenceContainer()