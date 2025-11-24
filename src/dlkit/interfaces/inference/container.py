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
    ModelLoadingUseCase,
    InferenceExecutionUseCase,
    ShapeInferenceUseCase
)

from .factory import PredictorFactory

from .infrastructure.adapters import (
    PyTorchModelLoader,
    CheckpointReconstructor,
    TorchModelStateManager,
    DirectInferenceExecutor
)

from .transforms.checkpoint_loader import CheckpointTransformLoader

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
        self._transform_loader: Optional[CheckpointTransformLoader] = None

        # Use cases (NEW ARCHITECTURE)
        self._shape_inference_use_case: Optional[ShapeInferenceUseCase] = None
        self._model_loading_use_case: Optional[ModelLoadingUseCase] = None
        self._inference_execution_use_case: Optional[InferenceExecutionUseCase] = None

        # Factory
        self._predictor_factory: Optional[PredictorFactory] = None

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

    def get_transform_loader(self) -> CheckpointTransformLoader:
        """Get or create transform loader."""
        if self._transform_loader is None:
            self._transform_loader = CheckpointTransformLoader()
        return self._transform_loader

    def get_shape_inference_use_case(self) -> ShapeInferenceUseCase:
        """Get or create shape inference use case."""
        if self._shape_inference_use_case is None:
            self._shape_inference_use_case = ShapeInferenceUseCase(
                shape_inference_chain=self.get_shape_inference_chain()
            )
        return self._shape_inference_use_case

    def get_model_loading_use_case(self) -> ModelLoadingUseCase:
        """Get or create model loading use case."""
        if self._model_loading_use_case is None:
            self._model_loading_use_case = ModelLoadingUseCase(
                checkpoint_reconstructor=self.get_checkpoint_reconstructor(),
                shape_inference_use_case=self.get_shape_inference_use_case(),
                model_loader=self.get_model_loader(),
                model_state_manager=self.get_model_state_manager(),
                transform_loader=self.get_transform_loader()
            )
        return self._model_loading_use_case

    def get_inference_execution_use_case(self) -> InferenceExecutionUseCase:
        """Get or create inference execution use case."""
        if self._inference_execution_use_case is None:
            self._inference_execution_use_case = InferenceExecutionUseCase(
                inference_executor=self.get_inference_executor(),
                model_state_manager=self.get_model_state_manager()
            )
        return self._inference_execution_use_case

    def get_predictor_factory(self) -> PredictorFactory:
        """Get or create predictor factory.

        This is the main entry point for the new inference architecture.
        """
        if self._predictor_factory is None:
            self._predictor_factory = PredictorFactory(
                model_loading_use_case=self.get_model_loading_use_case(),
                inference_execution_use_case=self.get_inference_execution_use_case()
            )
        return self._predictor_factory


# Global container instance
_container = InferenceContainer()


def get_predictor_factory() -> PredictorFactory:
    """Get the predictor factory (main entry point for new architecture).

    Returns:
        PredictorFactory for creating predictors
    """
    return _container.get_predictor_factory()


def reset_container() -> None:
    """Reset the container (useful for testing).

    Creates a new container instance to ensure clean state.
    """
    global _container
    _container = InferenceContainer()