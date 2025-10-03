from abc import ABC, abstractmethod
from typing import Any
from torch import Tensor, nn
import torch

from dlkit.tools.config.precision import PrecisionStrategy
from dlkit.interfaces.api.services.precision_service import get_precision_service
from dlkit.core.shape_specs import IShapeSpec


# New ABC hierarchy for shape handling
class ShapeAwareModel(ABC, nn.Module):
    """Abstract base class for models that require unified shape information.

    Shape-aware models MUST receive shape specifications during initialization
    and use them for proper model architecture configuration. This includes
    most DLKit internal models (FFNN, Graph, CAE, etc.) that need to know
    input/output dimensions at construction time.

    Args:
        unified_shape: IShapeSpec containing required shape information
        precision: Optional precision strategy override
        **kwargs: Model-specific parameters

    Example:
        shape_spec = create_shape_spec({"x": (784,), "y": (10,)})
        model = MyShapeAwareModel(unified_shape=shape_spec, **kwargs)
    """

    def __init__(self, *, unified_shape: IShapeSpec, precision: PrecisionStrategy | None = None, **kwargs):
        """Initialize ShapeAwareModel with required shape specification.

        Args:
            unified_shape: Required shape specification for model architecture
            precision: Optional precision strategy override
            **kwargs: Additional model-specific parameters
        """
        super().__init__()

        # Store precision and shape
        self._precision_strategy = precision
        self._unified_shape = unified_shape
        self._precision_applied = False

        # Validate shape immediately
        if not self.accepts_shape(unified_shape):
            raise ValueError(f"{self.__class__.__name__} cannot accept the provided shape specification: {unified_shape}")

    @abstractmethod
    def accepts_shape(self, shape_spec: IShapeSpec) -> bool:
        """Validate if this model can accept the given shape specification.

        Args:
            shape_spec: Shape specification to validate

        Returns:
            True if shape is acceptable, False otherwise
        """
        ...

    def get_unified_shape(self) -> IShapeSpec:
        """Get the unified shape specification for this model.

        Returns:
            The shape specification used to configure this model
        """
        return self._unified_shape

    def _apply_precision(self) -> None:
        """Apply precision strategy to model weights."""
        if self._precision_applied:
            return

        precision_service = get_precision_service()

        provider = None
        if self._precision_strategy is not None:
            strategy = self._precision_strategy

            class ModelPrecisionProvider:
                def get_precision_strategy(self) -> PrecisionStrategy:
                    return strategy

            provider = ModelPrecisionProvider()

        target_dtype = precision_service.get_torch_dtype(provider)
        self.to(dtype=target_dtype)
        self._precision_applied = True

    def ensure_precision_applied(self) -> None:
        """Ensure precision has been applied to model weights."""
        self._apply_precision()

    def cast_input(self, x: Tensor) -> Tensor:
        """Cast input tensor to model's precision.

        Args:
            x: Input tensor to cast

        Returns:
            Tensor cast to model's precision dtype
        """
        precision_service = get_precision_service()
        provider = None
        if self._precision_strategy is not None:
            strategy = self._precision_strategy

            class ModelPrecisionProvider:
                def get_precision_strategy(self) -> PrecisionStrategy:
                    return strategy

            provider = ModelPrecisionProvider()

        target_dtype = precision_service.get_torch_dtype(provider)
        return x.to(dtype=target_dtype)

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Forward pass through the model.

        Args:
            x: Input data - can be Tensor, GraphData, etc.

        Returns:
            Output from the model
        """
        ...


class ShapeAgnosticModel(ABC, nn.Module):
    """Abstract base class for models that don't require shape information.

    Shape-agnostic models handle their own architecture configuration and
    don't need shape specifications at construction time. This includes
    external models like PyTorch Forecasting models that determine their
    structure from data or have pre-defined architectures.

    Args:
        precision: Optional precision strategy override
        **kwargs: Model-specific parameters

    Example:
        model = MyShapeAgnosticModel(**model_specific_kwargs)
    """

    def __init__(self, *, precision: PrecisionStrategy | None = None, **kwargs):
        """Initialize ShapeAgnosticModel without shape requirements.

        Args:
            precision: Optional precision strategy override
            **kwargs: Model-specific parameters
        """
        super().__init__()

        # Store precision
        self._precision_strategy = precision
        self._precision_applied = False

    def _apply_precision(self) -> None:
        """Apply precision strategy to model weights."""
        if self._precision_applied:
            return

        precision_service = get_precision_service()

        provider = None
        if self._precision_strategy is not None:
            strategy = self._precision_strategy

            class ModelPrecisionProvider:
                def get_precision_strategy(self) -> PrecisionStrategy:
                    return strategy

            provider = ModelPrecisionProvider()

        target_dtype = precision_service.get_torch_dtype(provider)
        self.to(dtype=target_dtype)
        self._precision_applied = True

    def ensure_precision_applied(self) -> None:
        """Ensure precision has been applied to model weights."""
        self._apply_precision()

    def cast_input(self, x: Tensor) -> Tensor:
        """Cast input tensor to model's precision.

        Args:
            x: Input tensor to cast

        Returns:
            Tensor cast to model's precision dtype
        """
        precision_service = get_precision_service()
        provider = None
        if self._precision_strategy is not None:
            strategy = self._precision_strategy

            class ModelPrecisionProvider:
                def get_precision_strategy(self) -> PrecisionStrategy:
                    return strategy

            provider = ModelPrecisionProvider()

        target_dtype = precision_service.get_torch_dtype(provider)
        return x.to(dtype=target_dtype)

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Forward pass through the model.

        Args:
            x: Input data - can be Tensor, GraphData, etc.

        Returns:
            Output from the model
        """
        ...


