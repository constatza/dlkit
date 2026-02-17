from abc import ABC, abstractmethod
from typing import Any
from torch import nn
import torch

from dlkit.core.shape_specs import IShapeSpec


class DLKitModel(nn.Module):
    """Minimal base for DLKit models — provides only the dtype property.

    Replaces ShapeAwareModel/ShapeAgnosticModel with a simple nn.Module subclass.
    Models are plain PyTorch modules with PyTorch-standard constructor args
    (in_features/out_features for linear, in_channels/in_length for conv).
    """

    @property
    def dtype(self) -> torch.dtype:
        """Infer dtype from first parameter (Lightning pattern).

        Returns:
            The dtype of the model's first parameter.

        Raises:
            RuntimeError: If the model has no parameters.
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            raise RuntimeError(f"{self.__class__.__name__} has no parameters, cannot determine dtype")


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

    def __init__(self, *, unified_shape: IShapeSpec, **kwargs):
        """Initialize ShapeAwareModel with required shape specification.

        Args:
            unified_shape: Required shape specification for model architecture
            **kwargs: Additional model-specific parameters
        """
        super().__init__()

        # Store shape
        self._unified_shape = unified_shape

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

    @property
    def dtype(self) -> torch.dtype:
        """Get model's current dtype from parameters.

        This property infers the dtype from the model's parameters, following
        PyTorch Lightning's pattern. The actual dtype is determined by Lightning's
        precision parameter during training.

        Returns:
            The dtype of the model's parameters

        Raises:
            RuntimeError: If model has no parameters
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            raise RuntimeError(f"{self.__class__.__name__} has no parameters, cannot determine dtype")

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

    def __init__(self, **kwargs):
        """Initialize ShapeAgnosticModel without shape requirements.

        Args:
            **kwargs: Model-specific parameters
        """
        super().__init__()

    @property
    def dtype(self) -> torch.dtype:
        """Get model's current dtype from parameters.

        This property infers the dtype from the model's parameters, following
        PyTorch Lightning's pattern. The actual dtype is determined by Lightning's
        precision parameter during training.

        Returns:
            The dtype of the model's parameters

        Raises:
            RuntimeError: If model has no parameters
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            raise RuntimeError(f"{self.__class__.__name__} has no parameters, cannot determine dtype")

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Forward pass through the model.

        Args:
            x: Input data - can be Tensor, GraphData, etc.

        Returns:
            Output from the model
        """
        ...
