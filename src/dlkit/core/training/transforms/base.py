from abc import abstractmethod

import torch
import torch.nn as nn
from torch_geometric.transforms import BaseTransform


class Transform(BaseTransform, nn.Module):
    """Base class for tensor transformations with SOLID-compliant design.

    This class provides the foundation for all transforms in DLKit. It integrates
    with PyTorch's nn.Module for device management and checkpoint persistence.

    Architecture:
    - Subclasses MUST implement forward()
    - Subclasses SHOULD implement inverse_transform() if invertible (inherit from IInvertibleTransform)
    - Subclasses SHOULD implement fit() if fittable (inherit from IFittableTransform)
    - Subclasses SHOULD implement configure_shape() if shape-aware (inherit from IShapeAwareTransform)
    - Fitted state stored as torch.Tensor buffer for checkpoint persistence

    Design Patterns:
    - Template Method: Provides fitted property, subclasses implement fit()
    - State Pattern: fitted flag tracked via tensor buffer
    - Dependency Inversion: Shape info provided via IShapeSpec (not stored here)

    Shape Handling Philosophy:
    - Transforms no longer store input_shape as this duplicates the shape_spec system
    - Shape-aware transforms implement IShapeAwareTransform and receive shapes via configure_shape()
    - Shape-agnostic transforms work with any compatible tensor without shape info
    - This eliminates redundant shape tracking and couples transforms to the mature shape_spec system

    Example:
        >>> class MyTransform(Transform, IInvertibleTransform):
        ...     def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...         return x * 2
        ...     def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        ...         return x / 2
    """

    apply_inverse: bool
    _fitted: torch.Tensor

    def __init__(self) -> None:
        """Initialize the transform.

        Note:
            Shape information is no longer passed to __init__(). Shape-aware transforms
            should implement IShapeAwareTransform and receive shapes via configure_shape(),
            or allocate buffers lazily during fit() from data.shape.
        """
        super().__init__()
        self.apply_inverse = True
        self.register_buffer("_fitted", torch.zeros(1, requires_grad=False))

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the transformation to the input tensor.

        Args:
            x: Input tensor to transform.

        Returns:
            Transformed tensor.
        """
        ...

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward() - applies the transformation.

        Args:
            x: Input tensor to transform.

        Returns:
            Transformed tensor.
        """
        return self.forward(x)

    @property
    def fitted(self) -> bool:
        """Whether the transform has been fitted to data.

        Returns:
            True if fit() has been called, False otherwise.
        """
        return self.get_buffer("_fitted").item() == 1

    @fitted.setter
    def fitted(self, value: bool) -> None:
        """Set the fitted state.

        Args:
            value: True to mark as fitted, False otherwise.
        """
        self._fitted.fill_(1 if value else 0)
