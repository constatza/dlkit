from abc import abstractmethod
from collections.abc import Sequence
from pydantic import validate_call, ConfigDict

import torch
import torch.nn as nn
from torch_geometric.transforms import BaseTransform
from loguru import logger


class Transform(BaseTransform, nn.Module):
    """Base class for tensor transformations with SOLID-compliant design.

    This class provides the foundation for all transforms in DLKit. It integrates
    with PyTorch's nn.Module for device management and checkpoint persistence.

    Architecture:
    - Subclasses MUST implement forward()
    - Subclasses SHOULD implement inverse_transform() if invertible (inherit from IInvertibleTransform)
    - Subclasses SHOULD implement fit() if fittable (inherit from IFittableTransform)
    - Fitted state stored as torch.Tensor buffer for checkpoint persistence

    Design Patterns:
    - Template Method: Provides fitted property, subclasses implement fit()
    - State Pattern: fitted flag tracked via tensor buffer

    Example:
        >>> class MyTransform(Transform, IInvertibleTransform):
        ...     def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...         return x * 2
        ...     def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        ...         return x / 2
    """

    apply_inverse: bool
    _fitted: torch.Tensor
    input_shape: torch.Tensor

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(self, input_shape: Sequence[int] | torch.Size | torch.Tensor) -> None:
        """Initialize the transform.

        Args:
            input_shape: The shape of the input tensor INCLUDING the batch dimension.
                Example: (32, 64) for batch_size=32, features=64
        """
        super().__init__()
        if not input_shape:
            logger.warning("No input shape provided. Assuming input shape (1,)")
        self.apply_inverse = True
        self.register_buffer("_fitted", torch.zeros(1, requires_grad=False))
        self.register_buffer("input_shape", torch.tensor(input_shape))

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
