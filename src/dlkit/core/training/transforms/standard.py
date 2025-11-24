from typing import TYPE_CHECKING

import torch

from dlkit.core.training.transforms.base import Transform
from dlkit.core.training.transforms.interfaces import (
    IFittableTransform,
    IInvertibleTransform,
    IShapeAwareTransform,
)
from dlkit.core.training.transforms.errors import TransformNotFittedError
from dlkit.core.training.transforms.shape_inference import register_shape_inference

if TYPE_CHECKING:
    from dlkit.core.shape_specs import IShapeSpec


class StandardScaler(Transform, IFittableTransform, IInvertibleTransform, IShapeAwareTransform):
    """Standard scaler that normalizes data to zero mean and unit variance.

    This transform computes mean and standard deviation along specified dimensions
    and uses them to standardize the data. It supports both eager buffer allocation
    (via configure_shape()) and lazy allocation (during fit()).
    """

    mean: torch.Tensor
    std: torch.Tensor
    dim: int | list[int]
    _shape_configured: bool

    def __init__(self, dim: int | list[int] | None = None) -> None:
        """Initialize StandardScaler.

        Args:
            dim: The dimension(s) along which to compute mean and std.
                Defaults to 0 (batch dimension).

        Example:
            >>> scaler = StandardScaler(dim=0)
            >>> scaler.fit(train_data)
            >>> normalized = scaler(train_data)
        """
        super().__init__()
        self.dim = dim if dim is not None else 0
        self._shape_configured = False

    def configure_shape(self, shape_spec: "IShapeSpec", entry_name: str) -> None:
        """Configure scaler with shape information for buffer pre-allocation.

        Args:
            shape_spec: Shape specification containing entry shapes.
            entry_name: Name of the entry to get shape for.
        """
        shape = shape_spec.get_shape(entry_name)
        if shape is None:
            return

        # Pre-allocate mean/std buffers
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("std", torch.ones(shape))
        self._shape_configured = True

    def fit(self, data: torch.Tensor) -> None:
        """Compute mean and std along specified dimensions.

        If buffers haven't been pre-allocated via configure_shape(), they are
        allocated lazily from the data shape.

        Args:
            data: Input tensor to compute statistics from.
        """
        # Lazy buffer allocation if not configured
        if not self._shape_configured:
            self.register_buffer("mean", torch.zeros(data.shape, device=data.device))
            self.register_buffer("std", torch.ones(data.shape, device=data.device))
            self._shape_configured = True

        self.mean = torch.mean(data, dim=self.dim, keepdim=True)
        self.std = torch.std(data, dim=self.dim, keepdim=True)
        self.fitted = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standardize tensor to zero mean and unit variance.

        Args:
            x: Input tensor to standardize.

        Returns:
            Standardized tensor.

        Raises:
            TransformNotFittedError: If fit() hasn't been called yet.
        """
        if not self.fitted:
            raise TransformNotFittedError("StandardScaler")
        return (x - self.mean) / (self.std + 1e-8)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse standardization back to original scale.

        Args:
            x: Standardized tensor.

        Returns:
            Tensor in original scale.

        Raises:
            TransformNotFittedError: If fit() hasn't been called yet.
        """
        if not self.fitted:
            raise TransformNotFittedError("StandardScaler")
        return (x * self.std) + self.mean


# Register shape inference function (StandardScaler preserves shape)
@register_shape_inference(StandardScaler)
def _infer_standard_output_shape(input_shape: tuple[int, ...], **kwargs) -> tuple[int, ...]:
    """StandardScaler preserves input shape."""
    return input_shape
