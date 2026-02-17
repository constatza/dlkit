import torch

from dlkit.core.training.transforms.base import Transform
from dlkit.core.training.transforms.errors import TransformNotFittedError


class StandardScaler(Transform):
    """Standard scaler that normalizes data to zero mean and unit variance.

    This transform computes mean and standard deviation along specified dimensions
    and uses them to standardize the data. It supports both eager buffer allocation
    (via configure_shape()) and lazy allocation (during fit()).
    """

    mean: torch.Tensor
    std: torch.Tensor
    dim: int | list[int]

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

    def fit(self, data: torch.Tensor) -> None:
        """Compute mean and std along specified dimensions.

        If buffers haven't been pre-allocated via configure_shape(), they are
        allocated lazily from the data shape.

        Args:
            data: Input tensor to compute statistics from.
        """
        # Guard clause: Ensure buffers allocated
        self._ensure_buffers_allocated(data)

        self.mean = torch.mean(data, dim=self.dim, keepdim=True)
        self.std = torch.std(data, dim=self.dim, keepdim=True)
        self.fitted = True

    def _ensure_buffers_allocated(self, data: torch.Tensor) -> None:
        """Allocate mean/std buffers if not already allocated.

        Args:
            data: Input data to infer shape from.
        """
        # Guard: Early return if already allocated
        if hasattr(self, 'mean') and self.mean is not None:
            return

        self.register_buffer("mean", torch.zeros(data.shape, device=data.device))
        self.register_buffer("std", torch.ones(data.shape, device=data.device))

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

    def infer_output_shape(self, in_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Infer output shape. StandardScaler preserves input shape.

        Args:
            in_shape: Input tensor shape.

        Returns:
            Same as input shape.
        """
        return in_shape
