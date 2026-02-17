from collections.abc import Sequence

import torch
from pydantic import validate_call, ConfigDict

from dlkit.core.training.transforms.base import Transform
from dlkit.core.training.transforms.errors import TransformNotFittedError, ShapeMismatchError


class MinMaxScaler(Transform):
    """Minimum-Maximum Scaler that normalizes data to the range [-1, 1].

    This transform computes min and max statistics along specified dimensions
    and uses them to scale the data. It supports both eager buffer allocation
    (via configure_shape()) and lazy allocation (during fit()).

    The scaler accumulates global min/max when fit() is called multiple times,
    making it suitable for batch-wise fitting on large datasets.
    """

    min: torch.Tensor
    max: torch.Tensor
    dim: tuple[int, ...]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(self, *, dim: int | Sequence[int] = 0) -> None:
        """Initialize MinMaxScaler.

        Args:
            dim: The dimension(s) along which to compute min and max values.
                Defaults to 0 (batch dimension). Can be int or sequence of ints.

        Example:
            >>> # Create scaler for normalizing along batch dimension
            >>> scaler = MinMaxScaler(dim=0)
            >>>
            >>> # Fit directly (lazy allocation)
            >>> scaler.fit(train_data)
        """
        super().__init__()
        self.dim = dim if isinstance(dim, Sequence) else (dim,)

    def fit(self, data: torch.Tensor) -> None:
        """Compute (and accumulate) the min/max along specified dimensions.

        If buffers haven't been pre-allocated via configure_shape(), they are
        allocated lazily from the data shape. When called multiple times,
        accumulates global min/max values.

        Args:
            data: Input tensor to compute min/max statistics from.
                Shape varies but must be compatible with dim specification.

        Example:
            >>> scaler = MinMaxScaler(dim=0)
            >>> scaler.fit(batch1)  # Computes min/max
            >>> scaler.fit(batch2)  # Accumulates global min/max
        """
        # Guard clause: Ensure buffers allocated
        self._ensure_buffers_allocated(data)

        # Compute current batch statistics
        current_min = torch.amin(input=data, dim=self.dim, keepdim=True)
        current_max = torch.amax(input=data, dim=self.dim, keepdim=True)

        # Guard clause: First fit - initialize and return
        if not self.fitted:
            self.min = current_min
            self.max = current_max
            self.fitted = True
            return

        # Accumulate global min/max across multiple fit() calls
        self.min = torch.minimum(self.min, current_min)
        self.max = torch.maximum(self.max, current_max)

    def _ensure_buffers_allocated(self, data: torch.Tensor) -> None:
        """Allocate min/max buffers if not already allocated.

        Args:
            data: Input data to infer shape from.
        """
        # Guard: Early return if already allocated
        if hasattr(self, 'min') and self.min is not None:
            return

        # Normalize dim indices
        self.dim = tuple([idx % len(data.shape) for idx in self.dim])

        # Compute moments shape from data
        moments_shape = tuple([1 if i in self.dim else s for i, s in enumerate(data.shape)])
        self.register_buffer("min", torch.zeros(moments_shape, device=data.device))
        self.register_buffer("max", torch.ones(moments_shape, device=data.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale tensor to interval [-1, 1].

        Args:
            x: Input tensor to scale.

        Returns:
            Scaled tensor in range [-1, 1].

        Raises:
            TransformNotFittedError: If fit() hasn't been called yet.
        """
        if not self.fitted:
            raise TransformNotFittedError("MinMaxScaler")
        return 2 * (x - self.min) / (self.max - self.min + 1e-8) - 1

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse scale from [-1, 1] back to original range.

        Args:
            x: Scaled tensor in range [-1, 1].

        Returns:
            Tensor in original value range.

        Raises:
            TransformNotFittedError: If fit() hasn't been called yet.
        """
        if not self.fitted:
            raise TransformNotFittedError("MinMaxScaler")
        return (x + 1) / 2 * (self.max - self.min) + self.min

    def infer_output_shape(self, in_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Infer output shape. MinMaxScaler preserves input shape.

        Args:
            in_shape: Input tensor shape.

        Returns:
            Same as input shape.
        """
        return in_shape
