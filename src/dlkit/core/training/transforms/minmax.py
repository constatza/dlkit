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
        # Register empty placeholder buffers so state_dict() always contains these keys
        # and load_state_dict() can fill them without pre-registration hacks
        self.register_buffer("min", torch.tensor([]))
        self.register_buffer("max", torch.tensor([]))

    def fit(self, data: torch.Tensor) -> None:
        """Compute (and accumulate) the min/max along specified dimensions.

        When called multiple times, accumulates global min/max values.

        Args:
            data: Input tensor to compute min/max statistics from.
                Shape varies but must be compatible with dim specification.

        Example:
            >>> scaler = MinMaxScaler(dim=0)
            >>> scaler.fit(batch1)  # Computes min/max
            >>> scaler.fit(batch2)  # Accumulates global min/max
        """
        # Normalize dim indices
        dim = tuple(idx % len(data.shape) for idx in self.dim)
        self.dim = dim

        # Compute current batch statistics
        current_min = torch.amin(input=data, dim=list(dim), keepdim=True)
        current_max = torch.amax(input=data, dim=list(dim), keepdim=True)

        # First fit or accumulate
        if not self.fitted:
            self.register_buffer("min", current_min)
            self.register_buffer("max", current_max)
            self.fitted = True
        else:
            self.register_buffer("min", torch.minimum(self.min, current_min))
            self.register_buffer("max", torch.maximum(self.max, current_max))

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list,
        unexpected_keys: list,
        error_msgs: list,
    ) -> None:
        """Pre-allocate buffers with correct shape from checkpoint before loading.

        Args:
            state_dict: Full state dictionary.
            prefix: Module prefix for this module's keys.
            local_metadata: Local metadata dict.
            strict: Whether to enforce strict key matching.
            missing_keys: List to accumulate missing key names.
            unexpected_keys: List to accumulate unexpected key names.
            error_msgs: List to accumulate error messages.
        """
        for name in ("min", "max"):
            key = f"{prefix}{name}"
            if key in state_dict:
                self.register_buffer(name, torch.empty_like(state_dict[key]))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

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
