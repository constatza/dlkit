from collections.abc import Sequence
from typing import cast

import torch
from pydantic import ConfigDict, validate_call

from dlkit.domain.transforms.base import Transform
from dlkit.domain.transforms.errors import TransformNotFittedError


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
        dim_values = cast("Sequence[int]", dim) if isinstance(dim, Sequence) else (dim,)
        self.dim: tuple[int, ...] = tuple(int(index) for index in dim_values)
        # Register empty placeholder buffers so state_dict() always contains these keys
        # and load_state_dict() can fill them without pre-registration hacks
        self.register_buffer("min", torch.tensor([]))
        self.register_buffer("max", torch.tensor([]))
        self._fit_min: torch.Tensor | None = None
        self._fit_max: torch.Tensor | None = None

    def fit(self, data: torch.Tensor) -> None:
        """Compute min/max statistics from a single in-memory tensor.

        Args:
            data: Input tensor to compute min/max statistics from.
                Shape varies but must be compatible with dim specification.
        """
        self.reset_fit_state()
        self.update_fit(data)
        self.finalize_fit()

    def reset_fit_state(self) -> None:
        """Reset incremental fit accumulators."""
        self._fit_min = None
        self._fit_max = None
        self.fitted = False

    def update_fit(self, batch: torch.Tensor) -> None:
        """Accumulate min/max statistics from one batch."""
        # Normalize dim indices
        dim = tuple(index % len(batch.shape) for index in self.dim)
        self.dim = dim

        # Compute current batch statistics
        current_min = torch.amin(input=batch, dim=list(dim), keepdim=True)
        current_max = torch.amax(input=batch, dim=list(dim), keepdim=True)

        if self._fit_min is None or self._fit_max is None:
            self._fit_min = current_min
            self._fit_max = current_max
        else:
            self._fit_min = torch.minimum(self._fit_min, current_min)
            self._fit_max = torch.maximum(self._fit_max, current_max)

    def finalize_fit(self) -> None:
        """Finalize accumulated statistics into fitted buffers."""
        if self._fit_min is None or self._fit_max is None:
            raise ValueError("MinMaxScaler.finalize_fit() called before any update_fit() call.")
        self.min = self._fit_min
        self.max = self._fit_max
        self.fitted = True

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
