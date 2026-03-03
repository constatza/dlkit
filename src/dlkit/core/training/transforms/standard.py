import torch

from dlkit.core.training.transforms.base import Transform
from dlkit.core.training.transforms.errors import TransformNotFittedError


class StandardScaler(Transform):
    """Standard scaler that normalizes data to zero mean and unit variance.

    This transform computes mean and standard deviation along specified dimensions
    and uses them to standardize the data. It supports both eager buffer allocation
    (via configure_shape()) and lazy allocation (during fit()).
    """

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
        # Register empty placeholder buffers so state_dict() always contains these keys
        # and load_state_dict() can fill them without pre-registration hacks
        self.register_buffer("mean", torch.tensor([]))
        self.register_buffer("std", torch.tensor([]))
        self._fit_count: int = 0
        self._fit_mean: torch.Tensor | None = None
        self._fit_m2: torch.Tensor | None = None

    def fit(self, data: torch.Tensor) -> None:
        """Compute mean/std statistics from a single in-memory tensor.

        Args:
            data: Input tensor to compute statistics from.
        """
        self.reset_fit_state()
        self.update_fit(data)
        self.finalize_fit()

    def reset_fit_state(self) -> None:
        """Reset incremental fit accumulators."""
        self._fit_count = 0
        self._fit_mean = None
        self._fit_m2 = None
        self.fitted = False

    def update_fit(self, batch: torch.Tensor) -> None:
        """Accumulate running mean/variance statistics from one batch."""
        dim_raw = self.dim if isinstance(self.dim, (list, tuple)) else (self.dim,)
        dim = tuple(int(idx) % len(batch.shape) for idx in dim_raw)
        self.dim = dim

        batch_mean = torch.mean(batch, dim=dim, keepdim=True)
        batch_var = torch.var(batch, dim=dim, keepdim=True, unbiased=False)
        batch_count = 1
        for axis in dim:
            batch_count *= int(batch.shape[axis])

        if self._fit_mean is None or self._fit_m2 is None or self._fit_count == 0:
            self._fit_mean = batch_mean
            self._fit_m2 = batch_var * batch_count
            self._fit_count = batch_count
            return

        current_count = self._fit_count
        total_count = current_count + batch_count
        delta = batch_mean - self._fit_mean
        self._fit_mean = self._fit_mean + delta * (batch_count / total_count)
        self._fit_m2 = (
            self._fit_m2
            + (batch_var * batch_count)
            + delta.pow(2) * (current_count * batch_count / total_count)
        )
        self._fit_count = total_count

    def finalize_fit(self) -> None:
        """Finalize accumulated statistics into fitted buffers."""
        if self._fit_mean is None or self._fit_m2 is None or self._fit_count <= 0:
            raise ValueError("StandardScaler.finalize_fit() called before any update_fit() call.")

        variance = self._fit_m2 / float(self._fit_count)
        self.mean = self._fit_mean
        self.std = torch.sqrt(torch.clamp(variance, min=0.0))
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
        for name in ("mean", "std"):
            key = f"{prefix}{name}"
            if key in state_dict:
                self.register_buffer(name, torch.empty_like(state_dict[key]))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

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
