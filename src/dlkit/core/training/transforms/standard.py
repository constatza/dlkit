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

    def fit(self, data: torch.Tensor) -> None:
        """Compute mean and std along specified dimensions.

        Args:
            data: Input tensor to compute statistics from.
        """
        mean = torch.mean(data, dim=self.dim, keepdim=True)
        std = torch.std(data, dim=self.dim, keepdim=True)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
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
