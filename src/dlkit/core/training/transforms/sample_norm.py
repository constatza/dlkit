"""Per-sample normalization transform.

Normalizes each sample by its L2 norm, treating all features as a single vector.
This is useful when the magnitude of feature vectors matters less than their direction.
"""

from typing import TYPE_CHECKING

import torch
from pydantic import validate_call, ConfigDict

from dlkit.core.training.transforms.base import Transform
from dlkit.core.training.transforms.errors import TransformApplicationError
from dlkit.core.training.transforms.shape_inference import register_shape_inference

if TYPE_CHECKING:
    from dlkit.core.shape_specs import IShapeSpec


class SampleNormL2(Transform):
    """Normalize each sample by its L2 norm with mutable-attribute inverse pattern.

    This transform normalizes each sample (row) in the batch by its L2 norm,
    treating all features as a single vector. This makes each sample a unit vector.

    Key characteristics:
    - Per-sample operation: Each row is normalized independently
    - Preserves direction: Only scales magnitude to 1.0
    - Feature dimensions: Computes norm across all feature dimensions
    - Epsilon stabilization: Adds small constant to prevent division by zero
    - **Fully invertible**: Stores per-sample norms during forward() for inverse()

    Inverse Transform Pattern (Mutable Attribute):
    The transform stores per-sample norms in a mutable attribute (_last_norms)
    during forward(), enabling true inverse transformation. This is the standard
    PyTorch pattern (similar to BatchNorm) for runtime statistics.

    - forward() computes and stores norms for the current batch
    - inverse_transform() uses stored norms to denormalize
    - Fails fast if inverse_transform() called before forward() or with wrong batch size

    Example:
        >>> import torch
        >>> transform = SampleNormL2(eps=1e-8)
        >>> data = torch.randn(32, 10)
        >>>
        >>> # Forward: normalize and store norms
        >>> normalized = transform(data)
        >>> torch.norm(normalized[0], p=2).item()  # ≈ 1.0
        >>>
        >>> # Inverse: denormalize using stored norms
        >>> denormalized = transform.inverse_transform(normalized)
        >>> torch.allclose(denormalized, data, atol=1e-6)  # True
    """

    eps: float
    feature_dims: tuple[int, ...] | None
    _last_norms: torch.Tensor | None  # Mutable attribute for runtime statistics
    _shape_configured: bool

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        *,
        eps: float = 1e-8,
        feature_dims: tuple[int, ...] | None = None
    ) -> None:
        """Initialize the sample normalization transform.

        Args:
            eps: Small constant to prevent division by zero when computing norms.
                Defaults to 1e-8.
            feature_dims: Dimensions to compute norm across. If None, computes norm
                across all dimensions except the batch dimension (dim 0).
                This is inferred from data shape during forward() or from configure_shape().
                Example: For shape (B, H, W, C), feature_dims=(1, 2, 3) computes
                norm across spatial and channel dimensions for each sample.

        Example:
            >>> # Default: norm across all feature dims (inferred from data)
            >>> transform = SampleNormL2()
            >>>
            >>> # Explicit: norm across specific dimensions
            >>> transform = SampleNormL2(feature_dims=(1, 2, 3))
        """
        super().__init__()
        self.eps = eps
        self.feature_dims = feature_dims
        self._last_norms = None
        self._shape_configured = False

    def configure_shape(self, shape_spec: "IShapeSpec", entry_name: str) -> None:
        """Configure feature dimensions from shape information.

        Args:
            shape_spec: Shape specification containing entry shapes.
            entry_name: Name of the entry to get shape for.
        """
        shape = shape_spec.get_shape(entry_name)
        if shape is None:
            return

        # If feature_dims not explicitly set, default to all except batch (dim 0)
        if self.feature_dims is None:
            self.feature_dims = tuple(range(1, len(shape)))

        self._shape_configured = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize each sample by its L2 norm and store norms for inverse.

        This method computes the L2 norm for each sample, stores it in _last_norms,
        and returns the normalized tensor. The stored norms enable true inverse
        transformation.

        If feature_dims hasn't been configured, it's inferred from the data shape
        (all dimensions except batch dimension 0).

        Args:
            x: Input tensor with shape (batch_size, ...).

        Returns:
            Normalized tensor where each sample has L2 norm ≈ 1.0.
            Shape matches input shape.

        Side Effects:
            Stores per-sample norms in self._last_norms for inverse_transform().

        Example:
            >>> transform = SampleNormL2()
            >>> data = torch.randn(32, 10)
            >>> normalized = transform(data)
            >>> # Now _last_norms contains the norms for each of the 32 samples
        """
        # Lazy configuration from data shape if not already configured
        if self.feature_dims is None:
            self.feature_dims = tuple(range(1, len(x.shape)))

        # Compute L2 norm for each sample across feature dimensions
        # keepdim=True ensures broadcasting works correctly
        norms = torch.norm(x, p=2, dim=self.feature_dims, keepdim=True)

        # Store norms for inverse transformation (detach to avoid backprop issues)
        self._last_norms = norms.detach()

        # Normalize: x_normalized = x / ||x||_2
        # Add eps to prevent division by zero
        return x / (norms + self.eps)

    def inverse_transform(self, y: torch.Tensor) -> torch.Tensor:
        """Denormalize using norms stored from forward pass (fail-fast).

        This method multiplies the normalized tensor by the stored norms to
        recover the original magnitudes. It fails fast if:
        1. forward() has not been called yet (no norms stored)
        2. Batch size mismatch between forward() and inverse_transform()

        Args:
            y: Normalized tensor with shape (batch_size, ...).

        Returns:
            Denormalized tensor with original magnitudes restored.

        Raises:
            RuntimeError: If forward() has not been called yet.
            RuntimeError: If batch size mismatch between forward/inverse calls.

        Example:
            >>> transform = SampleNormL2(input_shape=(32, 10))
            >>> data = torch.randn(32, 10)
            >>> normalized = transform(data)
            >>> denormalized = transform.inverse_transform(normalized)
            >>> torch.allclose(denormalized, data, atol=1e-6)  # True
        """
        # Fail-fast: Check if forward() was called
        if self._last_norms is None:
            raise RuntimeError(
                "SampleNormL2.inverse_transform() called before forward(). "
                "You must call forward() first to compute and store per-sample norms. "
                "\n\nExample correct usage:"
                "\n  normalized = transform(data)  # forward() stores norms"
                "\n  denormalized = transform.inverse_transform(normalized)"
            )

        # Fail-fast: Check batch size matches
        if y.shape[0] != self._last_norms.shape[0]:
            raise RuntimeError(
                f"Batch size mismatch in inverse_transform(): "
                f"forward() was called with batch_size={self._last_norms.shape[0]}, "
                f"but inverse_transform() received batch_size={y.shape[0]}. "
                f"Ensure forward() and inverse_transform() are called on the same batch."
            )

        # Denormalize: x_original = y * ||x||_2
        return y * self._last_norms


# Register shape inference function (SampleNormL2 preserves shape)
@register_shape_inference(SampleNormL2)
def _infer_sample_norm_output_shape(input_shape: tuple[int, ...], **kwargs) -> tuple[int, ...]:
    """SampleNormL2 preserves input shape."""
    return input_shape
