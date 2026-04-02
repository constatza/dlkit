import torch

from dlkit.domain.transforms.base import Transform


class Permutation(Transform):
    """Permute tensor dimensions (shape-agnostic transform).

    This transform reorders the dimensions of a tensor according to a specified
    permutation. It's fully shape-agnostic - it works with any tensor shape that
    has enough dimensions for the permutation.

    Example:
        >>> # Swap last two dimensions: (B, H, W) → (B, W, H)
        >>> perm = Permutation(dims=(0, 2, 1))
        >>> data = torch.randn(32, 64, 128)
        >>> permuted = perm(data)  # Shape: (32, 128, 64)
        >>> restored = perm.inverse_transform(permuted)  # Back to (32, 64, 128)
    """

    dims: tuple[int, ...]
    _inverse_dims: tuple[int, ...]

    def __init__(self, *, dims: tuple[int, ...]) -> None:
        """Initialize permutation transform.

        Args:
            dims: Dimension permutation order. Must be a valid permutation
                (each index 0 to len(dims)-1 appears exactly once).

        Example:
            >>> # Swap last two dims
            >>> perm = Permutation(dims=(0, 2, 1))
        """
        super().__init__()
        self.dims = dims
        # Precompute inverse permutation for efficiency
        self._inverse_dims = self._compute_inverse_permutation(dims)

    @staticmethod
    def _compute_inverse_permutation(dims: tuple[int, ...]) -> tuple[int, ...]:
        """Compute the inverse permutation mapping.

        Args:
            dims: Forward permutation.

        Returns:
            Inverse permutation that undoes the forward permutation.
        """
        inverse = [0] * len(dims)
        for i, dim in enumerate(dims):
            inverse[dim] = i
        return tuple(inverse)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Permute tensor dimensions.

        Args:
            x: Input tensor.

        Returns:
            Tensor with permuted dimensions.
        """
        return x.permute(self.dims)

    def inverse_transform(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse permutation to restore original dimension order.

        Args:
            y: Permuted tensor.

        Returns:
            Tensor with original dimension order.
        """
        return y.permute(self._inverse_dims)

    def infer_output_shape(self, in_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Infer output shape. Permutation reorders dimensions according to dims.

        Args:
            in_shape: Input tensor shape.

        Returns:
            Output shape with reordered dimensions.
        """
        return tuple(in_shape[d] for d in self.dims)
