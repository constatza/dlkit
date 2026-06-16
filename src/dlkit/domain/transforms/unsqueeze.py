import torch

from dlkit.domain.transforms.base import Transform


class Unsqueeze(Transform):
    """Insert a size-1 dimension at a given position (shape-agnostic transform).

    Useful for adapting flat `(N, D)` tensors to sequence-like `(N, 1, D)` inputs
    expected by models such as DeepONet.  The inverse simply squeezes the same dim.

    Example:
        >>> t = Unsqueeze(dim=1)
        >>> x = torch.randn(32, 64)
        >>> t(x).shape
        torch.Size([32, 1, 64])
        >>> t.inverse_transform(t(x)).shape
        torch.Size([32, 64])
    """

    dim: int

    def __init__(self, *, dim: int) -> None:
        """Initialize Unsqueeze transform.

        Args:
            dim: Position at which to insert the new size-1 dimension.
                Negative values are counted from the end of the *output* shape,
                following PyTorch's ``unsqueeze`` convention.

        Example:
            >>> Unsqueeze(dim=1)
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Insert a size-1 dimension at self.dim.

        Args:
            x: Input tensor of any shape.

        Returns:
            Tensor with one extra dimension of size 1 at position self.dim.
        """
        return x.unsqueeze(self.dim)

    def inverse_transform(self, y: torch.Tensor) -> torch.Tensor:
        """Remove the size-1 dimension inserted by forward.

        Args:
            y: Tensor produced by forward (must have a size-1 dim at self.dim).

        Returns:
            Tensor with the inserted dimension removed.
        """
        return y.squeeze(self.dim)

    def infer_output_shape(self, in_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Compute output shape after unsqueezing.

        Args:
            in_shape: Input tensor shape (without batch dimension assumed or not).

        Returns:
            Output shape with a size-1 dimension inserted at the normalised position.
        """
        ndim_out = len(in_shape) + 1
        pos = self.dim % ndim_out
        out = list(in_shape)
        out.insert(pos, 1)
        return tuple(out)
