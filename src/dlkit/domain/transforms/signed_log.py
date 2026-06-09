import torch
from torch import Tensor

from dlkit.domain.transforms.base import PartialTransform


class SignedLogTransform(PartialTransform):
    """Signed logarithmic transform: sign(x) * log(|x| + shift).

    Handles negative and zero values by preserving the sign, making it
    suitable for signed heavy-tailed data or spatial coordinates.
    Inverse: sign(x) * (exp(|x|) - shift).

    Args:
        shift: Additive constant inside the log. Must be positive.
        indices: Feature indices to transform along index_dim. None applies to all.
        index_dim: Axis that holds the feature dimension. Defaults to -1.

    Example:
        >>> t = SignedLogTransform(shift=1.0)
        >>> y = t(x)
        >>> x_reconstructed = t.inverse_transform(y)
    """

    def __init__(
        self,
        *,
        shift: float = 1.0,
        indices=None,
        index_dim: int = -1,
    ) -> None:
        """Initialize SignedLogTransform.

        Args:
            shift: Offset added to |x| before applying log. Must be > 0.
            indices: Feature indices to transform. None means all.
            index_dim: Axis along which to select features.

        Raises:
            ValueError: If shift <= 0.
        """
        if shift <= 0:
            raise ValueError(f"shift must be positive, got {shift}.")
        super().__init__(indices=indices, index_dim=index_dim)
        self.shift = shift

    def _compute(self, x: Tensor) -> Tensor:
        return torch.sign(x) * torch.log(torch.abs(x) + self.shift)

    def _inverse_compute(self, x: Tensor) -> Tensor:
        return torch.sign(x) * (torch.exp(torch.abs(x)) - self.shift)
