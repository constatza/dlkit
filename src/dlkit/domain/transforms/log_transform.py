from collections.abc import Sequence

import torch
from torch import Tensor

from dlkit.domain.transforms.base import PartialTransform


class LogTransform(PartialTransform):
    """Logarithmic transform: log(x + shift) with inverse exp(x) - shift.

    The shift parameter ensures the argument is strictly positive, extending
    the domain to (-shift, ∞).

    Args:
        shift: Additive constant before taking the log. Must be positive.
        indices: Feature indices to transform along index_dim. None applies to all.
        index_dim: Axis that holds the feature dimension. Defaults to -1.

    Example:
        >>> t = LogTransform(shift=1.0, indices=[0, 2], index_dim=-1)
        >>> y = t(x)
        >>> x_reconstructed = t.inverse_transform(y)
    """

    def __init__(
        self,
        *,
        shift: float = 1.0,
        indices: Sequence[int] | None = None,
        index_dim: int = -1,
    ) -> None:
        """Initialize LogTransform.

        Args:
            shift: Offset added before applying log. Must be > 0.
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
        return torch.log(x + self.shift)

    def _inverse_compute(self, x: Tensor) -> Tensor:
        return torch.exp(x) - self.shift
