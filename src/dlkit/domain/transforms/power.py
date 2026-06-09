import torch
from torch import Tensor

from dlkit.domain.transforms.base import PartialTransform


class PowerTransform(PartialTransform):
    """Power transform: x ** exponent with inverse x ** (1 / exponent).

    Useful for skew correction. Use exponent=0.5 for square-root compression
    of right-skewed non-negative data.

    Args:
        exponent: Power to raise x to. Must be non-zero.
        indices: Feature indices to transform along index_dim. None applies to all.
        index_dim: Axis that holds the feature dimension. Defaults to -1.

    Example:
        >>> t = PowerTransform(exponent=0.5)  # sqrt transform
        >>> y = t(x)
        >>> x_reconstructed = t.inverse_transform(y)
    """

    def __init__(
        self,
        *,
        exponent: float,
        indices=None,
        index_dim: int = -1,
    ) -> None:
        """Initialize PowerTransform.

        Args:
            exponent: Exponent for the power transform. Must be non-zero.
            indices: Feature indices to transform. None means all.
            index_dim: Axis along which to select features.

        Raises:
            ValueError: If exponent is 0.
        """
        if exponent == 0:
            raise ValueError("exponent must be non-zero.")
        super().__init__(indices=indices, index_dim=index_dim)
        self.exponent = exponent

    def _compute(self, x: Tensor) -> Tensor:
        return torch.pow(x, self.exponent)

    def _inverse_compute(self, x: Tensor) -> Tensor:
        return torch.pow(x, 1.0 / self.exponent)
