import torch
from torch import Tensor

from dlkit.domain.transforms.base import PartialTransform


class TanhTransform(PartialTransform):
    """Hyperbolic tangent transform: tanh(x) with inverse atanh(x).

    Provides a smooth bounded output in (-1, 1), useful for soft-clipping
    features with heavy tails or as a smooth normalisation without fitting.

    Args:
        indices: Feature indices to transform along index_dim. None applies to all.
        index_dim: Axis that holds the feature dimension. Defaults to -1.

    Example:
        >>> t = TanhTransform(indices=[0, 1])
        >>> y = t(x)
        >>> x_reconstructed = t.inverse_transform(y)
    """

    def __init__(
        self,
        *,
        indices=None,
        index_dim: int = -1,
    ) -> None:
        """Initialize TanhTransform.

        Args:
            indices: Feature indices to transform. None means all.
            index_dim: Axis along which to select features.
        """
        super().__init__(indices=indices, index_dim=index_dim)

    def _compute(self, x: Tensor) -> Tensor:
        return torch.tanh(x)

    def _inverse_compute(self, x: Tensor) -> Tensor:
        return torch.atanh(x)
