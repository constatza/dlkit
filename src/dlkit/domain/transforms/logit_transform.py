import torch
from torch import Tensor

from dlkit.domain.transforms.base import PartialTransform


class LogitTransform(PartialTransform):
    """Logit transform: log(x / (1 - x)) with inverse sigmoid(x).

    Suitable for probability-like inputs in (0, 1). The input is clamped to
    [eps, 1 - eps] before applying the logit to guard against domain errors.

    Args:
        eps: Clamping epsilon for domain safety. Defaults to 1e-6.
        indices: Feature indices to transform along index_dim. None applies to all.
        index_dim: Axis that holds the feature dimension. Defaults to -1.

    Example:
        >>> t = LogitTransform(eps=1e-6)
        >>> y = t(probabilities)
        >>> p_reconstructed = t.inverse_transform(y)
    """

    def __init__(
        self,
        *,
        eps: float = 1e-6,
        indices=None,
        index_dim: int = -1,
    ) -> None:
        """Initialize LogitTransform.

        Args:
            eps: Clamping bound. Input is clamped to [eps, 1-eps]. Must be in (0, 0.5).
            indices: Feature indices to transform. None means all.
            index_dim: Axis along which to select features.

        Raises:
            ValueError: If eps is not in (0, 0.5).
        """
        if not (0 < eps < 0.5):
            raise ValueError(f"eps must be in (0, 0.5), got {eps}.")
        super().__init__(indices=indices, index_dim=index_dim)
        self.eps = eps

    def _compute(self, x: Tensor) -> Tensor:
        x_safe = torch.clamp(x, self.eps, 1.0 - self.eps)
        return torch.log(x_safe / (1.0 - x_safe))

    def _inverse_compute(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x)
