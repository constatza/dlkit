from torch import Tensor

from dlkit.domain.transforms.base import PartialTransform


class AffineTransform(PartialTransform):
    """Affine transform: scale * x + shift with inverse (x - shift) / scale.

    A parameter-free alternative to StandardScaler when the desired scale and
    shift are known in advance (e.g., unit conversion, manual normalisation).

    Args:
        scale: Multiplicative factor. Must be non-zero.
        shift: Additive term applied after scaling.
        indices: Feature indices to transform along index_dim. None applies to all.
        index_dim: Axis that holds the feature dimension. Defaults to -1.

    Example:
        >>> t = AffineTransform(scale=0.01, shift=-0.5)
        >>> y = t(x)
        >>> x_reconstructed = t.inverse_transform(y)
    """

    def __init__(
        self,
        *,
        scale: float = 1.0,
        shift: float = 0.0,
        indices=None,
        index_dim: int = -1,
    ) -> None:
        """Initialize AffineTransform.

        Args:
            scale: Multiplicative factor. Must be non-zero.
            shift: Additive term.
            indices: Feature indices to transform. None means all.
            index_dim: Axis along which to select features.

        Raises:
            ValueError: If scale is 0.
        """
        if scale == 0:
            raise ValueError("scale must be non-zero.")
        super().__init__(indices=indices, index_dim=index_dim)
        self.scale = scale
        self.shift = shift

    def _compute(self, x: Tensor) -> Tensor:
        return self.scale * x + self.shift

    def _inverse_compute(self, x: Tensor) -> Tensor:
        return (x - self.shift) / self.scale
