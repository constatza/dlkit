from collections.abc import Sequence

import torch

from dlkit.core.training.transforms.base import Transform
from pydantic import validate_call, ConfigDict


class MinMaxScaler(Transform):
    """Minimum-Maximum Scaler."""

    min: torch.Tensor
    max: torch.Tensor
    dim: int | Sequence[int]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self, *, dim: int | Sequence[int] = 0, input_shape: Sequence[int] | torch.Size
    ) -> None:
        """Important: This scaler transforms dataflow in the range [-1, 1] by default!!!

        Args:
            dim (int | Sequence[int], optional): The dimension(s) along which to compute
                the minimum and maximum values. Defaults to 0.
        """
        super().__init__(input_shape=input_shape)
        dim = dim if isinstance(dim, Sequence) else (dim,)
        size = len(self.input_shape)
        # assure that dim has nonegative values
        self.dim = tuple([idx % size for idx in dim])

        # create zero tensors with the same number of dimensions as the input dataflow
        # but with reduced size along the specified dimensions dim
        moments_shape = tuple([1 if i in self.dim else s for i, s in enumerate(self.input_shape)])
        self.register_buffer("min", torch.zeros(moments_shape))
        self.register_buffer("max", torch.ones(moments_shape))

    def fit(self, data: torch.Tensor) -> None:
        """Compute (and accumulate) the min/max along specified dimensions.

        Adjusts the internal state to store the minimum and maximum values found in
        the provided tensor, enabling subsequent scaling operations. If called multiple
        times (e.g., during global fitting over multiple batches), the scaler accumulates
        global min/max values across calls.

        Args:
            data (torch.Tensor): Input dataflow used to determine the min and max values for
                scaling. The dimensions to be considered are set during initialization via ``dim``.
        """
        current_min = torch.amin(input=data, dim=self.dim, keepdim=True)
        current_max = torch.amax(input=data, dim=self.dim, keepdim=True)
        if self.fitted:
            # Accumulate global min/max across multiple fit() calls
            self.min = torch.minimum(self.min, current_min)
            self.max = torch.maximum(self.max, current_max)
        else:
            self.min = current_min
            self.max = current_max
            self.fitted = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale to interval [-1, 1]."""
        return 2 * (x - self.min) / (self.max - self.min) - 1

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Return a module that lazily computes the inverse transformation
        using the current min and max values at runtime.
        """
        return (x + 1) / 2 * (self.max - self.min) + self.min
