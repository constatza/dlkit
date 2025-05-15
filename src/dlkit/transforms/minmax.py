from collections.abc import Sequence

import torch
from pydantic import validate_call

from dlkit.transforms.base import Transform


class MinMaxScaler(Transform):
	"""Minimum-Maximum Scaler."""

	min: torch.Tensor
	max: torch.Tensor
	dim: int | Sequence[int]

	@validate_call()
	def __init__(
		self, dim: int | Sequence[int] = 0, input_shape: tuple[int, ...] | None = None
	) -> None:
		"""
		Important: This scaler transforms data in the range [-1, 1] by default!!!

		Args:
		    dim (int | Sequence[int], optional): The dimension(s) along which to compute
		        the minimum and maximum values. Defaults to 0.
		    input_shape (tuple[int, ...] | None, optional): The shape of the input data.
		"""
		super().__init__()
		moments_shape = [1, *input_shape]
		dim = dim if isinstance(dim, Sequence) else (dim,)
		size = len(moments_shape)
		# assure that dim has nonegative values
		self.dim = tuple([idx % size for idx in dim if idx < 0])

		# create zero tensors with the same number of dimensions as the input data
		# but with reduced size along the specified dimensions dim
		moments_shape = tuple([1 if i in self.dim else s for i, s in enumerate(moments_shape)])
		self.register_buffer('min', torch.zeros(moments_shape))
		self.register_buffer('max', torch.zeros(moments_shape))

	def fit(self, data: torch.Tensor) -> None:
		"""Compute the minimum and maximum values along the specified dimensions of the data.

		This method adjusts the internal state of the scaler to store the minimum
		and maximum values found in the provided tensor, enabling subsequent scaling
		operations.

		Args:
		    data (torch.Tensor): Input data used to determine the min and max values
		        for scaling. The dimensions to be considered can be specified during
		        initialization.

		"""
		self.min = torch.amin(input=data, dim=self.dim, keepdim=True)
		self.max = torch.amax(input=data, dim=self.dim, keepdim=True)
		self.fitted = True

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Scale to interval [-1, 1]."""
		return 2 * (x - self.min) / (self.max - self.min) - 1

	def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
		"""Return a module that lazily computes the inverse transformation
		using the current min and max values at runtime.
		"""
		if not self.fitted:
			raise RuntimeError(
				'Scaler has not been fitted yet. Call `fit` before using the inverse transformation.'
			)

		return (x + 1) / 2 * (self.max - self.min) + self.min
