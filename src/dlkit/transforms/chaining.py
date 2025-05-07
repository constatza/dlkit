from collections.abc import Sequence

import torch
from torch.nn import ModuleList
from loguru import logger

from .base import Scaler


class Pipeline(Scaler):
	"""Base class for chaining multiple transformations together."""

	direct_transforms: ModuleList
	inverse_transforms: ModuleList
	fitted: bool
	input_shape: tuple[int, ...] | None
	output_shape: tuple[int, ...] | None

	def __init__(self, transforms: Sequence[torch.nn.Module] | None = None) -> None:
		super().__init__()
		self.direct_transforms = ModuleList(transforms)
		self.fitted = False
		self.inverse_transforms = ModuleList(reversed(transforms))
		self.input_shape = None
		self.output_shape = None

	def fit(self, data: torch.Tensor) -> None:
		"""One-shot fit for all scalers in the pipeline, in order.

		Args:
		    data (torch.Tensor): Data to fit scalers on.
		"""
		self.input_shape = data.shape
		for mod in self.direct_transforms:
			# If it's a scaler, call fit
			if hasattr(mod, 'fit') and callable(mod.fit):
				mod.fit(data)
			data = mod(data)
		self.fitted = True
		self.output_shape = data.shape

	def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
		"""One-shot fit for all scalers in the pipeline, in order.

		Args:
		    data (torch.Tensor): Data to fit scalers on.
		"""
		self.fit(data)
		return self(data)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Sequentially pass x through each step in the pipeline.

		Args:
		    x (torch.Tensor): Input data.

		Returns:
		    torch.Tensor: Final output after all modules.
		"""
		for transform in self.direct_transforms:
			x = transform(x)
		return x

	def inverse_transform(self, y: torch.Tensor) -> torch.Tensor:
		"""Sequentially pass x through each step in the pipeline.

		Args:
		    y (torch.Tensor): Input data.

		Returns:
		    torch.Tensor: Final output after all modules.
		"""
		for transform in self.inverse_transforms:
			if not hasattr(transform, 'inverse_transform'):
				error = f'Transform {transform.__name__} does not have an inverse_transform method. Skipping.'
				logger.error(error)
				raise ValueError(error)

			if transform.apply_inverse:
				y = transform.inverse_transform(y)
		return y
