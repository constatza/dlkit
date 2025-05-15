import abc

import torch
import torch.nn as nn


class Transform(nn.Module):
	"""Base class for tensor transformations.

	Subclasses must implement a forward method.
	The inverse method is optional; if provided, it must return a Maybe[torch.Tensor]:
	Some if the inverse is successful, or Nothing otherwise.
	The fit method is also optional; if present, it will be applied before forward.
	"""

	input_shape: tuple[int, ...] | None
	apply_inverse: bool
	_fitted: torch.Tensor

	def __init__(self, input_shape: tuple[int, ...] | None = None) -> None:
		super().__init__()
		self.apply_inverse = True
		self.register_buffer('_fitted', torch.zeros(1, requires_grad=False))
		self.input_shape = input_shape

	@abc.abstractmethod
	def forward(self, x: torch.Tensor) -> torch.Tensor: ...

	@abc.abstractmethod
	def inverse_transform(self, y: torch.Tensor) -> torch.Tensor: ...

	def transform(self, x: torch.Tensor) -> torch.Tensor:
		"""Apply the transformation to the input tensor."""
		return self.forward(x)

	def fit(self, data: torch.Tensor) -> None:
		"""Fit the transformation to the data.

		Args:
		    data (torch.Tensor): The data to fit the transformation to.
		"""
		self.fitted = True

	@property
	def fitted(self) -> bool:
		return self.get_buffer('_fitted').item() == 1

	@fitted.setter
	def fitted(self, value: bool) -> None:
		self._fitted.fill_(1 if value else 0)
