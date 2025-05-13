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

	apply_inverse: bool

	def __init__(self):
		super().__init__()
		self.apply_inverse = True
		self.fitted = False

	@abc.abstractmethod
	def forward(self, x: torch.Tensor) -> torch.Tensor: ...

	@abc.abstractmethod
	def inverse_transform(self, y: torch.Tensor) -> torch.Tensor: ...

	def transform(self, x: torch.Tensor) -> torch.Tensor:
		"""Apply the transformation to the input tensor."""
		return self.forward(x)

	def fit(self, data: torch.Tensor) -> None:
		self.fitted = True
