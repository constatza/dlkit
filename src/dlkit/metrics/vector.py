import torch
from torch import Tensor


def cosine_loss(predictions: Tensor, targets: Tensor) -> Tensor:
	# pylint: disable=not-callable
	return 1 - torch.nn.functional.cosine_similarity(predictions, targets, dim=-1)


def vector_norm(x: Tensor, dim: int = -1, ord: int = 2):
	# pylint: disable=not-callable
	return torch.linalg.vector_norm(x, ord=ord, dim=dim)


def sum_squares(x: Tensor, dim=-1):
	return torch.sum(x**2, dim=dim)


def mean_squares(x: Tensor, dim=-1):
	return torch.mean(torch.pow(x, 2), dim=dim)


def sum_abs(x: Tensor, dim=-1):
	return torch.sum(torch.abs(x), dim=dim)


def mean_abs(x, dim=-1):
	return torch.mean(torch.abs(x), dim=dim)


def rms_loss(predictions, targets):
	"""Root Mean Square Error."""
	return vector_norm(predictions - targets)


def rms_over_rms_loss(predictions, targets, eps=1e-8):
	"""Normalized Root Mean Square Error."""
	return rms_loss(predictions, targets) / (vector_norm(targets) + eps)


def mse_over_std_error(predictions: Tensor, targets: Tensor, eps: float = 1e-8) -> Tensor:
	"""Mean Squared Error over Standard Deviation."""
	return mean_squares(predictions - targets) / (torch.std(targets) + eps)
