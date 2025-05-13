from functools import wraps

import torch
from loguru import logger

from .base import Transform


def reshaper2d(func):
	@wraps(func)
	def wrapper(*args):
		data_position = len(args) - 1
		data = args[data_position]
		shape = data.shape
		if len(shape) < 2:
			return func(*args)
		else:
			data = data.reshape(-1, data.shape[-1])
			processed = func(*args[:-1], data)
			return processed.reshape((-1,) + shape[1:-1] + processed.shape[1:])

	return wrapper


class PCA(Transform):
	"""Principal Component Analysis (PCA) transformer using torch.pca_lowrank.

	This transformer computes the principal components efficiently by leveraging
	torch.pca_lowrank. It projects data onto the computed components and provides
	an inverse transformation module for approximate reconstruction.
	Additionally, it computes the explained variance ratio of the selected components
	relative to the total variance in the data.

	The transformer treats all dimensions prior to the last as sample dimensions.
	For example, for data with shape (N, T, D), it will be reshaped to (N*T, D)
	where the last dimension is considered as the feature dimension.
	"""

	def __init__(self, n_components: int, n_power_iterations: int = 2) -> None:
		"""Initialize the PCA transformer.

		Args:
		    n_components (int): Number of principal components to compute.
		    n_power_iterations (int, optional): Number of power iterations for pca_lowrank.
		        Defaults to 2.
		"""
		super().__init__()
		self.n_components = n_components
		self.n_power_iterations = n_power_iterations
		self.mean: torch.Tensor | None = None
		self.components: torch.Tensor | None = None  # Shape: (n_components, n_features)
		self.explained_variance: torch.Tensor | None = None
		self.explained_variance_ratio: torch.Tensor | None = None
		self.total_explained_variance: float | None = None
		self.fitted: bool = False
		self._orig_shape: tuple[int, ...] | None = None

	def fit(self, data: torch.Tensor, dim: int = -1) -> None:
		"""Fit the PCA transformer on the input data.

		This method computes the mean, principal components, explained variance,
		and explained variance ratio using torch.pca_lowrank. The data is manually
		centered before applying pca_lowrank. If the input data has more than 2 dimensions,
		all dimensions except the last are flattened into the sample dimension.

		Args:
		    data (torch.Tensor): Input data of shape (..., n_features). For example,
		        (N, T, D) where D is the feature dimension.
		    dim (int, optional): Dimension to be used as the feature dimension (to be reduced). Defaults to -1.
		"""
		# swqp dim to the last dimension for easier handling.
		data = data.transpose(dim, -1)

		if len(data.shape) > 2:
			data = data.reshape(-1, data.shape[-1])

		n_samples, _ = data.shape

		# Compute the mean manually for later use in forward and inverse transformations.
		self.mean = torch.mean(data, dim=0, keepdim=True)
		data_centered = data - self.mean

		# Compute total variance: sum of squared deviations divided by (n_samples - 1).
		total_variance = torch.sum(data_centered.pow(2)) / (n_samples - 1)

		# Use torch.pca_lowrank on already centered data by setting center=False.
		U, S, V = torch.pca_lowrank(
			data_centered,
			q=self.n_components,
			center=False,
			niter=self.n_power_iterations,
		)
		# Principal components are the right singular vectors.
		# V has shape (n_features, n_components); we transpose to (n_components, n_features)
		self.components = V.T

		# Compute the explained variance from singular values using Bessel's correction.
		self.explained_variance = S**2 / (n_samples - 1)

		# Compute the explained variance ratio with respect to the total variance.
		self.explained_variance_ratio = self.explained_variance / total_variance
		self.total_explained_variance = torch.sum(self.explained_variance_ratio).item()

		self.fitted = True
		logger.info(f'PCA total explained variance ratio: {self.total_explained_variance:.4e}')

	@reshaper2d
	def forward(self, data: torch.Tensor) -> torch.Tensor:
		"""Project the input data onto the principal components.

		Args:
		    data (torch.Tensor): Input data of shape (..., n_features). For example,
		        (N, T, D) where D is the feature dimension.

		Returns:
		    torch.Tensor: Projected data of shape (n_samples, n_components),
		    where n_samples is the product of all dimensions except the last.
		"""
		if not self.fitted or self.mean is None or self.components is None:
			raise RuntimeError('PCA has not been fitted yet. Call `fit` before `forward`.')

		# If data is more than 2D, flatten all dimensions except the last.

		# Center the data using the stored mean.
		data_centered = data - self.mean
		# Project the data onto the principal components.
		projected = torch.matmul(data_centered, self.components.T)

		return projected

	@reshaper2d
	def inverse_transform(self, projected: torch.Tensor) -> torch.Tensor:
		"""Return a module that performs the inverse transformation (approximate reconstruction).

		This module reconstructs the original data from the projected data. If the number
		of components is less than the original feature dimensions, the reconstruction is approximate.

		Returns:
		    nn.Module: A module that performs the inverse transformation.
		"""
		if not self.fitted or self.mean is None or self.components is None:
			raise RuntimeError(
				'PCA has not been fitted yet. Call `fit` before using the inverse transformation.'
			)
		device = projected.device
		mean = self.mean.to(device)
		components = self.components.to(device)
		# Reconstruct the data: approximate inverse of the PCA projection.
		reconstructed = torch.matmul(projected, components) + mean

		return reconstructed
