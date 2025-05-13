from collections.abc import Sequence
from itertools import zip_longest
from typing import Literal

import torch
from loguru import logger
from torch.nn import ModuleList

from dlkit.datatypes.dataset import Shape
from dlkit.settings.model_settings import TransformSettings
from .base import Transform


class Pipeline(Transform):
	"""Base class for chaining multiple transformations together."""

	feature_transforms: ModuleList
	target_transforms: ModuleList
	original_shape: Shape | None
	transformed_shape: Shape | None
	fitted: bool
	is_autoencoder: bool

	def __init__(
		self,
		feature_transforms: Sequence[TransformSettings],
		target_transforms: Sequence[TransformSettings] = (),
		is_autoencoder: bool = False,
	) -> None:
		super().__init__()
		feature_transforms = [d.construct_class_dynamic() for d in feature_transforms]
		target_transforms = [d.construct_class_dynamic() for d in target_transforms]
		self.feature_transforms = ModuleList(feature_transforms)
		self.target_transforms = (
			ModuleList(target_transforms) if not is_autoencoder else self.feature_transforms
		)
		self.fitted = False
		self.original_shape = None
		self.transformed_shape = None
		self.is_autoencoder = is_autoencoder

	def fit(self, x: torch.Tensor, y: torch.Tensor | None = None) -> None:
		"""One-shot fit for all scalers in the pipeline, in order.

		Args:
		    x (torch.Tensor): Input data.
		    y (torch.Tensor, optional): Target data. Defaults to None.
		"""
		self.original_shape = Shape(features=x.shape, targets=y.shape)
		with torch.no_grad():
			for transform_x, transform_y in zip_longest(
				self.feature_transforms,
				self.target_transforms,
			):
				# If it's a scaler, call fit
				if hasattr(transform_x, 'fit') and callable(transform_x.fit):
					logger.info(f'Using {transform_x.__class__.__name__}.')
					transform_x.fit(x)
					x = transform_x(x)
				if self.is_autoencoder:
					y = x
				elif hasattr(transform_y, 'fit') and callable(transform_y.fit):
					transform_y.fit(y)
					y = transform_y(y)
		self.fitted = True
		self.transformed_shape = Shape(features=x.shape, targets=y.shape)

	def forward(
		self, x: torch.Tensor, which: Literal['features', 'targets'] = 'features'
	) -> torch.Tensor:
		"""Sequentially pass x through each step in the pipeline.

		Args:
		    x (torch.Tensor): Input data.
		    which (Literal['features', 'targets'], optional): Which set to apply transformation. Defaults to 'features'.

		Returns:
		    torch.Tensor: Final output after all modules.
		"""
		if not self.fitted:
			warn_unfit_pipeline()

		pipeline = self.feature_transforms
		if which == 'targets':
			pipeline = self.target_transforms

		for transform in pipeline:
			x = transform(x)
		return x

	def inverse_transform(
		self, y: torch.Tensor, which: Literal['features', 'targets'] = 'targets'
	) -> torch.Tensor:
		"""Sequentially pass x through each step in the pipeline.

		Args:
		    y (torch.Tensor): Input data.
		    which (Literal['features', 'targets'], optional): Which set to apply inverse transformation.

		Returns:
		    torch.Tensor: Final output after all modules.
		"""
		if not self.fitted:
			warn_unfit_pipeline()

		pipeline = self.target_transforms
		if which == 'features':
			pipeline = self.feature_transforms

		for transform in pipeline[::-1]:
			y = transform.inverse_transform(y)
		return y


def warn_unfit_pipeline() -> None:
	error = 'Pipeline must be fitted before calling inverse_transform.'
	logger.error(error)
