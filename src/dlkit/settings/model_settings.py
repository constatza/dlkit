from pydantic import Field

from dlkit.datatypes.basic import FloatHyper, IntHyper
from dlkit.datatypes.dataset import Shape
from .base_settings import BaseSettings, HyperParameterSettings, ClassSettings


class OptimizerSettings(HyperParameterSettings):
	name: str = Field(default='Adam', description='Optimizer name.')
	lr: FloatHyper | None = Field(default=None, description='Learning rate.')
	weight_decay: float = Field(default=0.0, description='Optional weight decay.')


class SchedulerSettings(BaseSettings):
	name: str = Field(default='ReduceLROnPlateau', description='Scheduler name.')
	factor: float = Field(default=0.8, description='Reduction factor.')
	patience: int = Field(
		default=10,
		description='Number of epochs with no improvement before reducing the LR.',
	)
	min_lr: float = Field(default=1e-5, description='Minimum learning rate.')


class TransformSettings(ClassSettings):
	name: str = Field(..., description='Name of the transform.')
	module_path: str = Field(
		default='dlkit.transforms', description='Module path to the transform.'
	)
	dim: tuple[int, ...] | None = Field(
		default=None,
		description='List of dimensions to apply the transform on.',
	)


class ModelSettings(HyperParameterSettings, ClassSettings):
	name: str = Field(..., description='Model namespace path.')
	module_path: str = Field(
		default='dlkit.networks',
		description='Module path to the model.',
	)

	shape: Shape = Field(default=Shape(), description='Model shape.')
	optimizer: OptimizerSettings = Field(
		default=OptimizerSettings(), description='Optimizer settings.'
	)
	scheduler: SchedulerSettings = Field(
		default=SchedulerSettings(), description='Scheduler settings.'
	)

	feature_transforms: tuple[TransformSettings, ...] = Field(
		default=(), description='List of transforms to apply to features.'
	)
	target_transforms: tuple[TransformSettings, ...] = Field(
		default=(), description='List of transforms to apply to targets.'
	)

	is_autoencoder: bool = Field(default=False, description='Whether the model is an autoencoder.')

	num_layers: IntHyper | None = Field(default=None, description='Number of layers.')
	latent_size: IntHyper | None = Field(default=None, description='Latent dimension size.')
	kernel_size: IntHyper | None = Field(default=None, description='Convolution kernel size.')
	latent_channels: IntHyper | None = Field(
		default=None, description='Number of latent channels before reduce to vector.'
	)
	latent_width: IntHyper | None = Field(
		default=None, description='Latent width before reduce to vector.'
	)
	latent_height: IntHyper | None = Field(
		default=None, description='Latent height before reduce to vector.'
	)
