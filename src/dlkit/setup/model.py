from lightning.pytorch import LightningModule
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet

from dlkit.networks.blocks.basic_network import PipelineNetwork
from dlkit.settings import ModelSettings
from dlkit.transforms.pipeline import Pipeline
from dlkit.utils.loading import load_class, init_class


def initialize_model(
	settings: ModelSettings, settings_path: str | None = None, dataset=TimeSeriesDataSet | None
) -> LightningModule:
	"""Builds a LightningModule based on the provided settings and pipeline.
	Args:
	    settings (ModelSettings): The settings object for the model.
	    settings_path (str, optional): The path to the settings file. Defaults to None.
	    dataset (Dataset, optional): The dataset to use for the model. Defaults to None.

	Returns:
	    LightningModule: The LightningModule for the model.
	"""
	if settings.module_path != 'pytorch_forecasting':
		pipeline = Pipeline(
			feature_transforms=settings.feature_transforms,
			target_transforms=settings.target_transforms,
			is_autoencoder=settings.is_autoencoder,
		)
		model: LightningModule = init_class(settings, settings_path)  # noqa: D100
		return PipelineNetwork(settings=settings, model=model, pipeline=pipeline)  # noqa: D100

	if settings.module_path == 'pytorch_forecasting':
		class_name: type(TemporalFusionTransformer) = load_class(
			settings.name, settings.module_path, settings_path
		)  # noqa: D100
		return class_name(**settings.to_dict_compatible_with(class_name)).from_dataset(
			dataset,
		)  # noqa: D100

	raise ValueError(f'Unknown module path: {settings.module_path}')
