from lightning.pytorch import LightningModule
from pydantic import ValidationError

from dlkit.networks.blocks.basic_network import PipelineNetwork
from dlkit.settings import ModelSettings
from dlkit.transforms.pipeline import Pipeline
from dlkit.utils.import_utils import load_class


def initialize_model(
	settings: ModelSettings,
	settings_path: str,
) -> LightningModule:
	"""Dynamically imports and sets up the model based on the provided configuration.
	The configuration should include the name of the model as well as any parameters
	that need to be passed to the model's constructor.

	Args:
	    settings (Settings): The configuration object for the model.
	    settings_path (FilePath): The path to the settings file.

	Returns:
	    nn.Module: The instantiated model object.
	"""
	pipeline = Pipeline(
		feature_transforms=settings.feature_transforms,
		target_transforms=settings.target_transforms,
		is_autoencoder=settings.is_autoencoder,
	)

	try:
		model_class = load_class(
			class_name=settings.name, module_path=settings.module_path, settings_path=settings_path
		)
		model = model_class(**settings.to_dict_compatible_with(model_class))
	except ValidationError as e:
		raise ValueError(
			f'{e} \nIf you are trying hyperparameter optimization, please use the `hparams_optimization` script.'
		) from e

	return PipelineNetwork(settings=settings, model=model, pipeline=pipeline)
