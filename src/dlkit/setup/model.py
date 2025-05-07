from lightning.pytorch import LightningModule
from pydantic import ValidationError

from dlkit.datatypes.dataset import Shape
from dlkit.settings import ModelSettings
from dlkit.utils.system_utils import import_dynamic


def initialize_model(config: ModelSettings, shape: Shape) -> LightningModule:
	"""Dynamically imports and sets up the model based on the provided configuration.
	The configuration should include the name of the model as well as any parameters
	that need to be passed to the model's constructor.

	Args:
	    config (Settings): The configuration object for the model.
	    shape (dict): A dictionary containing the shapes of the features and targets.

	Returns:
	    nn.Module: The instantiated model object.
	"""
	config = config.model_copy(update={'shape': shape})

	prepend = 'dlkit.networks'
	try:
		model_class = import_dynamic(config.name, prepend=prepend)
		model = model_class(settings=config)
	except ValidationError as e:
		raise ValueError(
			f'{e} \nIf you are trying hyperparameter optimization, please use the `hparams_optimization` script.'
		)
	return model
