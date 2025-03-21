import torch.nn as nn
from pydantic_core._pydantic_core import ValidationError

from dlkit.utils.system_utils import import_dynamically

from dlkit.settings.classes import ModelSettings, Shape


def initialize_model(config: ModelSettings, shape: Shape) -> nn.Module:
    """
    Dynamically imports and sets up the model based on the provided configuration.
    The configuration should include the name of the model as well as any parameters
    that need to be passed to the model's constructor.

    Args:
        config (Settings): The configuration object for the model.
        shape (dict): A dictionary containing the shapes of the features and targets.

    Returns:
        nn.Module: The instantiated model object.
    """
    model_class = import_dynamically(config.name, prepend="dlkit.networks")
    config.shape = shape

    try:
        model = model_class(settings=config)
    except ValidationError as e:
        raise ValueError(
            f"{e} \nIf you are trying hyperparameter optimization, please use the `hparams_optimization` script."
        )
    return model
