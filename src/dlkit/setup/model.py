import torch.nn as nn
from dlkit.utils.system_utils import import_dynamically
from dlkit.utils.system_utils import filter_kwargs
from dlkit.setup.transforms import initialize_transforms

from dlkit.settings.classes import ModelSettings


def initialize_model(config: ModelSettings, shapes: dict) -> nn.Module:
    """
    Dynamically imports and sets up the model based on the provided configuration.
    The configuration should include the name of the model as well as any parameters
    that need to be passed to the model's constructor.

    Args:
        config (Settings): The configuration object for the model.
        shapes (dict): A dictionary containing the shapes of the features and targets.

    Returns:
        nn.Module: The instantiated model object.
    """
    model_class = import_dynamically(config.name, prepend="dlkit.networks")
    input_shape = tuple(shapes[0])
    output_shape = tuple(shapes[1])
    config.update(
        {
            "input_shape": input_shape,
            "output_shape": output_shape,
            "optimizer_config": config["optimizer"],
            "scheduler_config": config["scheduler"],
        }
    )
    model = model_class(**filter_kwargs(model_config))
    return model
