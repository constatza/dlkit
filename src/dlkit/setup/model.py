from lightning.pytorch import LightningModule
from pydantic_core._pydantic_core import ValidationError
from pydantic import validate_call

from dlkit.utils.system_utils import import_dynamically

from dlkit.settings import ModelSettings
from dlkit.settings.types import Shape


@validate_call
def initialize_model(config: ModelSettings, shape: Shape) -> LightningModule:
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
    config = config.model_copy(update={"shape": shape})

    try:
        model = model_class(settings=config)
    except ValidationError as e:
        raise ValueError(
            f"{e} \nIf you are trying hyperparameter optimization, please use the `hparams_optimization` script."
        )
    return model
