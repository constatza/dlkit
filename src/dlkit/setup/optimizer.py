from collections.abc import Iterator
from pydantic import validate_call
from torch.optim import Optimizer
from dlkit.utils.system_utils import import_dynamically
from dlkit.settings import OptimizerSettings


@validate_call(config={"arbitrary_types_allowed": True})
def initialize_optimizer(config: OptimizerSettings, parameters: Iterator) -> Optimizer:
    """
    Initializes and returns an optimizer based on the provided configuration and parameters.

    Args:
        config (OptimizerSettings): The configuration settings for the optimizer, including
            the optimizer's name and any additional parameters.
        parameters (Iterator): An iterator over the model parameters to optimize.

    Returns:
        Optimizer: An instance of a PyTorch optimizer initialized with the specified parameters
        and configuration settings.
    """
    optimizer_class = import_dynamically(config.name, prepend="torch.optim")
    optimizer = optimizer_class(
        parameters, **config.to_dict_compatible_with(optimizer_class)
    )
    return optimizer
