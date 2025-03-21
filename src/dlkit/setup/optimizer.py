from collections.abc import Iterator
from torch.optim import Optimizer
from dlkit.utils.system_utils import import_dynamically, filter_kwargs
from dlkit.settings.classes import OptimizerSettings


def initialize_optimizer(config: OptimizerSettings, parameters: Iterator) -> Optimizer:

    optimizer_class = import_dynamically(config.name, prepend="torch.optim")
    optimizer = optimizer_class(parameters, **filter_kwargs(config.model_dump()))
    return optimizer
