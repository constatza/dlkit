import torch.optim as optim

from dlkit.utils.system_utils import import_dynamically, filter_kwargs


def initialize_optimizer(config, parameters):

    optimizer_name = config.get("name", "Adam")
    optimizer_class = import_dynamically(optimizer_name, prepend="torch.optim")
    config["params"] = parameters
    optimizer = optimizer_class(**filter_kwargs(config))
    return optimizer
