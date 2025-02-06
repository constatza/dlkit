from torch import nn
from dlkit.utils.system_utils import import_dynamically, filter_kwargs
from dlkit.transforms.chaining import TransformationChain


def initialize_transforms(config):

    config = config.get("transforms", [])
    if config:
        transform_chain = TransformationChain(
            nn.ModuleList([initialize(d) for d in config])
        )
    else:
        transform_chain = TransformationChain(nn.ModuleList([]))
    return transform_chain


def initialize(d: dict):
    name = d.get("name")
    transform_class = import_dynamically(name, prepend="dlkit.transforms")
    return transform_class(**filter_kwargs(d))
