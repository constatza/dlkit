from collections.abc import Sequence
from torch import nn
from dlkit.utils.system_utils import import_dynamic
from dlkit.transforms.chaining import TransformationChain
from dlkit.settings import TransformSettings


def initialize_transforms(config: Sequence[TransformSettings]):

    if config:
        transform_chain = TransformationChain(
            nn.ModuleList([initialize(d) for d in config])
        )
    else:
        transform_chain = TransformationChain(nn.ModuleList([]))
    return transform_chain


def initialize(transform: TransformSettings):
    transform_class = import_dynamic(transform.name, prepend="dlkit.transforms")
    return transform_class(**transform.to_dict_compatible_with(transform_class))
