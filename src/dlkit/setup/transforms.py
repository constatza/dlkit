from collections.abc import Sequence


from dlkit.settings.datamodule_settings import TransformSettings
from dlkit.transforms.chaining import Pipeline
from dlkit.utils.system_utils import import_dynamic


def initialize_transforms(transformations: Sequence[TransformSettings]):
    return Pipeline([initialize(d) for d in transformations])


def initialize(transform: TransformSettings):
    transform_class = import_dynamic(transform.name, prepend="dlkit.transforms")
    return transform_class(**transform.to_dict_compatible_with(transform_class))
