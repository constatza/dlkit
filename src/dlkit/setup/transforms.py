from dlkit.utils.system_utils import import_dynamically, filter_kwargs
from dlkit.pipeline import Pipeline


def initialize_transforms(transforms_config):
    if not transforms_config:
        transforms_config = {
            "features": [{"name": "NumpyToTensor"}],
            "targets": [{"name": "NumpyToTensor"}],
        }

    features_pipeline = Pipeline(
        *[initialize(d) for d in transforms_config["features"]]
    )
    targets_pipeline = Pipeline(
        *[initialize(d) for d in transforms_config.get("targets", [])]
    )
    return features_pipeline, targets_pipeline


def initialize(d: dict):
    name = d.get("name")
    transform_class = import_dynamically(name, prepend="dlkit.transforms")
    return transform_class(**filter_kwargs(d))
