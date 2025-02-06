from pathlib import Path
from dlkit.transforms.chaining import TransformationChain
from dlkit.utils.system_utils import import_dynamically, filter_kwargs
from dlkit.setup.transforms import initialize_transforms
from dlkit.datamodules.numpy_module import NumpyModule


def initialize_datamodule(config):
    """
    Dynamically imports and sets up the datamodule based on the provided configuration.
    :param config: dict: The configuration object for the datamodule.
    :return: LightningDataModule: The instantiated datamodule object.
    """

    datamodel_config = config.get("datamodule")
    paths_config = config.get("paths")

    # datamodule_class = import_dynamically(
    #     datamodel_config.get("name"), prepend="dlkit.datamodules"
    # )

    save_dir = (
        paths_config.get("datamodule", None)
        or Path(paths_config.get("output")) / "datamodule"
    )

    features_path = paths_config.get("features")
    targets_path = paths_config.get("targets", None)

    save_dir.mkdir(parents=True, exist_ok=True)

    transforms: TransformationChain = initialize_transforms(config)

    datamodule_instance = NumpyModule(
        features_path=features_path,
        targets_path=targets_path,
        transform_chain=transforms,
        idx_path=save_dir,
        dataloader_config=config.get("dataloader"),
    )

    return datamodule_instance
