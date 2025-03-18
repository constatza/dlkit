from pathlib import Path
from dlkit.transforms.chaining import TransformationChain
from dlkit.utils.system_utils import import_dynamically, filter_kwargs
from dlkit.setup.transforms import initialize_transforms
from dlkit.datamodules.numpy_module import NumpyModule
from dlkit.settings.classes import Datamodule, Paths
from lightning import LightningDataModule


def initialize_datamodule(
    datamodule_config: Datamodule,
    paths_config: Paths,
) -> LightningDataModule:
    """
    Dynamically imports and sets up the datamodule based on the provided configuration.
    :param config: dict: The configuration object for the datamodule.
    :return: LightningDataModule: The instantiated datamodule object.
    """

    idx_split = paths_config.get("idx_split", None)

    features_path = paths_config.get("features")
    targets_path = paths_config.get("targets", None)

    transforms: TransformationChain = initialize_transforms(config)

    datamodule_instance = NumpyModule(
        features_path=features_path,
        targets_path=targets_path,
        transform_chain=transforms,
        idx_split_path=idx_split,
        dataloader_config=config.get("dataloader"),
    )

    return datamodule_instance
