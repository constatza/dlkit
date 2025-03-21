from pathlib import Path
from dlkit.transforms.chaining import TransformationChain
from dlkit.utils.system_utils import import_dynamically, filter_kwargs
from dlkit.setup.transforms import initialize_transforms
from dlkit.datamodules.numpy_module import NumpyModule
from dlkit.settings.classes import DatamoduleSettings, Paths
from lightning import LightningDataModule


def initialize_datamodule(
    datamodule_config: DatamoduleSettings,
    paths_config: Paths,
) -> LightningDataModule:
    """
    Dynamically imports and sets up the datamodule based on the provided configuration.
    :return: LightningDataModule: The instantiated datamodule object.
    """

    transforms: TransformationChain = initialize_transforms(
        datamodule_config.transforms
    )

    datamodule_instance = NumpyModule(
        settings=datamodule_config, paths=paths_config, transform_chain=transforms
    )

    return datamodule_instance
