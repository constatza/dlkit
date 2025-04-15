from typing import Literal

from dlkit.datamodules.numpy_module import NumpyModule
from dlkit.settings.general_settings import DataSettings, PathSettings
from dlkit.setup.transforms import initialize_transforms
from dlkit.transforms.chaining import TransformationChain


def initialize_datamodule(
    datamodule_settings: DataSettings,
    paths: PathSettings,
    datamodule_device: Literal["cpu", "cuda"] = "cpu",
) -> NumpyModule:
    """
    Dynamically imports and sets up the datamodule based on the provided configuration.
    :return: LightningDataModule: The instantiated datamodule object.
    """

    transforms: TransformationChain = initialize_transforms(
        datamodule_settings.transforms
    )

    datamodule_instance = NumpyModule(
        settings=datamodule_settings,
        paths=paths,
        device=datamodule_device,
        transform_chain=transforms,
    )

    return datamodule_instance
