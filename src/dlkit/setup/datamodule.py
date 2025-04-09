from dlkit.transforms.chaining import TransformationChain
from dlkit.setup.transforms import initialize_transforms
from dlkit.datamodules.numpy_module import NumpyModule
from dlkit.settings.general_settings import DataSettings, PathSettings


def initialize_datamodule(
    datamodule_settings: DataSettings,
    paths: PathSettings,
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
        transform_chain=transforms,
    )

    return datamodule_instance
