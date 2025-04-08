from dlkit.transforms.chaining import TransformationChain
from dlkit.setup.transforms import initialize_transforms
from dlkit.datamodules.numpy_module import NumpyModule
from dlkit.settings.general_settings import DatamoduleSettings, PathSettings


def initialize_datamodule(
    datamodule_config: DatamoduleSettings,
    paths_config: PathSettings,
) -> NumpyModule:
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
