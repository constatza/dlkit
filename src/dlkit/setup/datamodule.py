from typing import Literal

from dlkit.utils.system_utils import import_dynamic
from dlkit.datamodules.in_memory import InMemoryModule
from dlkit.settings.general_settings import DatamoduleSettings, PathSettings
from dlkit.setup.transforms import initialize_transforms
from dlkit.transforms.chaining import Pipeline


def initialize_datamodule(
    datamodule_settings: DatamoduleSettings,
    paths: PathSettings,
    datamodule_device: Literal["cpu", "cuda"] = "cpu",
) -> InMemoryModule:
    """
    Dynamically imports and sets up the datamodule based on the provided configuration.
    :return: LightningDataModule: The instantiated datamodule object.
    """

    feature_transforms: Pipeline = initialize_transforms(
        datamodule_settings.feature_transforms
    )

    target_transforms: Pipeline = initialize_transforms(
        datamodule_settings.target_transforms
    )

    dataset = import_dynamic(
        datamodule_settings.dataset.name,
        prepend=datamodule_settings.dataset.module_path,
    )

    dataset = dataset(
        **datamodule_settings.dataset.to_dict_compatible_with(dataset),
        **paths.to_dict_compatible_with(dataset),
    )

    datamodule_instance = InMemoryModule(
        dataset=dataset,
        settings=datamodule_settings,
        paths=paths,
        device=datamodule_device,
        features_pipeline=feature_transforms,
        targets_pipeline=target_transforms,
    )

    return datamodule_instance
