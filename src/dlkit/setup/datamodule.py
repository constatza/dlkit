from typing import Literal

from dlkit.utils.system_utils import import_dynamic
from dlkit.settings.general_settings import DataSettings, PathSettings
from dlkit.setup.transforms import initialize_transforms
from dlkit.transforms.chaining import Pipeline
from dlkit.datamodules.base import BaseDataModule
from dlkit.datamodules.utils import get_or_create_idx_split


def initialize_datamodule(
    data_settings: DataSettings,
    paths: PathSettings,
    datamodule_device: Literal["cpu", "cuda"] = "cpu",
) -> BaseDataModule:
    """
    Dynamically imports and sets up the datamodule based on the provided configuration.
    :return: LightningDataModule: The instantiated datamodule object.
    """

    feature_transforms: Pipeline = initialize_transforms(
        data_settings.feature_transforms
    )

    target_transforms: Pipeline = initialize_transforms(data_settings.target_transforms)

    dataset = import_dynamic(
        data_settings.dataset.name,
        prepend=data_settings.dataset.module_path,
    )

    dataset = dataset(
        **data_settings.dataset.to_dict_compatible_with(dataset),
        **paths.to_dict_compatible_with(dataset),
    )

    module = import_dynamic(
        data_settings.module.name,
        prepend=data_settings.module.module_path,
    )

    idx_split = get_or_create_idx_split(
        n=len(dataset),
        filepath=paths.idx_split,
        save_dir=paths.input_dir,
        test_size=data_settings.test_size,
        val_size=data_settings.val_size,
    )

    datamodule_instance = module(
        dataset=dataset,
        settings=data_settings,
        device=datamodule_device,
        features_pipeline=feature_transforms,
        targets_pipeline=target_transforms,
        idx_split=idx_split,
    )

    return datamodule_instance
