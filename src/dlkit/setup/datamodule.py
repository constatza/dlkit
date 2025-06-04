from dlkit.datamodules import InMemoryModule
from dlkit.utils.split import get_or_create_idx_split
from dlkit.settings import DataloaderSettings, DataModuleSettings, PathSettings
from dlkit.settings.datamodule_settings import DatasetSettings
from dlkit.utils.loading import init_class


def build_datamodule(
    settings: DataModuleSettings,
    dataset_settings: DatasetSettings,
    dataloader_settings: DataloaderSettings,
    paths: PathSettings,
) -> InMemoryModule:
    """Builds a datamodule based on the provided settings and dataset.

    Args:
        settings (DataModuleSettings): The settings for the datamodule.
        dataset_settings (DatasetSettings): The settings for the dataset.
        dataloader_settings (DataloaderSettings): The settings for the dataloader.
        paths (PathSettings): The paths for saving and loading data.

    Returns:
        LightningDataModule: The LightningDataModule for the dataset.
    """
    dataset = init_class(dataset_settings, **paths.model_dump())

    idx_split = get_or_create_idx_split(
        n=len(dataset),
        filepath=paths.idx_split,
        save_dir=paths.input_dir,
        test_size=settings.test_size,
        val_size=settings.val_size,
    )

    datamodule_instance = init_class(
        settings,
        dataset=dataset,
        idx_split=idx_split,
        dataloader_settings=dataloader_settings,
    )

    return datamodule_instance  # noqa
