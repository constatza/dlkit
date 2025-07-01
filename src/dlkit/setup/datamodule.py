from torch.utils.data import Dataset
from types import MappingProxyType
from lightning.pytorch import LightningDataModule
from dlkit.datamodules import InMemoryModule
from dlkit.utils.split import get_or_create_idx_split
from dlkit.settings import DataloaderSettings, DataModuleSettings
from dlkit.settings import PathSettings
from dlkit.utils.loading import init_class
from dlkit.datamodules import TimeSeriesDataModule, GraphDataModule

dataset_map = MappingProxyType({
    "GraphDataset": GraphDataModule,
    "TimeSeriesDataset": TimeSeriesDataModule,
    "InMemoryDataset": InMemoryModule,
})


def build_datamodule(
    *,
    settings: DataModuleSettings,
    dataset: Dataset,
    dataloader_settings: DataloaderSettings,
    paths: PathSettings,
) -> LightningDataModule:
    """Builds a datamodule based on the provided settings and dataset.

    Args:
        settings (DataModuleSettings): The settings for the datamodule.
        dataset (Dataset): The dataset to use for the datamodule.
        dataloader_settings (DataloaderSettings): The settings for the dataloader.
        paths (PathSettings): The paths for saving and loading data.

    Returns:
        LightningDataModule: The LightningDataModule for the dataset.
    """

    idx_split = get_or_create_idx_split(
        n=len(dataset),
        filepath=settings.idx_split_path,
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
