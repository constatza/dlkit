from lightning import LightningDataModule  # noqa: D100
from pydantic import Field, NonNegativeFloat

from dlkit.datasets.base import BaseDataset
from .base_settings import BaseSettings, ClassSettings


class DataloaderSettings(BaseSettings):
    """Settings for the pytorch dataloader."""

    num_workers: int = Field(default=1, description="Number of worker processes.")
    batch_size: int = Field(default=64, description="Batch size.")
    shuffle: bool = Field(default=False, description="Whether to shuffle the training data set.")
    persistent_workers: bool = Field(default=True, description="Whether to use persistent workers.")
    pin_memory: bool = Field(default=True, description="Whether to pin memory.")


class DatasetSettings(ClassSettings[BaseDataset]):
    """Settings for the pytorch dataset."""

    name: str = Field("NumpyDataset", description="Dataset name.")
    module_path: str = Field(
        default="dlkit.datasets",
        description="Module path where the dataset class is located.",
    )
    group_ids: list[str] | None = Field(
        default=None, description="Group IDs for pytorch forecasting TimeSeriesDataSet."
    )


class DataModuleSettings(ClassSettings[LightningDataModule]):
    """Settings for pytorch lightning datamodule."""

    name: str = Field(default="InMemoryModule", description="Datamodule name.")
    module_path: str = Field(
        default="dlkit.datamodules",
        description="Module path where the datamodule class is located.",
    )
    dataloader: DataloaderSettings = Field(
        default=DataloaderSettings(), description="Dataloader settings."
    )
    test_size: NonNegativeFloat = Field(
        default=0.15, description="Fraction of data used for testing."
    )
    val_size: NonNegativeFloat = Field(
        default=0.15, description="Fraction of data used for validation."
    )
