from pydantic import Field

from .base_settings import BaseSettings


class TransformSettings(BaseSettings):
    name: str = Field(..., description="Name of the transform.")
    dim: tuple[int, ...] = Field(
        None, description="List of dimensions to apply the transform on."
    )


class DataloaderSettings(BaseSettings):
    num_workers: int = Field(default=1, description="Number of worker processes.")
    batch_size: int = Field(default=64, description="Batch size.")
    shuffle: bool = Field(
        default=False, description="Whether to shuffle the training data set."
    )
    persistent_workers: bool = Field(
        default=True, description="Whether to use persistent workers."
    )
    pin_memory: bool = Field(default=True, description="Whether to pin memory.")


class DatasetSettings(BaseSettings):
    name: str = Field("NumpyDataset", description="Dataset name.")
    module_path: str = Field(
        default="dlkit.datasets",
        description="Module path where the dataset class is located.",
    )


class DataModuleSettings(BaseSettings):
    name: str = Field(default="InMemoryModule", description="Datamodule name.")
    module_path: str = Field(
        default="dlkit.datamodules",
        description="Module path where the datamodule class is located.",
    )


class DataSettings(BaseSettings):
    module: DataModuleSettings = Field(
        DataModuleSettings(), description="Dataset settings."
    )
    dataloader: DataloaderSettings = Field(
        DataloaderSettings(), description="Dataloader settings."
    )
    test_size: float = Field(
        default=0.15, description="Fraction of data used for testing."
    )
    val_size: float = Field(
        default=0.15, description="Fraction of data used for validation."
    )
    feature_transforms: tuple[TransformSettings, ...] = Field(
        default=(), description="List of transforms to apply to features."
    )
    target_transforms: tuple[TransformSettings, ...] = Field(
        default=(), description="List of transforms to apply to targets."
    )

    dataset: DatasetSettings = Field(
        default=DatasetSettings(), description="Dataset settings."
    )
    targets_exist: bool = Field(
        default=True,
        description="Whether dataset has both features and targets.",
        frozen=False,
    )
