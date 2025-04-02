from pydantic import Field
from .base_settings import BaseSettings


class TransformSettings(BaseSettings):
    name: str = Field(..., description="Name of the transform.")
    dim: tuple[int, ...] = Field(
        None, description="List of dimensions to apply the transform on."
    )


class DataloaderSettings(BaseSettings):
    num_workers: int = Field(default=5, description="Number of worker processes.")
    batch_size: int = Field(default=64, description="Batch size.")
    shuffle: bool = Field(
        default=False, description="Whether to shuffle the training data set."
    )
    persistent_workers: bool = Field(
        default=True, description="Whether to use persistent workers."
    )
    pin_memory: bool = Field(default=True, description="Whether to pin memory.")


class DatamoduleSettings(BaseSettings):
    name: str = Field(..., description="Datamodule name.")
    test_size: float = Field(
        default=0.2, description="Fraction of data used for testing."
    )
    val_size: float = Field(
        default=0.5, description="Fraction of test data used for validation."
    )
    dataloader: DataloaderSettings = Field(..., description="Dataloader settings.")
    transforms: tuple[TransformSettings, ...] = Field(
        ..., description="List of transforms to apply."
    )
    is_autoencoder: bool = Field(
        default=False,
        description="Whether targets and features are the same.",
        frozen=False,
    )
