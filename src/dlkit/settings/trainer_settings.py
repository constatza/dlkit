from pydantic import Field, DirectoryPath
from .base_settings import BaseSettings
from typing import Literal


class CallbackSettings(BaseSettings):

    name: str | None = Field(
        default=None,
        description="Name of the callback.",
    )
    module_path: str = Field(
        default="lightning.pytorch.callbacks",
        description="Module path where the callback class is located.",
    )


class LoggerSettings(BaseSettings):

    name: str | None = Field(
        default=None,
        description="Name of the logger.",
    )
    module_path: str = Field(
        default="lightning.pytorch.loggers",
        description="Module path where the logger class is located.",
    )

    # save_dir: DirectoryPath | None = Field(
    #     None, description="Directory path where the logger should save the model."
    # )


class TrainerSettings(BaseSettings):
    max_epochs: int = Field(
        default=100,
        description="Maximum number of epochs to train for.",
    )
    gradient_clip_val: float | None = Field(
        default=None, description="Value for gradient clipping (if any)."
    )
    fast_dev_run: bool | int = Field(
        default=False,
        description="Flag for fast development run or number of batches to run in fast dev mode.",
    )
    default_root_dir: DirectoryPath | None = Field(
        default=None, description="Default root directory for the model."
    )
    enable_checkpointing: bool = Field(
        default=False, description="Whether to enable checkpointing."
    )
    callbacks: tuple[CallbackSettings, ...] = Field(
        tuple(), description="List of callbacks."
    )

    logger: LoggerSettings = Field(
        default=LoggerSettings(), description="Logger settings."
    )

    accelerator: Literal["cpu", "cuda"] = Field(
        default="cuda", description="Accelerator to use for training."
    )
