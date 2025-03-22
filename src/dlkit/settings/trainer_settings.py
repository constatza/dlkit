from pydantic import Field, DirectoryPath
from .types import IntHyper
from .base_settings import BaseSettings


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
    logger: bool = Field(default=False, description="Whether to log the model.")
    enable_checkpointing: bool = Field(
        default=False, description="Whether to enable checkpointing."
    )
