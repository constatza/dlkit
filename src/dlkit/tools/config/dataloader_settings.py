import os

from pydantic import Field, PositiveInt

from .core.base_settings import BasicSettings


def _default_num_workers() -> int:
    """Get default number of workers (CPU count - 1, minimum 0)."""
    cpu_count = os.cpu_count() or 1
    return max(0, cpu_count - 1)


class DataloaderSettings(BasicSettings):
    """Settings for the pytorch dataloader."""

    num_workers: int = Field(
        default_factory=_default_num_workers, description="Number of worker processes."
    )
    batch_size: PositiveInt = Field(default=64, description="Batch size.")
    shuffle: bool = Field(default=True, description="Whether to shuffle the training dataflow set.")
    persistent_workers: bool = Field(default=True, description="Whether to use persistent workers.")
    pin_memory: bool = Field(default=True, description="Whether to pin memory.")
    follow_batch: tuple[str, ...] | None = Field(
        default=None, description="Follow batch dimensions."
    )
