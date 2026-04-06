from abc import ABC, abstractmethod
from typing import Any

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from dlkit.engine.data.datasets.base import BaseDataset
from dlkit.engine.data.splits import SplitDataset
from dlkit.infrastructure.config.dataloader_settings import DataloaderSettings
from dlkit.infrastructure.types.split import IndexSplit


class BaseDataModule(ABC, LightningDataModule):
    """Abstract base for PyTorch Lightning dataflow modules."""

    dataset: BaseDataset
    index_split: IndexSplit
    dataloader_settings: dict[str, Any] | DataloaderSettings
    split_dataset: SplitDataset
    fitted: bool

    def __init__(
        self,
        dataset: BaseDataset,
        split: IndexSplit,
        dataloader: dict[str, Any] | DataloaderSettings,
    ) -> None:
        """Store dataset, split indices, and loader settings."""
        super().__init__()
        self.dataset = dataset
        self.index_split = split
        self.dataloader_settings = dataloader
        self.split_dataset = SplitDataset(dataset, split)
        self.fitted = False

    def _get_dataloader_kwargs(self, loader_class: type, **overrides: Any) -> dict[str, Any]:
        """Get dataloader kwargs compatible with the given loader class.

        Args:
            loader_class: The DataLoader class to get compatible kwargs for
            **overrides: Additional kwargs to override/add

        Returns:
            dict[str, Any]: Compatible kwargs for the loader class
        """
        if isinstance(self.dataloader_settings, DataloaderSettings):
            kwargs = self.dataloader_settings.to_dict()
        else:
            kwargs = dict(self.dataloader_settings)

        kwargs.update(overrides)
        # Safety: persistent_workers requires num_workers > 0
        try:
            _nw = kwargs.get("num_workers", 0)
            num_workers = int(_nw) if isinstance(_nw, (int, str, float)) else 0
            if num_workers <= 0 and kwargs.get("persistent_workers"):
                kwargs["persistent_workers"] = False
        except Exception:
            kwargs.pop("persistent_workers", None)
        return kwargs

    @abstractmethod
    def setup(self, stage: str | None = None) -> None:
        """Prepare `self.dataset` for different stages (train/val/test)."""
        ...

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        """Get DataLoader for training set."""
        ...

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        """Get DataLoader for validation set."""
        ...

    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        """Get DataLoader for test set."""
        ...

    def predict_dataloader(self) -> DataLoader:
        """By default, use the test DataLoader for prediction."""
        return self.test_dataloader()
