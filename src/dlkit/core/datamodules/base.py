from typing import Any, Union, TYPE_CHECKING
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
from dlkit.core.datatypes.split import IndexSplit
from dlkit.core.datasets.base import BaseDataset
from dlkit.core.datatypes.dataset import SplitDataset

if TYPE_CHECKING:
    from dlkit.tools.config.dataloader_settings import DataloaderSettings


class BaseDataModule(ABC, LightningDataModule):
    """Abstract base for PyTorch Lightning dataflow modules."""

    dataset: BaseDataset
    index_split: IndexSplit
    dataloader_settings: Union[dict[str, Any], "DataloaderSettings"]
    split_dataset: SplitDataset
    fitted: bool

    def __init__(
        self,
        dataset: BaseDataset,
        split: IndexSplit,
        dataloader: Union[dict[str, Any], "DataloaderSettings"],
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
        if hasattr(self.dataloader_settings, "to_dict"):
            # Use strict dict serialization (no signature filtering)
            kwargs = self.dataloader_settings.to_dict()  # type: ignore[attr-defined]
        else:
            # Plain dict
            kwargs = dict(self.dataloader_settings)  # type: ignore[arg-type]

        kwargs.update(overrides)
        # Safety: persistent_workers requires num_workers > 0
        try:
            num_workers = int(kwargs.get("num_workers", 0))
            if num_workers <= 0 and kwargs.get("persistent_workers", None):
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
