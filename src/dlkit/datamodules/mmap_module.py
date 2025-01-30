from pathlib import Path
from typing import Optional
from lightning import LightningDataModule
from pydantic import FilePath, validate_call
from torch.utils.data import Dataset, DataLoader

from dlkit.datasets.lazy_dataset import LazyDataset, prepare_datasets
from dlkit.io.logging import get_logger


logger = get_logger(__name__)


class MMapModule(LightningDataModule):
    """
    LightningDataModule for handling datasets with train/val/test splits and lazy loading.

    Args:
        dataset (LazyDataset): Dataset to use.
        dataloader_config (dict, optional): Configuration for DataLoader. Defaults to None.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 64.
        save_dir (Path, optional): Directory to save indices. Defaults to Path(".").
    """

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        dataset: LazyDataset,
        dataloader_config: dict | None = None,
        batch_size: int = 64,
        save_dir: Path = Path("."),
    ):
        super().__init__()
        self.dataset = dataset
        if dataloader_config is None:
            dataloader_config = {}
        self.batch_size = batch_size
        self.dataloader_config = dataloader_config
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.predict_set = None

    def prepare_data(self) -> None:
        """
        Prepare data by generating or loading train/val/test indices.
        """
        if not self.dataset.indices_path:
            logger.info("Indices not found. Generating new indices.")
        else:
            logger.info(f"Using indices from {self.dataset.indices_path}.")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up datasets for different stages.

        Args:
            stage (Optional[str], optional): Stage ('fit', 'test', 'predict'). Defaults to None.
        """
        self.train_set, self.val_set, self.test_set, self.predict_set = (
            prepare_datasets(
                features_path=self.dataset.features_path,
                targets_path=self.dataset.targets_path,
                transform_chain=self.dataset.transforms,
                indices_path=self.dataset.indices_path,
                test_size=self.dataset.test_size,
                val_size=self.dataset.val_size,
            )
        )

    def create_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        """
        Create a DataLoader for a given dataset.

        Args:
            dataset (Dataset): Dataset to load.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            DataLoader: DataLoader for the dataset.
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            **self.dataloader_config,
        )

    def train_dataloader(self) -> DataLoader:
        return self.create_dataloader(self.train_set, shuffle=False)

    def val_dataloader(self) -> DataLoader:
        return self.create_dataloader(self.val_set, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self.create_dataloader(self.test_set, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for predictions.

        Returns:
            DataLoader: DataLoader for the prediction dataset.
        """
        return self.create_dataloader(self.predict_set, shuffle=False)
