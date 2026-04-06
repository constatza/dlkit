from typing import Any

from torch_geometric.loader import DataLoader as PyGDataLoader

from dlkit.engine.adapters.lightning.datamodules.base import BaseDataModule
from dlkit.engine.data.datasets.base import BaseDataset
from dlkit.infrastructure.types.split import IndexSplit


class GraphDataModule(BaseDataModule):
    def __init__(self, dataset: BaseDataset, split: IndexSplit, dataloader: dict[str, Any]) -> None:
        super().__init__(dataset, split, dataloader)

    def setup(self, stage: str | None = None) -> None:
        self.fitted = True

    def train_dataloader(self) -> PyGDataLoader:
        kwargs = self._get_dataloader_kwargs(PyGDataLoader, shuffle=True)
        return PyGDataLoader(self.split_dataset.train, **kwargs)

    def val_dataloader(self) -> PyGDataLoader:
        kwargs = self._get_dataloader_kwargs(PyGDataLoader, shuffle=False)
        return PyGDataLoader(self.split_dataset.validation, **kwargs)

    def test_dataloader(self) -> PyGDataLoader:
        kwargs = self._get_dataloader_kwargs(PyGDataLoader, shuffle=False)
        return PyGDataLoader(self.split_dataset.test, **kwargs)

    def predict_dataloader(self) -> PyGDataLoader:
        kwargs = self._get_dataloader_kwargs(PyGDataLoader, shuffle=False)
        return PyGDataLoader(self.split_dataset.predict, **kwargs)
