from lightning.pytorch import LightningDataModule
from torch_geometric.loader import DataLoader as PyGDataLoader


class GraphDataModule(LightningDataModule):
    def __init__(self, *, dataset, idx_split, dataloader_settings):
        super().__init__()
        self.dataset = dataset
        self.idx_split = idx_split
        self.dataloader_kwargs = dataloader_settings.to_dict_compatible_with(PyGDataLoader)

    def train_dataloader(self):
        return PyGDataLoader(self.dataset[self.idx_split.train], **self.dataloader_kwargs)

    def val_dataloader(self):
        return PyGDataLoader(self.dataset[self.idx_split.validation], **self.dataloader_kwargs)

    def test_dataloader(self):
        return PyGDataLoader(self.dataset[self.idx_split.test], **self.dataloader_kwargs)
