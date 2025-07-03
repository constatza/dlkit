from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Subset

from dlkit.datatypes.dataset import Shape, SplitDatasetOfType, SplitIndices
from dlkit.settings import DataloaderSettings


class InMemoryModule[Dataset_T](LightningDataModule):
    idx_split: SplitIndices
    split_dataset: SplitDatasetOfType
    dataloader_settings: DataloaderSettings
    shape: Shape
    fitted: bool

    def __init__(
        self,
        dataset: Dataset_T,
        idx_split: SplitIndices,
        dataloader_settings: DataloaderSettings,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.split_dataset = SplitDatasetOfType(raw=dataset)
        self.idx_split = idx_split
        self.fitted = False
        self.dataloader_settings = dataloader_settings

        self.split_dataset.train = Subset(self.split_dataset.raw, self.idx_split.train)
        self.split_dataset.validation = Subset(self.split_dataset.raw, self.idx_split.validation)
        self.split_dataset.test = Subset(self.split_dataset.raw, self.idx_split.test)
        self.split_dataset.predict = self.split_dataset.raw

    def train_dataloader(self):
        if self.split_dataset.train is None:
            raise RuntimeError("Call setup('fit') before setup('train')")
        return DataLoader(self.split_dataset.train, **self.dataloader_settings.model_dump())

    def val_dataloader(self):
        if self.split_dataset.validation is None:
            raise RuntimeError("Call setup('fit') before setup('validation')")
        return DataLoader(self.split_dataset.validation, **self.dataloader_settings.model_dump())

    def test_dataloader(self):
        if self.split_dataset.test is None:
            raise RuntimeError("Call setup('fit') before setup('test')")
        return DataLoader(self.split_dataset.test, **self.dataloader_settings.model_dump())

    def predict_dataloader(self):
        if self.split_dataset.predict is None:
            raise RuntimeError("Call setup('fit') before setup('predict')")
        return DataLoader(
            self.split_dataset.predict,
            shuffle=False,
            **self.dataloader_settings.to_dict_compatible_with(DataLoader, exclude=("shuffle",)),
        )
