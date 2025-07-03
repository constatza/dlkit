from abc import ABC
from lightning.pytorch import LightningDataModule
from dlkit.datatypes.dataset import Shape, SplitIndices, DLkitDataset


class DLkitDataModule(LightningDataModule, ABC):
    dataset: DLkitDataset
    idx_split: SplitIndices
    shape: Shape
