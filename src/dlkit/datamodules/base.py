from typing import Literal

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from dlkit.datatypes.dataset import Shape, SplitDatasetOfType, SplitIndices, Dataset_T
from dlkit.settings import DataSettings


class BaseDataModule(LightningDataModule):
	device: torch.device
	settings: DataSettings
	idx_split: SplitIndices
	dataset: SplitDatasetOfType
	shape: Shape

	def __init__(
		self,
		dataset: Dataset_T,
		settings: DataSettings,
		idx_split: SplitIndices,
		device: Literal['cpu', 'cuda'] = 'cpu',
	) -> None:
		super().__init__()

		self.dataset = SplitDatasetOfType[Dataset_T](raw=dataset)
		self.settings = settings
		self.device = torch.device(device)
		self.idx_split = idx_split

	def train_dataloader(self):
		if self.dataset.train is None:
			raise RuntimeError("Call setup('fit') before setup('train')")
		return DataLoader(self.dataset.train, **self.settings.dataloader.model_dump())

	def val_dataloader(self):
		if self.dataset.validation is None:
			raise RuntimeError("Call setup('fit') before setup('validation')")
		return DataLoader(self.dataset.validation, **self.settings.dataloader.model_dump())

	def test_dataloader(self):
		if self.dataset.test is None:
			raise RuntimeError("Call setup('fit') before setup('test')")
		return DataLoader(self.dataset.test, **self.settings.dataloader.model_dump())

	def predict_dataloader(self):
		if self.dataset.predict is None:
			raise RuntimeError("Call setup('fit') before setup('predict')")
		return DataLoader(
			self.dataset.predict,
			shuffle=False,
			**self.settings.dataloader.to_dict_compatible_with(DataLoader, exclude=('shuffle',)),
		)
