import torch
from typing import Literal
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset, DataLoader


from dlkit.transforms.chaining import Pipeline
from dlkit.settings import DataSettings
from dlkit.datatypes.dataset import Shape, SplitDataset, SplitIndices


class BaseDataModule(LightningDataModule):
	device: torch.device
	settings: DataSettings
	idx_split: SplitIndices
	features_pipeline: Pipeline
	targets_pipeline: Pipeline
	dataset: SplitDataset
	shape: Shape

	def __init__(
		self,
		dataset: Dataset,
		settings: DataSettings,
		device: Literal['cpu', 'cuda'] = 'cpu',
	) -> None:
		super().__init__()

		self.dataset = SplitDataset(raw=dataset)
		self.settings = settings
		self.device = torch.device(device)

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
