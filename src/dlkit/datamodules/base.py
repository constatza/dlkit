from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Subset, ConcatDataset

from dlkit.datatypes.dataset import Shape, SplitDatasetOfType, SplitIndices, Dataset_T
from dlkit.settings import DataloaderSettings


class InMemoryModule(LightningDataModule):
	idx_split: SplitIndices
	dataset: SplitDatasetOfType
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

		self.dataset = SplitDatasetOfType[Dataset_T]()
		self.idx_split = idx_split
		self.fitted = False
		self.dataloader_settings = dataloader_settings

		self.dataset.raw = dataset
		self.dataset.train = Subset(self.dataset.raw, self.idx_split.train)
		self.dataset.validation = Subset(self.dataset.raw, self.idx_split.validation)
		self.dataset.test = Subset(self.dataset.raw, self.idx_split.test)
		self.dataset.predict = ConcatDataset(
			[self.dataset.train, self.dataset.validation, self.dataset.test]
		)

		self.shape = Shape(
			features=self.dataset.train[0][0].shape, targets=self.dataset.train[0][1].shape
		)

	def train_dataloader(self):
		if self.dataset.train is None:
			raise RuntimeError("Call setup('fit') before setup('train')")
		return DataLoader(self.dataset.train, **self.dataloader_settings.model_dump())

	def val_dataloader(self):
		if self.dataset.validation is None:
			raise RuntimeError("Call setup('fit') before setup('validation')")
		return DataLoader(self.dataset.validation, **self.dataloader_settings.model_dump())

	def test_dataloader(self):
		if self.dataset.test is None:
			raise RuntimeError("Call setup('fit') before setup('test')")
		return DataLoader(self.dataset.test, **self.dataloader_settings.model_dump())

	def predict_dataloader(self):
		if self.dataset.predict is None:
			raise RuntimeError("Call setup('fit') before setup('predict')")
		return DataLoader(
			self.dataset.predict,
			shuffle=False,
			**self.dataloader_settings.to_dict_compatible_with(DataLoader, exclude=('shuffle',)),
		)
