from typing import Literal

from pytorch_forecasting import TimeSeriesDataSet
from torch.utils.data import DataLoader

from dlkit.datamodules.base import BaseDataModule
from dlkit.datasets import ForecastingDataset
from dlkit.datatypes.dataset import SplitIndices
from dlkit.settings import DataSettings


class TimeSeriesDataModule(BaseDataModule):
	"""LightningDataModule for multivariate autoregressive forecasting, using read_data()."""

	def __init__(
		self,
		dataset: ForecastingDataset,
		settings: DataSettings,
		idx_split: SplitIndices,
		device: Literal['cpu', 'cuda'] = 'cpu',
		validation_length: int = 200,
	) -> None:
		super().__init__(
			dataset=dataset,
			settings=settings,
			device=device,
			idx_split=idx_split,
		)
		self.time_idx = dataset.time_idx
		self.df = dataset.df

	def setup(self, stage: str | None = None) -> None:
		"""Called on each GPU/process. Load via read_data(), then build datasets for fit/val/test/predict."""
		if stage == 'fit' and not self.fitted:
			# split samples w.r.t. group_ids and self.idx_split

			time_varying_unknown_reals = list(
				set(self.dataset.raw.variable_names)
				- set(self.settings.dataset.time_varying_known)
				- set(self.settings.dataset.static_reals)
			)

			dataset = TimeSeriesDataSet(
				data=self.dataset.raw[:].to_pandas(),
				time_idx=self.time_idx,
				group_ids=self.settings.dataset.group_ids,
				target=self.settings.dataset.target,
				max_encoder_length=self.settings.dataset.encoder_length,
				max_prediction_length=self.settings.dataset.prediction_length,
				time_varying_known_reals=self.settings.dataset.time_varying_known,
				time_varying_unknown_reals=time_varying_unknown_reals,
				static_reals=self.settings.dataset.static_reals,
			)

			train_ds = TimeSeriesDataSet.from_dataset(
				dataset=dataset,
				data=self.dataset.train[:].to_pandas(),
				stop_randomization=True,
			)
			val_ds = TimeSeriesDataSet.from_dataset(
				dataset=dataset,
				data=self.dataset.validation[:].to_pandas(),
				stop_randomization=True,
				predict=True,
			)
			test_ds = TimeSeriesDataSet.from_dataset(
				dataset=dataset,
				data=self.dataset.validation[:].to_pandas(),
				stop_randomization=True,
				predict=True,
			)

			self.dataset.train = train_ds
			self.dataset.validation = val_ds
			self.dataset.test = test_ds
			self.dataset.predict = dataset
			self.fitted = True

	def train_dataloader(self) -> DataLoader:
		if self.dataset.train is None:
			raise RuntimeError("`setup('fit')` must be called before `train_dataloader()`")
		dataloader_func = self.dataset.train.to_dataloader
		return dataloader_func(
			train=True, **self.settings.dataloader.to_dict_compatible_with(dataloader_func)
		)

	def val_dataloader(self) -> DataLoader:
		if self.dataset.validation is None:
			raise RuntimeError("`setup('fit')` must be called before `val_dataloader()`")
		return self.dataset.validation.to_dataloader(
			train=False, **self.settings.dataloader.to_dict_compatible_with(DataLoader)
		)

	def test_dataloader(self) -> DataLoader:
		if self.dataset.test is None:
			raise RuntimeError("`setup('test')` must be called before `test_dataloader()`")
		return self.dataset.test.to_dataloader(
			train=False, **self.settings.dataloader.to_dict_compatible_with(DataLoader)
		)

	def predict_dataloader(self) -> DataLoader:
		if self.dataset.predict is None:
			raise RuntimeError("`setup('predict')` must be called before `predict_dataloader()`")
		return self.dataset.predict.to_dataloader(
			train=False, **self.settings.dataloader.to_dict_compatible_with(DataLoader)
		)
