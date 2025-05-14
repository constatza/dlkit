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
		self.df = dataset.df_pandas
		self.max_t = self.df[dataset.time_idx].max()
		minimum_length_for_prediction = (
			self.settings.dataset.encoder_length + self.settings.dataset.prediction_length
		)
		self.train_cutoff = self.df[self.time_idx].max() - 2 * minimum_length_for_prediction
		self.validation_cutoff = self.df[self.time_idx].max() - minimum_length_for_prediction
		self.timeseries_dataset = dataset

	def prepare_data(self) -> None:
		"""Called once per node (no state assignment). Only verify file existence here."""
		pass

	def setup(self, stage: str | None = None) -> None:
		"""Called on each GPU/process. Load via read_data(), then build datasets for fit/val/test/predict."""
		if stage in (None, 'fit'):
			# 1. Load raw data (Polars → pandas)

			# 2. Compute cutoff
			df = self.df
			train_df = self.df[self.df[self.time_idx] <= self.train_cutoff]

			time_varying_unknown_reals = list(
				set(self.timeseries_dataset.variable_names)
				- set(self.settings.dataset.time_varying_known)
				- set(self.settings.dataset.static_reals)
			)

			train_ds = TimeSeriesDataSet(
				train_df,
				time_idx=self.time_idx,
				group_ids=self.settings.dataset.group_ids,
				target=self.settings.dataset.target,
				max_encoder_length=self.settings.dataset.encoder_length,
				max_prediction_length=self.settings.dataset.prediction_length,
				time_varying_known_reals=self.settings.dataset.time_varying_known,
				time_varying_unknown_reals=time_varying_unknown_reals,
				static_reals=self.settings.dataset.static_reals,
				# target_normalizer=MultiNormalizer(
				# 	normalizers=[
				# 		TorchNormalizer(method='identity')
				# 		for target in self.settings.dataset.target
				# 	]
				# ),
			)
			val_data = df
			# (df[self.time_idx] > self.train_cutoff)
			# & (df[self.time_idx] <= self.validation_cutoff + 1)
			validation_ds = TimeSeriesDataSet.from_dataset(
				dataset=train_ds,
				data=val_data,
				stop_randomization=True,
				predict=True,
			)
			self.dataset.train = train_ds
			self.dataset.validation = validation_ds

		# 4. Build test for 'test'
		if stage in (None, 'test'):
			test_df = self.df[self.df[self.time_idx] > self.validation_cutoff]
			test_ds = TimeSeriesDataSet.from_dataset(
				dataset=self.dataset.train,
				data=test_df,
				stop_randomization=True,
				predict=True,
			)
			self.dataset.test = test_ds

		if stage in (None, 'predict'):
			predict_ds = TimeSeriesDataSet.from_dataset(
				dataset=self.dataset.train, data=self.df, stop_randomization=True, predict=True
			)
			self.dataset.predict = predict_ds

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
			train=False, batch_size=self.settings.dataloader.batch_size
		)

	def test_dataloader(self) -> DataLoader:
		if self.dataset.test is None:
			raise RuntimeError("`setup('test')` must be called before `test_dataloader()`")
		return self.dataset.test.to_dataloader(
			train=False,
		)

	def predict_dataloader(self) -> DataLoader:
		if self.dataset.predict is None:
			raise RuntimeError("`setup('predict')` must be called before `predict_dataloader()`")
		return self.dataset.predict.to_dataloader(train=False)
