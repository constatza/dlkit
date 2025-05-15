from collections.abc import Sequence

import polars as pl
from pydantic import FilePath, validate_call
from torch.utils.data import Dataset

from dlkit.io.tables import read_table


class ForecastingDataset(Dataset):
	@validate_call
	def __init__(
		self,
		filepath: FilePath,
		time_idx: str,
		group_ids: Sequence[str] = ('sample',),
		target: Sequence[str] | str = 'target',
		encoder_length: int = 5,
		prediction_length: int = 3,
		time_varying_known_reals: Sequence[str] = ('time',),
		static_reals: Sequence[str] = ('param',),
	):
		"""
		A dataset for time series forecasting with Pytorch Forecasting
		that wraps a polars dataframe.
		Args:
		    filepath (FilePath): Path to the dataset file.
		    time_idx (str): Column name for time index.
		    group_ids (Sequence[str]): Column names for group IDs.
		    target (Sequence[str] | str): Column names for target variables.
		    encoder_length (int): Length of the encoder sequence.
		    prediction_length (int): Length of the prediction sequence.
		    time_varying_known_reals (Sequence[str]): Column names for known time-varying  real-valued variables.
		    static_reals (Sequence[str]): Column names for static real-valued variables.
		"""
		super().__init__()
		self.filepath = filepath
		self.time_idx = time_idx
		self.target = target
		self.encoder_length = encoder_length
		self.prediction_length = prediction_length
		self.static_reals = static_reals
		self.time_varying_known_reals = time_varying_known_reals
		self.group_ids = (
			[
				group_ids,
			]
			if isinstance(group_ids, str)
			else group_ids
		)
		self._time_idx_as_list = list((self.time_idx,))

		df_pl = read_table(self.filepath)
		df = df_pl.select(pl.all()).sort(pl.col(self.group_ids + self._time_idx_as_list))
		self.df = df

		# subtract index columns
		self.variable_names = tuple(set(df_pl.columns) - {*self.group_ids, self.time_idx})
		self.max_time_idx = df.select(pl.col(self.time_idx)).max().item(0, 0)

		self.time_varying_unknown_reals = list(
			set(self.variable_names) - set(self.time_varying_known_reals) - set(self.static_reals)
		)

	def __len__(self) -> int:
		return self.df.select(pl.struct(self.group_ids).n_unique()).item(0, 0)

	def __getitem__(self, idx):
		if isinstance(idx, Sequence):
			return self.__getitems__(idx)
		if isinstance(idx, slice):
			start = idx.start or 0
			stop = idx.stop or len(self)
			step = idx.step or 1
			idx = list(range(start, stop, step))
			return self.__getitems__(idx)
		return self.df.filter(pl.col(self.group_ids) == idx)

	def __getitems__(self, indices: Sequence[int]):
		"""Get items by group_ids column"""
		return self.df.filter(pl.col(self.group_ids).is_in(indices))
