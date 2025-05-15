from collections.abc import Sequence

import polars as pl
from pydantic import FilePath
from torch.utils.data import Dataset

from dlkit.io.tables import read_table


class ForecastingDataset(Dataset):
	def __init__(
		self,
		filepath: FilePath,
		time_idx: str,
		group_ids: Sequence[str] | str,
		target: Sequence[str] | str,
	):
		"""
		A dataset for time series forecasting with Pytorch Forecasting
		that wraps a polars dataframe.
		Args:
		    filepath:
		    time_idx:
		    group_ids:
		    target:
		"""
		super().__init__()
		self.filepath = filepath
		self.time_idx = time_idx
		self.target = target
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

		# subtract index columns
		self.variable_names = tuple(set(df_pl.columns) - {*self.group_ids, self.time_idx})
		self.df = df
		self.max_time_idx = df.select(pl.col(self.time_idx)).max().item(0, 0)

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
