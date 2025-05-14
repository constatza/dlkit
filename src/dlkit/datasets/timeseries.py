from collections.abc import Iterable

import polars as pl
from pydantic import FilePath
from torch.utils.data import Dataset

from dlkit.io.tables import read_table


class ForecastingDataset(Dataset):
	def __init__(
		self,
		filepath: FilePath,
		time_idx: str,
		group_ids: Iterable[str] | str,
		target: Iterable[str] | str,
	):
		super().__init__()
		self.filepath = filepath
		self.time_idx = time_idx
		self.target = (
			[
				target,
			]
			if isinstance(target, str)
			else target
		)
		self.group_ids = (
			[
				group_ids,
			]
			if isinstance(group_ids, str)
			else group_ids
		)

		df_pl = read_table(self.filepath)
		df = df_pl.select(pl.all()).sort(
			pl.col(
				self.group_ids
				+ [
					self.time_idx,
				]
			)
		)

		# subtract index columns
		self.variable_names = tuple(set(df_pl.columns) - {*self.group_ids, self.time_idx})
		self._df_polars = df
		self.df_pandas = df.to_pandas()

	def __len__(self) -> int:
		return len(self._df_polars)

	def __getitem__(self, idx: int):
		return self._df_polars[idx]
