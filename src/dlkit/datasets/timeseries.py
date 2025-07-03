import polars as pl
from collections.abc import Sequence
from pydantic import FilePath, validate_call
from pytorch_forecasting import TimeSeriesDataSet

from dlkit.io.tables import read_table
from dlkit.datasets.base import BaseDataset
from dlkit.datatypes.dataset import Shape
from dlkit.utils.general import slice_to_list
from .base import register_dataset


@register_dataset
class ForecastingDataset(BaseDataset):
    @validate_call
    def __init__(
        self,
        *,
        features: FilePath,
        time_idx: str,
        target: list[str] | str,
        group_ids: list[str],
        **kwargs,
    ):
        """A dataset for time series forecasting with Pytorch Forecasting
        that wraps a polars dataframe.

        Args:
            features (FilePath): Path to the dataset file.
            time_idx (str): Column name for time index.
            group_ids (Sequence[str]): Column names for group IDs.
            target (Sequence[str] | str): Column names for target variables.
            encoder_length (int): Length of the encoder sequence.
            prediction_length (int): Length of the prediction sequence.
            time_varying_known_reals (Sequence[str]): Column names for known time-varying  real-valued variables.
            static_reals (Sequence[str]): Column names for static real-valued variables.
        """
        super().__init__(features, None)
        df_pl = read_table(str(features))
        self.kwargs = kwargs

        self.timeseries = TimeSeriesDataSet(
            df_pl.to_pandas(), time_idx=time_idx, target=target, group_ids=group_ids, **kwargs
        )
        self.target = target
        self.time_idx = time_idx
        self.group_ids = (
            [
                group_ids,
            ]
            if isinstance(group_ids, str)
            else group_ids
        )
        self._time_idx_as_list = list((time_idx,))

        df = df_pl.select(pl.all()).sort(pl.col(self.group_ids + self._time_idx_as_list))
        self.df = df

        # subtract index columns
        self.variables = tuple(set(df_pl.columns) - {*self.group_ids, time_idx})

    def __len__(self) -> int:
        return self.df.select(pl.struct(self.group_ids).n_unique()).item(0, 0)

    def __getitem__(self, idx):
        if isinstance(idx, Sequence):
            return self.__getitems__(idx)
        if isinstance(idx, slice):
            idx = slice_to_list(idx, self.__len__())
            return self.__getitems__(idx)
        return self.df.filter(pl.col(self.group_ids) == idx)

    def __getitems__(self, indices: Sequence[int]):
        """Get items by group_ids column"""
        return self.df.filter(pl.col(self.group_ids).is_in(indices))

    def to_timeseries_dataset(self, idx: slice | Sequence[int] | int, **kwargs):
        return TimeSeriesDataSet(
            self[idx].to_pandas(),
            self.time_idx,
            self.target,
            self.group_ids,
            **self.kwargs,
            **kwargs,
        )

    #
    @property
    def shape(self):
        return Shape(
            features=self.df.filter(pl.col(self.time_idx).unique())
            .select(pl.col(self.variables))
            .shape,
            targets=self.df.filter(pl.col(self.time_idx).unique())
            .select(pl.col(self.target))
            .shape,
        )


def polars_to_timeseries(
    df: pl.DataFrame, time_idx: str, target: str | list[str], group_ids: list[str], **kwargs
):
    return TimeSeriesDataSet(
        df.to_pandas(),
        time_idx,
        target,
        group_ids,
        **kwargs,
    )
