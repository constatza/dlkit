import polars as pl
from collections.abc import Sequence
from pydantic import FilePath, validate_call
from pytorch_forecasting import TimeSeriesDataSet

from dlkit.tools.io.tables import read_table
from dlkit.core.datasets.base import BaseDataset
from dlkit.tools.utils.general import slice_to_list
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
        group_ids: list[str] | str,
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
        super().__init__()
        df_pl = read_table(str(features))
        self.kwargs = kwargs

        # Normalize group_ids to list[str]
        normalized_group_ids = group_ids if isinstance(group_ids, list) else [group_ids]

        # Underlying PyTorch Forecasting dataset built once and reused
        # Provide robust defaults for tiny toy datasets to avoid empty windows
        ts_kwargs = dict(kwargs)
        ts_kwargs.setdefault("min_encoder_length", 1)
        ts_kwargs.setdefault("min_prediction_length", 1)
        # Ensure minimal prediction index exists; PF default can be too strict for tiny dataflow
        if ts_kwargs.get("min_prediction_idx") is None:
            ts_kwargs["min_prediction_idx"] = 1
        self.timeseries = TimeSeriesDataSet(
            df_pl.to_pandas(),
            time_idx=time_idx,
            target=target,
            group_ids=normalized_group_ids,
            **ts_kwargs,
        )
        self.target = target
        self.time_idx = time_idx
        self.group_ids = normalized_group_ids
        self._time_idx_as_list = list((time_idx,))

        # Keep a sorted polars DataFrame for lightweight grouping/indexing operations
        df = df_pl.select(pl.all()).sort(pl.col(self.group_ids + self._time_idx_as_list))
        self.df = df

        # subtract index columns
        group_id_set = {*self.group_ids} if isinstance(self.group_ids, list) else {self.group_ids}
        self.variables = tuple(set(df_pl.columns) - group_id_set - {time_idx})

        # Materialize ordered group keys for robust ordinal indexing
        unique_groups = df.select(pl.struct(self.group_ids).alias("g")).unique()
        # Preserve order as in sorted df
        unique_groups = unique_groups.with_row_index(name="_ord").sort("_ord").drop("_ord")
        # Convert struct to Python tuples in consistent column order
        self.group_keys: list[tuple] = [
            tuple(g.values()) if hasattr(g, "values") else tuple(g)  # type: ignore[arg-type]
            for g in unique_groups["g"].to_list()
        ]

    def __len__(self) -> int:
        # Number of unique time series (group ids)
        return len(self.group_keys)

    def __getitem__(self, idx):
        if isinstance(idx, Sequence):
            return self.__getitems__(idx)
        if isinstance(idx, slice):
            idx = slice_to_list(idx, self.__len__())
            return self.__getitems__(idx)
        # A single sample is the full time series for an ordinal group index
        key = self._ordinal_to_key(idx)
        return self._filter_by_key(key)

    def __getitems__(self, indices: Sequence[int]):
        """Get multiple time series by ordinal group indices."""
        keys = [self._ordinal_to_key(i) for i in indices]
        return self._filter_by_keys(keys)

    def to_timeseries_dataset(self, idx: slice | Sequence[int] | int, **kwargs):
        return TimeSeriesDataSet(
            self[idx].to_pandas(),
            self.time_idx,
            self.target,
            self.group_ids,
            **self.kwargs,
            **kwargs,
        )


    # Thin-wrapper attribute access: expose TimeSeriesDataSet API transparently
    def __getattr__(self, name: str):
        """Delegate unknown attributes to the underlying TimeSeriesDataSet.

        This keeps the wrapper minimal while retaining full functionality
        expected from `pytorch_forecasting.TimeSeriesDataSet`.
        """
        try:
            return getattr(self.timeseries, name)
        except AttributeError:
            # As a secondary convenience, expose DataFrame attributes if present
            if hasattr(self, "df") and hasattr(self.df, name):
                return getattr(self.df, name)
            raise

    def __dir__(self) -> list[str]:
        base = set(super().__dir__())
        base.update(dir(self.timeseries))
        # include a few convenient DataFrame attributes for discoverability
        if hasattr(self, "df"):
            base.update(dir(self.df))
        return sorted(base)

    # Internal helpers
    def _ordinal_to_key(self, idx: int) -> tuple:
        if not isinstance(idx, int):
            raise TypeError("Index must be an integer for ordinal group selection")
        return self.group_keys[idx]

    def _filter_by_key(self, key: tuple) -> pl.DataFrame:
        # Build conjunction over group_id columns
        expr = None
        for col, val in zip(self.group_ids, key):
            cond = pl.col(col) == val
            expr = cond if expr is None else (expr & cond)
        return self.df.filter(expr) if expr is not None else self.df.head(0)

    def _filter_by_keys(self, keys: list[tuple]) -> pl.DataFrame:
        if not keys:
            return self.df.head(0)
        # Build disjunction of conjunctions
        disj = None
        for key in keys:
            conj = None
            for col, val in zip(self.group_ids, key):
                cond = pl.col(col) == val
                conj = cond if conj is None else (conj & cond)
            disj = conj if disj is None else (disj | conj)
        return self.df.filter(disj) if disj is not None else self.df.head(0)


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
