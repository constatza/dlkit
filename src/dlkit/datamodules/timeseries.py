from pytorch_forecasting import TimeSeriesDataSet
from torch.utils.data import DataLoader

from dlkit.datamodules.base import InMemoryModule
from dlkit.datasets import ForecastingDataset
from dlkit.datasets.timeseries import polars_to_timeseries
from dlkit.datatypes.dataset import SplitIndices
from dlkit.settings import DataloaderSettings


class TimeSeriesDataModule(InMemoryModule):
    """LightningDataModule for multivariate autoregressive forecasting, using read_data()."""

    def __init__(
        self,
        dataset: ForecastingDataset,
        dataloader_settings: DataloaderSettings,
        idx_split: SplitIndices,
    ) -> None:
        super().__init__(
            dataset=dataset,
            dataloader_settings=dataloader_settings,
            idx_split=idx_split,
        )

    def setup(self, stage: str | None = None) -> None:
        """Called on each GPU/process. Load via read_data(), then build datasets for fit/val/test/predict."""
        if not self.fitted:
            # split samples w.r.t. group_ids and self.idx_split

            dataset = polars_to_timeseries(
                self.dataset.raw[:],
                self.dataset.raw.time_idx,
                self.dataset.raw.target,
                self.dataset.raw.group_ids,
                **self.dataset.raw.kwargs,
            )

            train_ds = TimeSeriesDataSet.from_dataset(
                dataset,
                data=self.dataset.train[:].to_pandas(),
            )
            val_ds = TimeSeriesDataSet.from_dataset(
                dataset,
                data=self.dataset.validation[:].to_pandas(),
                predict=True,
                stop_randomization=True,
            )
            test_ds = TimeSeriesDataSet.from_dataset(
                dataset,
                data=self.dataset.test[:].to_pandas(),
                predict=True,
                stop_randomization=True,
            )

            predict_ds = TimeSeriesDataSet.from_dataset(
                dataset,
                data=self.dataset.predict[:].to_pandas(),
                predict=True,
                stop_randomization=True,
            )

            self.dataset.train = train_ds
            self.dataset.validation = val_ds
            self.dataset.test = test_ds
            self.dataset.predict = predict_ds
            self.fitted = True

    def train_dataloader(self) -> DataLoader:
        if self.dataset.train is None:
            raise RuntimeError("`setup('fit')` must be called before `train_dataloader()`")
        dataloader_func = self.dataset.train.to_dataloader
        return dataloader_func(
            train=True, **self.dataloader_settings.to_dict_compatible_with(dataloader_func)
        )

    def val_dataloader(self) -> DataLoader:
        if self.dataset.validation is None:
            raise RuntimeError("`setup('fit')` must be called before `val_dataloader()`")
        return self.dataset.validation.to_dataloader(
            train=False, **self.dataloader_settings.to_dict_compatible_with(DataLoader)
        )

    def test_dataloader(self) -> DataLoader:
        if self.dataset.test is None:
            raise RuntimeError("`setup('test')` must be called before `test_dataloader()`")
        return self.dataset.test.to_dataloader(
            train=False, **self.dataloader_settings.to_dict_compatible_with(DataLoader)
        )

    def predict_dataloader(self) -> DataLoader:
        if self.dataset.predict is None:
            raise RuntimeError("`setup('predict')` must be called before `predict_dataloader()`")
        return self.dataset.predict.to_dataloader(
            train=False, **self.dataloader_settings.to_dict_compatible_with(DataLoader)
        )
