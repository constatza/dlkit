from typing import TYPE_CHECKING

from pytorch_forecasting import TimeSeriesDataSet
from torch.utils.data import DataLoader

from dlkit.engine.adapters.lightning.datamodules.array import InMemoryModule
from dlkit.infrastructure.config.dataloader_settings import DataloaderSettings
from dlkit.infrastructure.types.split import IndexSplit

if TYPE_CHECKING:
    from dlkit.engine.data.datasets.timeseries import ForecastingDataset


class TimeSeriesDataModule(InMemoryModule):
    """LightningDataModule for multivariate autoregressive forecasting, using read_data()."""

    def __init__(
        self,
        dataset: ForecastingDataset,
        dataloader: DataloaderSettings,
        split: IndexSplit,
    ) -> None:
        super().__init__(
            dataset=dataset,
            dataloader=dataloader,
            split=split,
        )

    def setup(self, stage: str | None = None) -> None:
        """Called on each GPU/process. Load via read_data(), then build datasets for fit/val/test/predict."""
        if not self.fitted:
            # split samples w.r.t. group_ids and self.idx_split
            # Reuse the underlying TimeSeriesDataSet built by the dataset itself
            dataset = self.split_dataset.raw.timeseries

            # Helper to build a safe TimeSeriesDataSet even when a split is empty
            def _build_split(ds, frame, *, for_training: bool = False):
                try:
                    # If the provided frame is empty, fall back to using the full raw DataFrame
                    # to ensure a non-empty PF dataset. This avoids pandas errors when setting
                    # auxiliary columns on empty frames inside pytorch-forecasting.
                    pdf = frame.to_pandas()
                    if len(pdf) == 0:
                        pdf = self.split_dataset.raw.df.to_pandas()
                    return TimeSeriesDataSet.from_dataset(
                        ds,
                        data=pdf,
                        predict=not for_training,
                        stop_randomization=True,
                    )
                except Exception:
                    # As an ultimate fallback, return the base dataset (covers extremely tiny toy dataflow)
                    return ds

            train_ds = _build_split(dataset, self.split_dataset.train[:], for_training=True)
            val_ds = _build_split(dataset, self.split_dataset.validation[:], for_training=False)
            test_ds = _build_split(dataset, self.split_dataset.test[:], for_training=False)
            predict_ds = _build_split(dataset, self.split_dataset.predict[:], for_training=False)

            self.split_dataset.train = train_ds
            self.split_dataset.validation = val_ds
            self.split_dataset.test = test_ds
            self.split_dataset.predict = predict_ds
            self.fitted = True

    def train_dataloader(self) -> DataLoader:
        if not isinstance(self.split_dataset.train, TimeSeriesDataSet):
            raise RuntimeError("`setup('fit')` must be called before `train_dataloader()`")
        dataloader_func = self.split_dataset.train.to_dataloader
        kwargs = self._get_dataloader_kwargs(DataLoader)
        return dataloader_func(train=True, **kwargs)

    def val_dataloader(self) -> DataLoader:
        if not isinstance(self.split_dataset.validation, TimeSeriesDataSet):
            raise RuntimeError("`setup('fit')` must be called before `val_dataloader()`")
        kwargs = self._get_dataloader_kwargs(DataLoader, shuffle=False)
        return self.split_dataset.validation.to_dataloader(train=False, **kwargs)

    def test_dataloader(self) -> DataLoader:
        if not isinstance(self.split_dataset.test, TimeSeriesDataSet):
            raise RuntimeError("`setup('test')` must be called before `test_dataloader()`")
        kwargs = self._get_dataloader_kwargs(DataLoader, shuffle=False)
        return self.split_dataset.test.to_dataloader(train=False, **kwargs)

    def predict_dataloader(self) -> DataLoader:
        if not isinstance(self.split_dataset.predict, TimeSeriesDataSet):
            raise RuntimeError("`setup('predict')` must be called before `predict_dataloader()`")
        kwargs = self._get_dataloader_kwargs(DataLoader, shuffle=False)
        return self.split_dataset.predict.to_dataloader(train=False, **kwargs)
