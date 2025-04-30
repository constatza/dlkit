from pathlib import Path
from typing import List

import pandas as pd
from lightning.pytorch import LightningDataModule
from pytorch_forecasting import TimeSeriesDataSet
from torch.utils.data import DataLoader

# Polars-only data reader
from dlkit.io.tables import read_table


class TimeSeriesDataModule(LightningDataModule):
    """LightningDataModule for multivariate autoregressive forecasting, using read_data()."""

    def __init__(
        self,
        data_file: str,
        sample_id_col: str,
        time_col: str,
        target_features: List[str],
        encoder_length: int,
        prediction_length: int,
    ) -> None:
        super().__init__()
        # Configuration (no magic values, batch sizes and num_workers removed)
        self.data_file = data_file
        self.sample_id_col = sample_id_col
        self.time_col = time_col
        self.target_features = target_features
        self.encoder_length = encoder_length
        self.prediction_length = prediction_length

        # datasets initialized in setup()
        self._df: pd.DataFrame | None = None
        self._train_ds: TimeSeriesDataSet | None = None
        self._val_ds: TimeSeriesDataSet | None = None
        self._test_ds: TimeSeriesDataSet | None = None
        self._predict_ds: TimeSeriesDataSet | None = None

    def prepare_data(self) -> None:
        """
        Called once per node (no state assignment). Only verify file existence here.
        """
        path = Path(self.data_file)
        if not path.is_file():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")

    def setup(self, stage: str | None = None) -> None:
        """
        Called on each GPU/process. Load via read_data(), then build datasets for fit/val/test/predict.
        """
        if stage in (None, "fit"):
            # 1. Load raw data (Polars → pandas)
            df_pl = read_table(self.data_file)
            df = (
                df_pl.select([self.sample_id_col, self.time_col] + self.target_features)
                .sort([self.sample_id_col, self.time_col])
                .to_pandas()
            )
            self._df = df

            # 2. Compute cutoff
            max_t = df[self.time_col].max()
            cutoff = max_t - self.prediction_length
            train_df = df[df[self.time_col] <= cutoff]
            self._train_ds = TimeSeriesDataSet(
                train_df,
                time_idx=self.time_col,
                target=self.target_features,
                group_ids=[self.sample_id_col],
                min_encoder_length=self.encoder_length,
                max_encoder_length=self.encoder_length,
                min_prediction_length=self.prediction_length,
                max_prediction_length=self.prediction_length,
                time_varying_known_reals=[self.time_col],
                time_varying_unknown_reals=self.target_features,
                add_encoder_length=True,
            )
            self._val_ds = TimeSeriesDataSet.from_dataset(
                self._train_ds,
                df,
                predict=True,
                stop_randomization=True,
            )

        # 4. Build test for 'test'
        if stage in (None, "test"):
            base_ds = self._train_ds or self._val_ds
            self._test_ds = TimeSeriesDataSet.from_dataset(
                base_ds,
                df,
                predict=True,
                stop_randomization=True,
            )

        # 5. Build predict for 'predict'
        if stage in (None, "predict"):
            self._predict_ds = TimeSeriesDataSet(
                df,
                time_idx=self.time_col,
                target=self.target_features,
                group_ids=[self.sample_id_col],
                min_encoder_length=self.encoder_length,
                max_encoder_length=self.encoder_length,
                min_prediction_length=0,
                max_prediction_length=self.prediction_length,
                time_varying_known_reals=[self.time_col],
                time_varying_unknown_reals=self.target_features,
                add_encoder_length=True,
                predict_mode=True,
            )

    def train_dataloader(self) -> DataLoader:
        if self._train_ds is None:
            raise RuntimeError(
                "`setup('fit')` must be called before `train_dataloader()`"
            )
        return self._train_ds.to_dataloader(train=True)

    def val_dataloader(self) -> DataLoader:
        if self._val_ds is None:
            raise RuntimeError(
                "`setup('fit')` must be called before `val_dataloader()`"
            )
        return self._val_ds.to_dataloader(train=False)

    def test_dataloader(self) -> DataLoader:
        if self._test_ds is None:
            raise RuntimeError(
                "`setup('test')` must be called before `test_dataloader()`"
            )
        return self._test_ds.to_dataloader(train=False)

    def predict_dataloader(self) -> DataLoader:
        if self._predict_ds is None:
            raise RuntimeError(
                "`setup('predict')` must be called before `predict_dataloader()`"
            )
        return self._predict_ds.to_dataloader(train=False)
