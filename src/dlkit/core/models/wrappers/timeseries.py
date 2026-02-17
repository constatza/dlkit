"""Time Series Lightning wrapper.

Lightweight specialization of base wrapper with tuple batch handling.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from dlkit.tools.config import ModelComponentSettings, WrapperComponentSettings
from dlkit.tools.config.data_entries import DataEntry
from .base import ProcessingLightningWrapper


class TimeSeriesLightningWrapper(ProcessingLightningWrapper):
    """Lightning wrapper for time series models with direct processing methods.

    Handles both dict batches (using base class methods) and tuple batches
    (PyTorch Forecasting style) with special handling.
    """

    def __init__(
        self,
        *,
        settings: WrapperComponentSettings,
        model_settings: ModelComponentSettings,
        entry_configs: dict[str, DataEntry] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            settings=settings,
            model_settings=model_settings,
            entry_configs=entry_configs or {},
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    # Override steps to handle PyTorch Forecasting style tuple batches (x, y)
    # For dict/Batch batches, delegates to base class direct processing methods

    def training_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:
        """Training step handling PF-style tuple batches and dict/Batch batches."""
        if isinstance(batch, (tuple, list)):
            # PyTorch Forecasting format: (x, (y, weight))
            x, y_tuple = batch
            y = y_tuple[0] if isinstance(y_tuple, (tuple, list)) else y_tuple
            predictions = self.model(x)
            loss = self.loss_function(predictions, y.to(predictions.dtype))
            self._log_stage_outputs("train", loss)
            return {"loss": loss}
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:
        """Validation step handling PF-style tuple batches and dict/Batch batches."""
        if isinstance(batch, (tuple, list)):
            x, y_tuple = batch
            y = y_tuple[0] if isinstance(y_tuple, (tuple, list)) else y_tuple
            predictions = self.model(x)
            val_loss = self.loss_function(predictions, y.to(predictions.dtype))
            self._log_stage_outputs("val", val_loss)
            return {"val_loss": val_loss}
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:
        """Test step handling PF-style tuple batches and dict/Batch batches."""
        if isinstance(batch, (tuple, list)):
            x, y_tuple = batch
            y = y_tuple[0] if isinstance(y_tuple, (tuple, list)) else y_tuple
            predictions = self.model(x)
            test_loss = self.loss_function(predictions, y.to(predictions.dtype))
            self._log_stage_outputs("test", test_loss)
            return {"test_loss": test_loss}
        return super().test_step(batch, batch_idx)
