"""Time Series Lightning wrapper.

Lightweight specialization of StandardLightningWrapper with tuple batch handling
for PyTorch Forecasting-style (x, y) batches. TensorDict batches are handled
by the inherited step methods from the standard wrapper.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from dlkit.tools.config import ModelComponentSettings, WrapperComponentSettings
from dlkit.tools.config.data_entries import DataEntry
from .standard import StandardLightningWrapper


class TimeSeriesLightningWrapper(StandardLightningWrapper):
    """Lightning wrapper for time series models with PF-style tuple batch handling.

    Handles both TensorDict batches (via inherited StandardLightningWrapper step
    methods) and PyTorch Forecasting-style tuple batches (x, (y, weight)).
    """

    def __init__(
        self,
        *,
        settings: WrapperComponentSettings,
        model_settings: ModelComponentSettings,
        entry_configs: tuple[DataEntry, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise time series wrapper.

        Args:
            settings: Wrapper configuration settings.
            model_settings: Model configuration settings.
            entry_configs: Data entry configurations.
            **kwargs: Forwarded to StandardLightningWrapper.
        """
        super().__init__(
            settings=settings,
            model_settings=model_settings,
            entry_configs=entry_configs or (),
            **kwargs,
        )
        # Store loss_function for PF-style tuple batch handling
        self.loss_function = self._loss_computer._loss_fn  # type: ignore[attr-defined]

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor.

        Returns:
            Model output tensor.
        """
        return self.model(x)

    # Override steps to handle PyTorch Forecasting style tuple batches (x, y).
    # For TensorDict batches, delegates to the base class step methods.

    def training_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:
        """Training step handling PF-style tuple batches and TensorDict batches.

        Args:
            batch: Either a (x, y_tuple) PF-style tuple or a TensorDict.
            batch_idx: Index of the batch.

        Returns:
            Dictionary containing the training loss.
        """
        if isinstance(batch, (tuple, list)):
            x, y_tuple = batch
            y = y_tuple[0] if isinstance(y_tuple, (tuple, list)) else y_tuple
            predictions = self.model(x)
            loss = self.loss_function(predictions, y.to(predictions.dtype))
            self._log_stage_outputs("train", loss)
            return {"loss": loss}
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:
        """Validation step handling PF-style tuple batches and TensorDict batches.

        Args:
            batch: Either a (x, y_tuple) PF-style tuple or a TensorDict.
            batch_idx: Index of the batch.

        Returns:
            Dictionary containing validation loss.
        """
        if isinstance(batch, (tuple, list)):
            x, y_tuple = batch
            y = y_tuple[0] if isinstance(y_tuple, (tuple, list)) else y_tuple
            predictions = self.model(x)
            val_loss = self.loss_function(predictions, y.to(predictions.dtype))
            self._log_stage_outputs("val", val_loss)
            return {"val_loss": val_loss}
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:
        """Test step handling PF-style tuple batches and TensorDict batches.

        Args:
            batch: Either a (x, y_tuple) PF-style tuple or a TensorDict.
            batch_idx: Index of the batch.

        Returns:
            Dictionary containing test loss.
        """
        if isinstance(batch, (tuple, list)):
            x, y_tuple = batch
            y = y_tuple[0] if isinstance(y_tuple, (tuple, list)) else y_tuple
            predictions = self.model(x)
            test_loss = self.loss_function(predictions, y.to(predictions.dtype))
            self._log_stage_outputs("test", test_loss)
            return {"test_loss": test_loss}
        return super().test_step(batch, batch_idx)
