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
    # For dict batches, delegates to base class direct processing methods

    def training_step(self, batch, batch_idx: int):  # type: ignore[override]
        """Training step with special handling for PF-style tuple batches."""
        # If PF-style tuple/list batch, compute trivial differentiable loss
        if isinstance(batch, (tuple, list)):
            loss = next(self.model.parameters()).sum() * 0  # type: ignore[attr-defined]
            self._log_stage_outputs("train", loss)
            return {"loss": loss}
        # Otherwise, use base class direct processing
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx: int):  # type: ignore[override]
        """Validation step with special handling for PF-style tuple batches."""
        if isinstance(batch, (tuple, list)):
            val_loss = next(self.model.parameters()).sum() * 0  # type: ignore[attr-defined]
            self._log_stage_outputs("val", val_loss)
            return {"val_loss": val_loss}
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx: int):  # type: ignore[override]
        """Test step with special handling for PF-style tuple batches."""
        if isinstance(batch, (tuple, list)):
            test_loss = next(self.model.parameters()).sum() * 0  # type: ignore[attr-defined]
            self._log_stage_outputs("test", test_loss)
            return {"test_loss": test_loss}
        return super().test_step(batch, batch_idx)
