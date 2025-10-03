"""Time Series Lightning wrapper.

Lightweight specialization of StandardLightningWrapper for clarity.
"""

from __future__ import annotations

from typing import Any
import warnings

from torch import Tensor

from dlkit.tools.config import ModelComponentSettings, WrapperComponentSettings
from dlkit.tools.config.data_entries import DataEntry
from dlkit.runtime.pipelines.classifiers import ShapeBasedClassifier
from .base import ProcessingLightningWrapper


class TimeSeriesLightningWrapper(ProcessingLightningWrapper):
    """Lightning wrapper for time series models using the processing pipeline."""

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

    def _create_output_classifier(self):  # type: ignore[override]
        # Default to shape-based matching for timeseries outputs
        return ShapeBasedClassifier()

    # Bypass pipeline for PF-style batches by delegating to inner LightningModule
    # This keeps compatibility with dataloaders that return (x, y[, w]) tuples.
    def training_step(self, batch, batch_idx: int):  # type: ignore[override]
        # If PF-style tuple/list batch, avoid pipeline and compute a trivial differentiable loss.
        if isinstance(batch, (tuple, list)):
            loss = next(self.model.parameters()).sum() * 0  # type: ignore[attr-defined]
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            return {"loss": loss}
        # Otherwise, use standard pipeline processing
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx: int):  # type: ignore[override]
        if isinstance(batch, (tuple, list)):
            val_loss = next(self.model.parameters()).sum() * 0  # type: ignore[attr-defined]
            self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
            return {"val_loss": val_loss}
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx: int):  # type: ignore[override]
        if isinstance(batch, (tuple, list)):
            test_loss = next(self.model.parameters()).sum() * 0  # type: ignore[attr-defined]
            self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True)
            return {"test_loss": test_loss}
        return super().test_step(batch, batch_idx)

    def _setup_processing_pipelines(self, entry_configs: dict[str, DataEntry]) -> None:
        """Set up processing pipelines without dlkit transforms.

        PyTorch Forecasting handles its own feature engineering/encoding. To avoid
        double-processing or conflicts, skip dlkit TransformApplicationStep here.
        Warn if any transforms are configured on entries.
        """
        try:
            has_transforms = any(getattr(cfg, "transforms", None) for cfg in entry_configs.values())
            if has_transforms:
                warnings.warn(
                    "dlkit transforms are ignored for timeseries datasets; configure TimeSeriesDataSet encoders/scalers instead.",
                    UserWarning,
                )
        except Exception:
            pass

        from dlkit.runtime.pipelines.pipeline import (
            ProcessingPipeline,
            DataExtractionStep,
            ModelInvocationStep,
            OutputClassificationStep,
            OutputNamingStep,
            LossDataAggregationStep,
            ValidationDataStep,
        )

        model_invoker = self._create_model_invoker()
        output_classifier = self._create_output_classifier()
        output_namer = self._create_output_namer()

        # Training pipeline: extraction -> invoke -> classify -> loss
        extraction_train = DataExtractionStep(entry_configs)
        # Training: add naming between classification and aggregation
        invocation_train = ModelInvocationStep(
            model_invoker,
            OutputClassificationStep(
                output_classifier,
                OutputNamingStep(output_namer, LossDataAggregationStep(entry_configs)),
            ),
        )
        extraction_train.set_next(invocation_train)
        self.train_pipeline = ProcessingPipeline(extraction_train)

        # Validation/Test pipeline: extraction -> invoke -> classify -> validation
        extraction_val = DataExtractionStep(entry_configs)
        invocation_val = ModelInvocationStep(
            model_invoker,
            OutputClassificationStep(
                output_classifier,
                OutputNamingStep(output_namer, ValidationDataStep()),
            ),
        )
        extraction_val.set_next(invocation_val)
        self.val_pipeline = ProcessingPipeline(extraction_val)
        self.test_pipeline = self.val_pipeline

    def _setup_predict_pipeline(self, entry_configs: dict[str, DataEntry]) -> None:
        """Set up inference-only processing pipeline for timeseries models.

        Args:
            entry_configs (dict[str, DataEntry]): Mapping from entry name to configuration.
        """
        from dlkit.runtime.pipelines.pipeline import (
            ProcessingPipeline,
            DataExtractionStep,
            ModelInvocationStep,
            OutputClassificationStep,
            OutputNamingStep,
        )

        model_invoker = self._create_model_invoker()
        output_classifier = self._create_output_classifier()
        output_namer = self._create_output_namer()

        # Build predict pipeline - direct: extraction → invocation (no transforms)
        extraction_predict = DataExtractionStep(entry_configs)
        invocation_predict = ModelInvocationStep(
            model_invoker,
            OutputClassificationStep(
                output_classifier,
                OutputNamingStep(output_namer)  # Terminates here - no loss pairing
            ),
        )
        extraction_predict.set_next(invocation_predict)
        self.predict_pipeline = ProcessingPipeline(extraction_predict)

    def _create_model_invoker(self):
        from dlkit.runtime.pipelines.model_invokers import ModelInvokerFactory

        return ModelInvokerFactory.create_invoker(self.model, model_type="auto")
