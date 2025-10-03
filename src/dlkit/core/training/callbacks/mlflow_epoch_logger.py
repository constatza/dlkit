"""Callback for logging metrics to MLflow using epoch numbers instead of steps."""

from typing import Any

from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger


class MLflowEpochLogger(Callback):
    """Callback that logs metrics to MLflow using epoch numbers as the x-axis.

    This callback intercepts metrics at the end of each training and validation epoch
    and logs them to MLflow with the current epoch number as the step parameter.
    This ensures that MLflow plots show epochs on the x-axis instead of global steps.

    Attributes:
        run_context: MLflow run context for logging metrics.
    """

    def __init__(self, run_context: Any):
        """Initialize the MLflow epoch logger.

        Args:
            run_context: MLflow run context with log_metrics method.
        """
        super().__init__()
        self.run_context = run_context

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log training metrics with epoch number at the end of training epoch.

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: Lightning module being trained.
        """
        self._log_metrics(trainer, prefix="train")

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log validation metrics with epoch number at the end of validation epoch.

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: Lightning module being trained.
        """
        self._log_metrics(trainer, prefix="val")

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log test metrics once at the end of all testing.

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: Lightning module being tested.
        """
        self._log_metrics(trainer, prefix="test")

    def _log_metrics(self, trainer: Trainer, prefix: str) -> None:
        """Extract and log metrics from trainer's callback metrics.

        Args:
            trainer: PyTorch Lightning trainer.
            prefix: Metric prefix to filter (e.g., "train", "val", "test").
        """
        try:
            # Get current epoch
            epoch = trainer.current_epoch

            # Get callback metrics (accumulated during the epoch)
            metrics = trainer.callback_metrics

            if not metrics:
                return

            # Filter metrics by prefix and prepare for logging
            filtered_metrics = {}
            for key, value in metrics.items():
                # Log metrics that match the prefix (or learning rate which is always included)
                if key.startswith(prefix) or key == "lr":
                    # Convert tensor to float if needed
                    try:
                        metric_value = float(value)
                        filtered_metrics[key] = metric_value
                    except (TypeError, ValueError):
                        logger.debug(f"Skipping non-numeric metric: {key}")

            # Log to MLflow with epoch as step
            if filtered_metrics:
                self.run_context.log_metrics(filtered_metrics, step=epoch)
                logger.debug(f"Logged {len(filtered_metrics)} metrics at epoch {epoch}")

        except Exception as e:
            logger.warning(f"Failed to log metrics with epoch logger: {e}")
